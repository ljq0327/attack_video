import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from attack_utils import (
    build_recognizer, # 根据模型名字和config里的checkpoint，把一个可用于攻击/评估的模型加载出来
    build_metrics_summary, # 把整个实验的所有信息打包成一个完整字典,summary.json
    ensure_runtime_temp_dirs, # 重定向运行时临时目录与matplotlib缓存目录到项目文件夹
    evaluate_transfer, # 迁移总控
    generate_examples, # 生成对抗样本：逐batch攻击、保存.pt、收集元信息，返回metadata_rows
    parse_float, # 将输入的数值或带有除法的字符串（16/255）安全地解析为浮点数
    print_final_summary, # 打印最终实验总结
    print_run_header, # 打印当前实验的运行信息
    save_json, # 保存全局汇总JSON
    save_sample_summary, # 调用save_csv将组装好的总表落盘保存为sample_summary.csv
    select_eval_models, # 选择评估模型, 解析和校验用户输入的待评测黑盒模型列表，如果没传该参数，则默认评测模型配置文件中的所有模型
)
from dataloader import build_cfg, build_loader, load_model_zoo, set_seed
from project_paths import PROJECT_ROOT, add_third_party_paths


add_third_party_paths()


def parse_args():
    parser = argparse.ArgumentParser("Attack selected clean samples and evaluate transferability.")
    parser.add_argument("--config", required=True, help="Model zoo config, e.g. configs/model_zoo_ntu120_cs.json.")
    parser.add_argument("--source_model", default="dvanet", help="White-box source model name in the config.")
    parser.add_argument("--attack_method", default="tt", help="Attack name: bim, pgd, tt, tt_ti.")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--views", default=None, help="Optional camera filter, e.g. 1 or 1,2,3.")
    parser.add_argument("--max_samples", type=int, default=0, help="Use>0 for a quick subset smoke test.")
    parser.add_argument("--seed", type=int, default=42)
    # 控制对抗扰动的最大强度上限
    parser.add_argument("--epsilon", type=parse_float, default=16/255)
    # 优化迭代的步数，决定了算法进行多少轮梯度更新
    parser.add_argument("--steps", type=int, default=10, help='Multi-step or One-step.')
    # 卷积核长度，控制核矩阵的尺寸大小（时序分支总数-输入32帧设置的15，输入16帧视频改为7）
    parser.add_argument("--kernlen", type=int, default=7)
    # 时序移动类型，邻近帧平移（adj）、远距离平移（remote）或随机平移（random）
    parser.add_argument("--move_type", default="adj", choices=["adj", "remote", "random"])
    # 卷积核权重分布模式，高斯平滑（gaussian）、线性（linear）或均匀分布（uniform）
    parser.add_argument("--kernel_mode", default="gaussian", choices=["gaussian", "linear", "uniform"])
    # 动量更新
    parser.add_argument("--momentum", action="store_true")
    # 动量Momentum项的衰减系数
    parser.add_argument("--delay", type=float, default=1.0)
    # 定向攻击开关，默认为非定向攻击（只要模型猜错即可），若开启此选项则会强制模型将样本预测为指定的某个特定目标类别
    parser.add_argument("--targeted", action="store_true", help="Run a targeted attack instead of untargeted.")
    # 定向攻击时的目标标签选择策略
    # next为取原始真实标签的下一个类别作为攻击目标标签
    # least_likely为用干净样本送入模型预测，输出概率最小的类别作为目标标签
    # closest为排除真实标签后，源模型干净样本logit最高的类别（最容易混淆的类别）
    parser.add_argument(
        "--target_strategy",
        default="next",
        choices=["next", "closest", "least_likely"],
        help="Target rule: next, closest (clean-logit runner-up), or least_likely.",
    )
    # 时序平移批次大小，解决显存爆炸
    parser.add_argument("--shift_batch_size", type=int, default=5, help="Number of temporal shifts evaluated per gradient mini-batch.")
    # TI平移不变性
    parser.add_argument("--ti_kernlen", type=int, default=15)
    # TI攻击中高斯核的标准差倍数
    parser.add_argument("--ti_nsig", type=float, default=3.0)
    # PGD攻击中每一步的更新步长，不传值，触发自动计算
    parser.add_argument("--pgd_step_size", type=parse_float, default=None, help="PGD step size in pixel [0,1] scale. Defaults to epsilon / steps.")
    # PGD是否在扰动球内做随机初始化
    parser.add_argument("--pgd_random_start", action="store_true", help="Use random start inside the L-inf ball for PGD.")
    # 实验结果输出的根目录路径
    parser.add_argument("--output_root", default="/media/abc/ware/ljq/attack_results")
    # 保存对抗样本张量时的数据类型
    parser.add_argument("--save_dtype", default="float32", choices=["float32", "float16"])
    # 额外再存一份反归一化到[0,1]浮点数版本视频
    parser.add_argument("--save_video_01", action="store_true", help="Also save denormalized [0,1] tensors in every pt file. This roughly doubles disk usage.")
    # 跳过黑盒迁移评测阶段，仅生成并保存对抗样本
    parser.add_argument("--skip_eval", action="store_true", help="Only generate and save clean/adv examples.")
    # 指定评测哪些特定的黑盒模型（用逗号隔开）
    parser.add_argument("--eval_models", default=None, help="Optional comma-separated eval model subset. Default evaluates all models in the config.")
    return parser.parse_args()

# 读取实验配置并构建攻击数据加载器
def load_run(args):
    # 读取json文件（含有视频路径、动作类别数量、模型配置等）
    model_zoo = load_model_zoo(args.config)
    num_actions = int(model_zoo.get("num_actions", 0))
    # 把model_zoo加工成运行配置对象
    base_cfg = build_cfg(model_zoo)
    # 把一堆零散变量打包成一个对象
    loader_args = SimpleNamespace(batch_size=args.batch_size, num_workers=args.num_workers, views=args.views)
    # 构建数据集Dataset和批量迭代DataLoader
    # loader：for循环迭代取batch做PGD；dataset：用于索引单样本、读取样本元信息
    loader, dataset = build_loader(base_cfg, loader_args)
    return model_zoo, base_cfg, loader, dataset, num_actions

def main():
    # 解析命令行参数
    args = parse_args()

    # 准备运行环境
    os.chdir(PROJECT_ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ensure_runtime_temp_dirs()
    set_seed(args.seed)

    # 加载配置和数据
    model_zoo, base_cfg, loader, dataset, num_actions = load_run(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建输出目录（包含对抗样本、攻击准确率等）
    attack_name = args.attack_method.lower()
    attack_type = (
        f"targeted_{args.target_strategy}"
        if args.targeted
        else "untargeted"
    )
    eps_name = f"eps{args.epsilon * 255:g}_255"
    config_name = Path(args.config).stem.replace("model_zoo_", "", 1)
    run_name = f"{config_name}_{args.source_model}_{args.attack_method}_{eps_name}_{args.steps}_{attack_type}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 打印实验信息
    print_run_header(args, model_zoo, base_cfg, dataset, device, run_dir, attack_name, num_actions)

    # 加载白盒源模型
    source_model = build_recognizer(args.source_model, model_zoo["models"][args.source_model], base_cfg, model_zoo, device)
    
    # 生成对抗样本（内部执行PGD/TT攻击循环......）
    metadata_rows = generate_examples(source_model, loader, device, run_dir, args, num_actions)
    
    # 释放白盒模型显存
    del source_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 评估迁移效果（命令行参数加了--skip_eval，只生成对抗样本，不做评估）
    eval_models = []
    summary_rows = []
    if not args.skip_eval:
        eval_models = select_eval_models(model_zoo, args.eval_models)
        clean_predictions, adv_predictions, summary_rows = evaluate_transfer(
            model_zoo, base_cfg, loader, metadata_rows, eval_models, device, args,)
    else:
        clean_predictions = []
        adv_predictions = []

    # 保存逐样本明细CSV
    sample_summary_path = save_sample_summary(run_dir, metadata_rows, clean_predictions, adv_predictions, eval_models)
    # 生成整体指标字典
    metrics_summary = build_metrics_summary(args, model_zoo, base_cfg, run_dir, metadata_rows, summary_rows, attack_name, num_actions)
    metrics_summary_path = run_dir / "metrics_summary.json"
    # 保存整体指标JSON
    save_json(metrics_summary_path, metrics_summary)
    # 在终端打印总结
    print_final_summary(sample_summary_path, metrics_summary_path, metadata_rows, summary_rows, args.targeted)

if __name__ == "__main__":
    main()
