import torch
import torch.nn as nn

from project_paths import PYTORCHVIDEO_ROOT
 
class SlowFastClassifier(nn.Module): 
    """ 
    输入:  [B, V, T, C, H, W] (推荐 T=16)
    输出:  {"per_view_logits": [B, V, num_actions]} 
    """ 
    def __init__(self, num_actions, model_name="slowfast_r50", pretrained=True): 
        super().__init__() 
        self.num_actions = num_actions 
 
        # 显式指定 head_pool_kernel_sizes 适配 16 帧变体配置 (Slow=4, Fast=16)
        self.model = torch.hub.load( 
            str(PYTORCHVIDEO_ROOT), 
            model_name, 
            pretrained=pretrained, 
            source="local", 
            head_pool_kernel_sizes=((4, 7, 7), (16, 7, 7)), 
        ) 
 
        head = self.model.blocks[-1].proj 
        self.feature_dim = head.in_features 
        self.model.blocks[-1].proj = nn.Linear(self.feature_dim, num_actions) 
 
    def forward_single_view(self, x): 
        """ 
        x: [B, T, C, H, W] (T=16) 
        """ 
        # 1. 调整为 PyTorchVideo 要求的 [B, C, T, H, W] 
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # 2. 在当前 device 上显式构造 Slow/Fast 双通路输入
        t = x.shape[2]
        slow_t = max(1, t // 4)
        slow_idx = torch.linspace(0, t - 1, slow_t, device=x.device).long()
        fast_idx = torch.linspace(0, t - 1, t, device=x.device).long()
        slow_pathway = torch.index_select(x, 2, slow_idx)
        fast_pathway = torch.index_select(x, 2, fast_idx)
        x_list = [slow_pathway, fast_pathway]

        # 3. 前向传播
        logits = self.model(x_list)
        return logits
 
    def forward(self, inputs): 
        """ 
        inputs: [B, V, T, C, H, W] 
        """ 
        B, V, T, C, H, W = inputs.shape 
        per_view_logits_list = [] 
 
        # 采用清晰的循环逐视角处理，逻辑稳健且便于调试 
        for v_idx in range(V): 
            v_input = inputs[:, v_idx]  # [B, T, C, H, W] 
            v_logits = self.forward_single_view(v_input) 
            per_view_logits_list.append(v_logits) 
 
        per_view_logits = torch.stack(per_view_logits_list, dim=1)  # [B, V, num_actions] 
 
        return { 
            "per_view_logits": per_view_logits, 
        } 
 
if __name__ == "__main__": 
    model = SlowFastClassifier(num_actions=10, pretrained=False).eval() 
    inp = torch.randn(2, 4, 16, 3, 224, 224) 
    out = model(inp) 
    print("SlowFast Loop output shape:", out["per_view_logits"].shape)
