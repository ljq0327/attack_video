#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CONFIGS = [
    "configs/model_zoo_ntu60_cs.json",
    "configs/model_zoo_ntu60_cv.json",
    "configs/model_zoo_ntu120_cs.json",
    "configs/model_zoo_ntu120_cset.json",
    "configs/model_zoo_pkummd_cs.json",
    "configs/model_zoo_pkummd_cv.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        "Run all configured recognizers on the selected annotation CSVs and check whether all predictions are correct."
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="Config JSON files to evaluate. Defaults to the six selected-set model_zoo configs.",
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated model subset, for example: dvanet,tbcnet,i3d.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/verify_selected",
        help="Directory for per-model predictions, merged CSVs, mistakes and summary files.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Use >0 for a quick smoke test. Default 0 evaluates all rows.",
    )
    parser.add_argument(
        "--reuse_predictions",
        action="store_true",
        help="Reuse existing per-model prediction CSVs under output_dir when present.",
    )
    parser.add_argument(
        "--views",
        default=None,
        help="Optional camera filter shared by all configs, e.g. 1 or 1,2,3. Default evaluates every row in each CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic transforms/model behavior where applicable.",
    )
    parser.add_argument(
        "--fail_on_not_100",
        action="store_true",
        help="Exit with code 1 if any config is not 100 percent correct for all evaluated rows.",
    )
    return parser.parse_args()


def config_key(config_path):
    stem = Path(config_path).stem
    return stem.replace("model_zoo_", "", 1)


def count_csv_rows(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def load_runtime():
    import pandas as pd
    import torch

    from select_common_correct import (
        build_cfg,
        build_loader,
        load_model_zoo,
        merge_predictions,
        parse_int_list,
        predict_one_model,
        prediction_path,
        resolve_path,
        set_seed,
    )

    return SimpleNamespace(
        pd=pd,
        torch=torch,
        build_cfg=build_cfg,
        build_loader=build_loader,
        load_model_zoo=load_model_zoo,
        merge_predictions=merge_predictions,
        parse_int_list=parse_int_list,
        predict_one_model=predict_one_model,
        prediction_path=prediction_path,
        resolve_path=resolve_path,
        set_seed=set_seed,
    )


def resolve_config_list(configs):
    resolved = []
    for config in configs:
        path = Path(config)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved.append(path)
    return resolved


def select_model_names(model_zoo, models_arg):
    model_names = list(model_zoo["models"].keys())
    if not models_arg:
        return model_names

    requested = [name.strip() for name in models_arg.split(",") if name.strip()]
    missing = [name for name in requested if name not in model_zoo["models"]]
    if missing:
        raise ValueError(f"Models not found in config: {missing}")
    return requested


def make_loader_args(args):
    return SimpleNamespace(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        views=args.views,
    )


def make_predict_args(args):
    return SimpleNamespace(
        gpu=args.gpu,
        max_batches=args.max_batches,
    )


def summarize_predictions(merged, model_names, dataset_rows, max_batches):
    evaluated_rows = int(len(merged))
    model_stats = []
    for model_name in model_names:
        correct_column = f"{model_name}_correct"
        correct = int(merged[correct_column].sum())
        model_stats.append(
            {
                "model": model_name,
                "correct": correct,
                "total": evaluated_rows,
                "accuracy": correct / max(evaluated_rows, 1),
            }
        )

    all_correct = int(merged["all_correct"].sum()) if evaluated_rows else 0
    return {
        "dataset_rows": int(dataset_rows),
        "evaluated_rows": evaluated_rows,
        "max_batches": int(max_batches),
        "all_models_correct": all_correct,
        "all_models_accuracy": all_correct / max(evaluated_rows, 1),
        "is_100_percent": evaluated_rows > 0 and all_correct == evaluated_rows,
        "model_stats": model_stats,
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def ensure_runtime_temp_dirs():
    tmp_dir = PROJECT_ROOT / "results" / "tmp"
    mpl_dir = tmp_dir / f"matplotlib-{os.getuid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)


def run_one_config(config_path, args, runtime):
    model_zoo = runtime.load_model_zoo(str(config_path))
    base_cfg = runtime.build_cfg(model_zoo)
    model_names = select_model_names(model_zoo, args.models)
    dataset_rows = count_csv_rows(base_cfg.test_annotations)

    run_name = config_key(config_path)
    run_dir = Path(args.output_dir) / run_name
    prediction_dir = run_dir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    loader_args = make_loader_args(args)
    loader, dataset = runtime.build_loader(base_cfg, loader_args)
    predict_args = make_predict_args(args)

    print("=" * 80)
    print(f"config: {config_path}")
    print(f"dataset: {model_zoo['dataset']}")
    print(f"training_mode: {model_zoo.get('training_mode', 'CS')}")
    print(f"annotations: {base_cfg.test_annotations}")
    print(f"csv_rows: {dataset_rows}")
    print(f"loaded_rows: {len(dataset)}")
    print(f"models: {', '.join(model_names)}")
    print(f"output: {run_dir}")
    print("=" * 80)

    for model_name in model_names:
        output_csv = runtime.prediction_path(str(prediction_dir), model_name)
        if args.reuse_predictions and os.path.exists(output_csv):
            print(f"reuse: {output_csv}")
            continue

        stat = runtime.predict_one_model(
            model_name,
            model_zoo["models"][model_name],
            loader,
            base_cfg,
            model_zoo,
            runtime.torch.device("cuda" if runtime.torch.cuda.is_available() else "cpu"),
            output_csv,
            predict_args,
        )
        print(f"{run_name}/{model_name}: acc={stat['acc']:.6f} ({stat['correct']}/{stat['total']})")

    merged = runtime.merge_predictions(model_names, str(prediction_dir))
    summary = summarize_predictions(merged, model_names, dataset_rows, args.max_batches)

    merged_path = run_dir / "merged_predictions.csv"
    mistakes_path = run_dir / "mistakes.csv"
    summary_path = run_dir / "summary.json"

    merged.to_csv(merged_path, index=False)
    merged[~merged["all_correct"]].to_csv(mistakes_path, index=False)

    run_summary = {
        "config": str(config_path),
        "dataset": model_zoo["dataset"],
        "training_mode": model_zoo.get("training_mode", "CS"),
        "test_annotations": base_cfg.test_annotations,
        "models": model_names,
        **summary,
        "outputs": {
            "prediction_dir": str(prediction_dir),
            "merged_predictions": str(merged_path),
            "mistakes": str(mistakes_path),
            "summary": str(summary_path),
        },
    }
    write_json(summary_path, run_summary)

    print(f"{run_name}: all_models_acc={summary['all_models_accuracy']:.6f} "
          f"({summary['all_models_correct']}/{summary['evaluated_rows']})")
    print(f"mistakes: {mistakes_path}")
    return {"name": run_name, **run_summary}


def write_master_outputs(output_dir, summaries):
    rows = []
    for summary in summaries:
        row = {
            "name": summary["name"],
            "dataset": summary["dataset"],
            "training_mode": summary["training_mode"],
            "test_annotations": summary["test_annotations"],
            "dataset_rows": summary["dataset_rows"],
            "evaluated_rows": summary["evaluated_rows"],
            "all_models_correct": summary["all_models_correct"],
            "all_models_accuracy": summary["all_models_accuracy"],
            "is_100_percent": summary["is_100_percent"],
        }
        for model_stat in summary["model_stats"]:
            row[f"{model_stat['model']}_accuracy"] = model_stat["accuracy"]
            row[f"{model_stat['model']}_correct"] = model_stat["correct"]
        rows.append(row)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_dir / "summary.json"
    write_json(summary_json, summaries)

    summary_csv = output_dir / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    front = [
        "name",
        "dataset",
        "training_mode",
        "test_annotations",
        "dataset_rows",
        "evaluated_rows",
        "all_models_correct",
        "all_models_accuracy",
        "is_100_percent",
    ]
    fieldnames = front + [name for name in fieldnames if name not in front]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return summary_json, summary_csv


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("PYTHONPATH", "")
    os.environ.setdefault("TORCH_HOME", "/mnt/data/ljq/torch_cache")
    ensure_runtime_temp_dirs()

    runtime = load_runtime()
    runtime.set_seed(args.seed)

    device = runtime.torch.device("cuda" if runtime.torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda_count: {runtime.torch.cuda.device_count()}")

    summaries = []
    for config_path in resolve_config_list(args.configs):
        summaries.append(run_one_config(config_path, args, runtime))

    summary_json, summary_csv = write_master_outputs(args.output_dir, summaries)
    all_ok = all(summary["is_100_percent"] for summary in summaries)

    print("=" * 80)
    print(f"summary_json: {summary_json}")
    print(f"summary_csv: {summary_csv}")
    print(f"all_100_percent: {all_ok}")
    print("=" * 80)

    if args.fail_on_not_100 and not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
