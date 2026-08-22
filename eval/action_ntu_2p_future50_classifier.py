import argparse
import copy
import json
import os
from collections import OrderedDict
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_loaders.forecasting import NTULabelXYZCacheDataset, ntu_label_xyz_cache_collate
from eval.action_xyz_classifier import (
    GATE_THRESHOLDS,
    MODEL_TYPE,
    NUM_ACTIONS,
    TemporalCNNXYZActionClassifier,
    _class_weights_from_counts,
    _compute_normalizer,
    _count_parameters,
    _label_counts,
    _normalizer_to_device,
    _train_step,
    evaluate_classifier,
)
from utils.fixseed import fixseed


FIXED_OUTPUTS = (
    "args.json",
    "train_log.jsonl",
    "classifier_model.pt",
    "best_classifier_model.pt",
    "normalizer.pt",
    "val_metrics.json",
    "real_test_metrics.json",
    "real_test_predictions.jsonl",
    "confusion_matrix.npy",
)


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.write("\n")


def _append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=False, ensure_ascii=False))
        f.write("\n")


def _clear_outputs(save_dir):
    for filename in FIXED_OUTPUTS:
        path = os.path.join(save_dir, filename)
        if os.path.isfile(path):
            os.remove(path)


def _prepare_save_dir(args):
    if os.path.exists(args.save_dir):
        has_files = len(os.listdir(args.save_dir)) > 0
        if has_files and not args.overwrite:
            raise FileExistsError("save_dir 已存在，使用 --overwrite: {}".format(args.save_dir))
        if has_files and args.overwrite:
            _clear_outputs(args.save_dir)
    else:
        os.makedirs(args.save_dir)


def _build_dataset(path, max_samples=-1):
    return NTULabelXYZCacheDataset(path, max_samples=max_samples)


def _build_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=ntu_label_xyz_cache_collate,
        drop_last=False,
    )


def _build_model(args):
    return TemporalCNNXYZActionClassifier(
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        num_classes=NUM_ACTIONS,
        pred_len=args.pred_len,
    )


def _save_normalizer(args, normalizer):
    path = os.path.join(args.save_dir, "normalizer.pt")
    payload = dict(normalizer)
    payload["created_at"] = _utc_now()
    payload["train_xyz_cache"] = args.train_xyz_cache
    payload["val_xyz_cache"] = args.val_xyz_cache
    payload["test_xyz_cache"] = args.test_xyz_cache
    torch.save(payload, path)
    return path


def _gate_pass(metrics):
    for key, threshold in GATE_THRESHOLDS.items():
        value = metrics.get(key)
        if value is None or float(value) < float(threshold):
            return False
    return True


def _score_val(metrics):
    return (
        float(metrics["balanced_acc"]),
        float(metrics["top1_acc"]),
        float(metrics["top5_acc"]),
        -float(metrics["loss"]),
    )


def _save_checkpoint(args, model, step, normalizer_path, val_metrics=None, real_test_metrics=None, filename="classifier_model.pt"):
    path = os.path.join(args.save_dir, filename)
    torch.save(
        {
            "model_type": MODEL_TYPE,
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "num_classes": NUM_ACTIONS,
            "step": int(step),
            "seed": int(args.seed),
            "train_xyz_cache": args.train_xyz_cache,
            "val_xyz_cache": args.val_xyz_cache,
            "test_xyz_cache": args.test_xyz_cache,
            "normalizer_path": normalizer_path,
            "val_metrics": val_metrics,
            "real_test_metrics": real_test_metrics,
            "protocol": {
                "window_len": int(args.window_len),
                "obs_len": int(args.obs_len),
                "pred_len": int(args.pred_len),
                "selection": "best_internal_val_balanced_top1_top5_loss",
            },
            "created_at": _utc_now(),
        },
        path,
    )
    return path


def _save_args(args, device, model, train_dataset, val_dataset, test_dataset, normalizer):
    payload = OrderedDict()
    for key, value in sorted(vars(args).items()):
        payload[key] = value
    payload["device"] = str(device)
    payload["num_params"] = int(_count_parameters(model))
    payload["model_config"] = model.config()
    payload["train_count"] = int(len(train_dataset))
    payload["val_count"] = int(len(val_dataset))
    payload["test_count"] = int(len(test_dataset))
    payload["train_label_counts"] = _label_counts(train_dataset)
    payload["val_label_counts"] = _label_counts(val_dataset)
    payload["test_label_counts"] = _label_counts(test_dataset)
    payload["normalizer_count"] = int(normalizer["count"])
    payload["gate_thresholds"] = GATE_THRESHOLDS
    payload["created_at"] = _utc_now()
    _write_json(os.path.join(args.save_dir, "args.json"), payload)


def run_classifier_training(args):
    if int(args.obs_len) + int(args.pred_len) != int(args.window_len):
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    if int(args.window_len) != 60 or int(args.obs_len) != 10 or int(args.pred_len) != 50:
        raise ValueError("本入口固定 window_len=60, obs_len=10, pred_len=50")
    fixseed(args.seed)
    _prepare_save_dir(args)
    device = _device()

    train_dataset = _build_dataset(args.train_xyz_cache, max_samples=args.max_samples)
    val_dataset = _build_dataset(args.val_xyz_cache, max_samples=args.eval_max_samples)
    test_dataset = _build_dataset(args.test_xyz_cache, max_samples=args.eval_max_samples)
    train_loader = _build_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = _build_loader(val_dataset, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = _build_loader(test_dataset, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    normalizer = _compute_normalizer(train_dataset, args.eval_batch_size)
    normalizer_path = _save_normalizer(args, normalizer)
    normalizer = _normalizer_to_device(normalizer, device)
    model = _build_model(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = _class_weights_from_counts(_label_counts(train_dataset), device)
    _save_args(args, device, model, train_dataset, val_dataset, test_dataset, normalizer)

    print(
        "Training NTU2P future50 classifier: params={} device={} train={} val={} test={}".format(
            _count_parameters(model), device, len(train_dataset), len(val_dataset), len(test_dataset)
        )
    )

    iterator = iter(train_loader)
    best_score = None
    best_state = None
    best_step = 0
    best_val_metrics = None
    for step in range(1, int(args.num_steps) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        train_metrics = _train_step(model, optimizer, batch, normalizer, class_weights, args, device)
        record = OrderedDict([("step", int(step))])
        record.update(train_metrics)
        if step == 1 or step % int(args.eval_interval) == 0 or step == int(args.num_steps):
            val_metrics = evaluate_classifier(model, val_loader, normalizer, device)
            score = _score_val(val_metrics)
            if best_score is None or score > best_score:
                best_score = score
                best_step = int(step)
                best_val_metrics = val_metrics
                best_state = copy.deepcopy(model.state_dict())
            record["val_top1_acc"] = val_metrics["top1_acc"]
            record["val_top5_acc"] = val_metrics["top5_acc"]
            record["val_balanced_acc"] = val_metrics["balanced_acc"]
            record["val_handshaking_acc"] = val_metrics["handshaking_acc"]
            record["val_classifier_gate_pass"] = _gate_pass(val_metrics)
            print(
                "step[{}]: train_loss[{:.6f}] val_top1[{:.4f}] val_balanced[{:.4f}] best_step[{}]".format(
                    step,
                    train_metrics["train_loss"],
                    val_metrics["top1_acc"],
                    val_metrics["balanced_acc"],
                    best_step,
                )
            )
        elif step % int(args.log_interval) == 0:
            print("step[{}]: train_loss[{:.6f}] train_top1[{:.4f}]".format(step, train_metrics["train_loss"], train_metrics["train_top1_acc"]))
        record["lr"] = float(args.lr)
        record["seed"] = int(args.seed)
        _append_jsonl(os.path.join(args.save_dir, "train_log.jsonl"), record)

    if best_state is None:
        raise RuntimeError("未产生 best_state")
    model.load_state_dict(best_state)
    _write_json(os.path.join(args.save_dir, "val_metrics.json"), best_val_metrics)
    best_checkpoint = _save_checkpoint(
        args,
        model,
        best_step,
        normalizer_path,
        val_metrics=best_val_metrics,
        filename="best_classifier_model.pt",
    )
    real_test_metrics = evaluate_classifier(
        model,
        test_loader,
        normalizer,
        device,
        prediction_path=os.path.join(args.save_dir, "real_test_predictions.jsonl"),
        confusion_path=os.path.join(args.save_dir, "confusion_matrix.npy"),
    )
    real_test_metrics["classifier_gate_pass"] = _gate_pass(real_test_metrics)
    _write_json(os.path.join(args.save_dir, "real_test_metrics.json"), real_test_metrics)
    final_checkpoint = _save_checkpoint(
        args,
        model,
        best_step,
        normalizer_path,
        val_metrics=best_val_metrics,
        real_test_metrics=real_test_metrics,
        filename="classifier_model.pt",
    )
    print(
        "Finished NTU2P future50 classifier. best_checkpoint={} final_checkpoint={} test_top1={:.4f} test_balanced={:.4f} gate_pass={}".format(
            best_checkpoint,
            final_checkpoint,
            real_test_metrics["top1_acc"],
            real_test_metrics["balanced_acc"],
            real_test_metrics["classifier_gate_pass"],
        )
    )
    return {
        "best_checkpoint": best_checkpoint,
        "checkpoint": final_checkpoint,
        "best_val_metrics": best_val_metrics,
        "real_test_metrics": real_test_metrics,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_xyz_cache", required=True)
    parser.add_argument("--val_xyz_cache", required=True)
    parser.add_argument("--test_xyz_cache", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--eval_max_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    run_classifier_training(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
