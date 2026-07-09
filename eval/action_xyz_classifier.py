import argparse
import json
import math
import os
import random
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_loaders.forecasting import NTULabelXYZCacheDataset, ntu_label_xyz_cache_collate
from data_loaders.forecasting.ntu_label import HANDSHAKING_LABEL, NUM_ACTIONS
from utils.fixseed import fixseed
from utils.ntu_smplx_2p_xyz import NTU_NUM_PERSONS, NTU_SMPLX_BODY_JOINTS, XYZ_COORD_DIM, check_ntu_xyz


MODEL_TYPE = "temporal_cnn_xyz_action_classifier"
GATE_THRESHOLDS = OrderedDict(
    [
        ("top1_acc", 0.80),
        ("top5_acc", 0.90),
        ("balanced_acc", 0.55),
        ("handshaking_acc", 0.80),
    ]
)
FIXED_OUTPUTS = (
    "args.json",
    "train_log.jsonl",
    "classifier_model.pt",
    "normalizer.pt",
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


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_ready(value), f, indent=2, sort_keys=False, ensure_ascii=False)


def _append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(_json_ready(record), sort_keys=False, ensure_ascii=False))
        f.write("\n")


def _clear_stage_outputs(save_dir):
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
            _clear_stage_outputs(args.save_dir)
    else:
        os.makedirs(args.save_dir)


def _ensure_finite(name, value):
    if torch.is_tensor(value):
        ok = bool(torch.isfinite(value).all().item())
    else:
        ok = bool(np.isfinite(np.asarray(value)).all())
    if not ok:
        raise ValueError("{} 存在非有限数值".format(name))


def _action_code(label):
    return "A{:03d}".format(int(label) + 1)


def _count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _group_count(channels):
    for group_count in (16, 8, 4, 2):
        if int(channels) % group_count == 0:
            return group_count
    return 1


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(ResidualTemporalBlock, self).__init__()
        groups = _group_count(hidden_dim)
        self.norm1 = nn.GroupNorm(groups, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.dropout(x)
        x = self.conv2(F.gelu(self.norm2(x)))
        x = self.dropout(x)
        return residual + x


class TemporalCNNXYZActionClassifier(nn.Module):
    def __init__(
        self,
        input_channels=NTU_NUM_PERSONS * NTU_SMPLX_BODY_JOINTS * XYZ_COORD_DIM,
        hidden_dim=128,
        num_blocks=3,
        dropout=0.1,
        num_classes=NUM_ACTIONS,
        pred_len=40,
    ):
        super(TemporalCNNXYZActionClassifier, self).__init__()
        self.input_channels = int(input_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.dropout = float(dropout)
        self.num_classes = int(num_classes)
        self.pred_len = int(pred_len)

        self.input_proj = nn.Conv1d(self.input_channels, self.hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualTemporalBlock(self.hidden_dim, self.dropout) for _ in range(self.num_blocks)]
        )
        self.out_norm = nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

    def config(self):
        return {
            "model_type": MODEL_TYPE,
            "input_channels": int(self.input_channels),
            "hidden_dim": int(self.hidden_dim),
            "num_blocks": int(self.num_blocks),
            "dropout": float(self.dropout),
            "num_classes": int(self.num_classes),
            "pred_len": int(self.pred_len),
        }

    def features(self, xyz):
        check_ntu_xyz("classifier xyz", xyz, seq_len=self.pred_len)
        batch_size = int(xyz.shape[0])
        x = xyz.reshape(batch_size, self.pred_len, self.input_channels)
        x = x.transpose(1, 2).contiguous()
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = F.gelu(self.out_norm(x))
        return x.mean(dim=-1)

    def forward(self, xyz, return_features=False):
        features = self.features(xyz)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits


def normalize_xyz_for_action(value, normalizer):
    return (value - normalizer["mean"]) / normalizer["std"].clamp_min(1e-6)


def extract_xyz_action_features(model, xyz, normalizer):
    normalized = normalize_xyz_for_action(xyz, normalizer)
    return model(normalized, return_features=True)


def _normalizer_to_device(normalizer, device):
    return {
        "mean": normalizer["mean"].to(device),
        "std": normalizer["std"].to(device),
        "count": int(normalizer["count"]),
    }


def load_xyz_action_classifier(path, device=None):
    if device is None:
        device = _device()
    state = torch.load(path, map_location=device)
    if state.get("model_type") != MODEL_TYPE:
        raise ValueError("unsupported classifier model_type: {}".format(state.get("model_type")))
    config = state["model_config"]
    model = TemporalCNNXYZActionClassifier(
        input_channels=config.get("input_channels", NTU_NUM_PERSONS * NTU_SMPLX_BODY_JOINTS * XYZ_COORD_DIM),
        hidden_dim=config.get("hidden_dim", 128),
        num_blocks=config.get("num_blocks", 3),
        dropout=config.get("dropout", 0.1),
        num_classes=config.get("num_classes", NUM_ACTIONS),
        pred_len=config.get("pred_len", 40),
    )
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    normalizer_path = state.get("normalizer_path")
    if normalizer_path is None or not os.path.exists(normalizer_path):
        normalizer_path = os.path.join(os.path.dirname(path), "normalizer.pt")
    normalizer = torch.load(normalizer_path, map_location=device)
    normalizer = _normalizer_to_device(normalizer, device)
    return model, normalizer, state


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


def _compute_normalizer(dataset, batch_size):
    loader = _build_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sum_x = torch.zeros(NTU_NUM_PERSONS, NTU_SMPLX_BODY_JOINTS, XYZ_COORD_DIM, dtype=torch.float64)
    sum_x2 = torch.zeros_like(sum_x)
    count = 0
    for batch in loader:
        xyz = batch["target_xyz"].double()
        _ensure_finite("normalizer target_xyz", xyz)
        sum_x = sum_x + xyz.sum(dim=(0, 1))
        sum_x2 = sum_x2 + (xyz * xyz).sum(dim=(0, 1))
        count += int(xyz.shape[0]) * int(xyz.shape[1])
    if count < 1:
        raise ValueError("normalizer count 为空")
    mean = sum_x / float(count)
    var = (sum_x2 / float(count)) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=0.0)).clamp_min(1e-6)
    return {
        "mean": mean.float().view(1, 1, NTU_NUM_PERSONS, NTU_SMPLX_BODY_JOINTS, XYZ_COORD_DIM),
        "std": std.float().view(1, 1, NTU_NUM_PERSONS, NTU_SMPLX_BODY_JOINTS, XYZ_COORD_DIM),
        "count": int(count),
    }


def _save_normalizer(args, normalizer):
    path = os.path.join(args.save_dir, "normalizer.pt")
    payload = dict(normalizer)
    payload["created_at"] = _utc_now()
    payload["train_xyz_cache"] = args.train_xyz_cache
    torch.save(payload, path)
    return path


def _label_counts(dataset):
    counts = [0 for _ in range(NUM_ACTIONS)]
    for action in dataset.actions.view(-1).tolist():
        counts[int(action)] += 1
    return counts


def _class_weights_from_counts(counts, device):
    weights = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    for idx, count in enumerate(counts):
        if int(count) > 0:
            weights[idx] = 1.0 / math.sqrt(float(count))
    positive = weights > 0
    if positive.any():
        weights[positive] = weights[positive] / weights[positive].mean().clamp_min(1e-12)
    return weights.to(device)


def _build_model(args):
    return TemporalCNNXYZActionClassifier(
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        num_classes=NUM_ACTIONS,
        pred_len=args.pred_len,
    )


def _save_checkpoint(args, model, step, normalizer_path, real_test_metrics=None):
    path = os.path.join(args.save_dir, "classifier_model.pt")
    torch.save(
        {
            "model_type": MODEL_TYPE,
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "num_classes": NUM_ACTIONS,
            "step": int(step),
            "seed": int(args.seed),
            "train_xyz_cache": args.train_xyz_cache,
            "test_xyz_cache": args.test_xyz_cache,
            "normalizer_path": normalizer_path,
            "real_test_metrics": real_test_metrics,
            "created_at": _utc_now(),
        },
        path,
    )
    return path


def _topk(logits, k=5):
    probs = torch.softmax(logits, dim=1)
    top_probs, top_labels = torch.topk(probs, k=min(int(k), int(logits.shape[1])), dim=1)
    return top_labels, top_probs


def _metrics_from_predictions(loss_sum, total, top1_correct, top5_correct, class_correct, class_count):
    if int(total) < 1:
        raise ValueError("eval total 为空")
    per_class_acc = []
    valid_acc = []
    for idx in range(NUM_ACTIONS):
        count = int(class_count[idx])
        if count > 0:
            acc = float(class_correct[idx]) / float(count)
            per_class_acc.append(acc)
            valid_acc.append(acc)
        else:
            per_class_acc.append(None)
    handshaking_acc = per_class_acc[HANDSHAKING_LABEL]
    top1_acc = float(top1_correct) / float(total)
    top5_acc = float(top5_correct) / float(total)
    balanced_acc = sum(valid_acc) / float(len(valid_acc)) if valid_acc else 0.0
    gate_pass = (
        top1_acc >= GATE_THRESHOLDS["top1_acc"]
        and top5_acc >= GATE_THRESHOLDS["top5_acc"]
        and balanced_acc >= GATE_THRESHOLDS["balanced_acc"]
        and handshaking_acc is not None
        and handshaking_acc >= GATE_THRESHOLDS["handshaking_acc"]
    )
    return OrderedDict(
        [
            ("top1_acc", top1_acc),
            ("top5_acc", top5_acc),
            ("balanced_acc", balanced_acc),
            ("per_class_acc", per_class_acc),
            ("per_class_count", [int(item) for item in class_count]),
            ("handshaking_acc", handshaking_acc),
            ("top1_random", 1.0 / float(NUM_ACTIONS)),
            ("top5_random", 5.0 / float(NUM_ACTIONS)),
            ("classifier_gate_pass", bool(gate_pass)),
            ("gate_thresholds", GATE_THRESHOLDS),
            ("loss", float(loss_sum) / float(total)),
            ("num_samples", int(total)),
        ]
    )


def _make_prediction_record(index, meta, target, pred, top_labels, top_probs):
    return OrderedDict(
        [
            ("index", int(index)),
            ("sample_id", meta.get("sample_id")),
            ("start", int(meta.get("start", -1))),
            ("length", int(meta.get("length", -1))),
            ("target_label", int(target)),
            ("target_action_code", _action_code(target)),
            ("predicted_label", int(pred)),
            ("predicted_action_code", _action_code(pred)),
            ("top5_labels", [int(item) for item in top_labels]),
            ("top5_action_codes", [_action_code(item) for item in top_labels]),
            ("top5_probs", [float(item) for item in top_probs]),
        ]
    )


@torch.no_grad()
def evaluate_classifier(model, loader, normalizer, device, prediction_path=None, confusion_path=None):
    if prediction_path and os.path.exists(prediction_path):
        os.remove(prediction_path)
    model.eval()
    loss_sum = 0.0
    total = 0
    top1_correct = 0
    top5_correct = 0
    class_correct = np.zeros(NUM_ACTIONS, dtype=np.int64)
    class_count = np.zeros(NUM_ACTIONS, dtype=np.int64)
    confusion = np.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=np.int64)
    for batch in loader:
        xyz = batch["target_xyz"].to(device)
        labels = batch["action"].view(-1).to(device)
        logits, _ = extract_xyz_action_features(model, xyz, normalizer)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        top_labels, top_probs = _topk(logits, k=5)
        preds = top_labels[:, 0]
        batch_size = int(labels.shape[0])
        loss_sum += float(loss.detach().cpu().item())
        total += batch_size
        top1_correct += int((preds == labels).sum().detach().cpu().item())
        top5_correct += int((top_labels == labels.unsqueeze(1)).any(dim=1).sum().detach().cpu().item())
        labels_cpu = labels.detach().cpu().numpy()
        preds_cpu = preds.detach().cpu().numpy()
        top_labels_cpu = top_labels.detach().cpu().numpy()
        top_probs_cpu = top_probs.detach().cpu().numpy()
        for row_idx in range(batch_size):
            target = int(labels_cpu[row_idx])
            pred = int(preds_cpu[row_idx])
            class_count[target] += 1
            if pred == target:
                class_correct[target] += 1
            confusion[target, pred] += 1
            if prediction_path:
                record = _make_prediction_record(
                    total - batch_size + row_idx,
                    batch["meta"][row_idx],
                    target,
                    pred,
                    top_labels_cpu[row_idx],
                    top_probs_cpu[row_idx],
                )
                _append_jsonl(prediction_path, record)
    if confusion_path:
        np.save(confusion_path, confusion)
    return _metrics_from_predictions(loss_sum, total, top1_correct, top5_correct, class_correct, class_count)


def _next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _train_step(model, optimizer, batch, normalizer, class_weights, args, device):
    model.train()
    xyz = batch["target_xyz"].to(device)
    labels = batch["action"].view(-1).to(device)
    logits, _ = extract_xyz_action_features(model, xyz, normalizer)
    loss = F.cross_entropy(logits, labels, weight=class_weights)
    _ensure_finite("train loss", loss)
    optimizer.zero_grad()
    loss.backward()
    if float(args.clip_grad_norm) > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
    optimizer.step()
    top1_acc = (logits.argmax(dim=1) == labels).float().mean()
    return {
        "train_loss": float(loss.detach().cpu().item()),
        "train_top1_acc": float(top1_acc.detach().cpu().item()),
    }


def _save_args(args, device, model, train_dataset, test_dataset, normalizer):
    payload = OrderedDict()
    for key, value in sorted(vars(args).items()):
        payload[key] = value
    payload["device"] = str(device)
    payload["num_params"] = int(_count_parameters(model))
    payload["model_config"] = model.config()
    payload["train_count"] = int(len(train_dataset))
    payload["test_count"] = int(len(test_dataset))
    payload["train_label_counts"] = _label_counts(train_dataset)
    payload["test_label_counts"] = _label_counts(test_dataset)
    payload["normalizer_count"] = int(normalizer["count"])
    payload["created_at"] = _utc_now()
    _write_json(os.path.join(args.save_dir, "args.json"), payload)


def run_classifier_training(args):
    if int(args.obs_len) + int(args.pred_len) != int(args.window_len):
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    fixseed(args.seed)
    _prepare_save_dir(args)
    device = _device()
    train_dataset = _build_dataset(args.train_xyz_cache, max_samples=args.max_samples)
    test_dataset = _build_dataset(args.test_xyz_cache, max_samples=args.eval_max_samples)
    train_loader = _build_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = _build_loader(test_dataset, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    normalizer = _compute_normalizer(train_dataset, args.eval_batch_size)
    normalizer_path = _save_normalizer(args, normalizer)
    normalizer = _normalizer_to_device(normalizer, device)
    model = _build_model(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = _class_weights_from_counts(_label_counts(train_dataset), device)
    _save_args(args, device, model, train_dataset, test_dataset, normalizer)
    print(
        "Training xyz action classifier: params={} device={} train={} test={}".format(
            _count_parameters(model), device, len(train_dataset), len(test_dataset)
        )
    )
    iterator = iter(train_loader)
    latest_checkpoint = None
    for step in range(1, int(args.num_steps) + 1):
        batch, iterator = _next_batch(train_loader, iterator)
        train_metrics = _train_step(model, optimizer, batch, normalizer, class_weights, args, device)
        eval_metrics = None
        if int(args.eval_interval) > 0 and step % int(args.eval_interval) == 0:
            eval_metrics = evaluate_classifier(model, test_loader, normalizer, device)
        if step == 1 or step % int(args.log_interval) == 0:
            message = "step[{}]: train_loss[{:.6f}] train_top1[{:.4f}]".format(
                step, train_metrics["train_loss"], train_metrics["train_top1_acc"]
            )
            if eval_metrics is not None:
                message += " eval_top1[{:.4f}] eval_balanced[{:.4f}]".format(
                    eval_metrics["top1_acc"], eval_metrics["balanced_acc"]
                )
            print(message)
        if step % int(args.save_interval) == 0 or step == int(args.num_steps):
            latest_checkpoint = _save_checkpoint(args, model, step, normalizer_path)
        record = OrderedDict([("step", int(step))])
        record.update(train_metrics)
        if eval_metrics is not None:
            record["eval_top1_acc"] = eval_metrics["top1_acc"]
            record["eval_top5_acc"] = eval_metrics["top5_acc"]
            record["eval_balanced_acc"] = eval_metrics["balanced_acc"]
            record["eval_handshaking_acc"] = eval_metrics["handshaking_acc"]
            record["eval_classifier_gate_pass"] = eval_metrics["classifier_gate_pass"]
        if latest_checkpoint is not None:
            record["checkpoint"] = latest_checkpoint
        record["lr"] = float(args.lr)
        record["seed"] = int(args.seed)
        _append_jsonl(os.path.join(args.save_dir, "train_log.jsonl"), record)
    real_metrics = evaluate_classifier(
        model,
        test_loader,
        normalizer,
        device,
        prediction_path=os.path.join(args.save_dir, "real_test_predictions.jsonl"),
        confusion_path=os.path.join(args.save_dir, "confusion_matrix.npy"),
    )
    _write_json(os.path.join(args.save_dir, "real_test_metrics.json"), real_metrics)
    latest_checkpoint = _save_checkpoint(args, model, int(args.num_steps), normalizer_path, real_test_metrics=real_metrics)
    print(
        "Finished xyz action classifier. checkpoint={} top1={:.4f} top5={:.4f} balanced={:.4f} handshaking={} gate_pass={}".format(
            latest_checkpoint,
            real_metrics["top1_acc"],
            real_metrics["top5_acc"],
            real_metrics["balanced_acc"],
            real_metrics["handshaking_acc"],
            real_metrics["classifier_gate_pass"],
        )
    )
    return {"checkpoint": latest_checkpoint, "real_test_metrics": real_metrics}


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_xyz_cache", default="results/forecasting/ntu120_label/xyz_cache_len60_o20_p40/train_xyz.pt")
    parser.add_argument("--test_xyz_cache", default="results/forecasting/ntu120_label/xyz_cache_len60_o20_p40/test_xyz.pt")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
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
