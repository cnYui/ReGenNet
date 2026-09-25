"""从 train xyz 序列缓存构建 A3 检索库并保存（库文件只含索引与规范系参数，键在加载时由缓存重算）。

用法：
    PYTHONPATH=. python scripts/build_ntu2p_retrieval_bank.py --device cpu
输出：<out_dir>/train_retrieval_bank.pt 与同名 .json（配置、manifest_hash、各动作窗口/序列/受试者数、τ 初值）。
"""

import argparse
import datetime
import json
import os
import time
from collections import OrderedDict

import torch

from data_loaders.forecasting.ntu2p_retrieval_bank import (
    DEFAULT_BANK_DIR,
    DEFAULT_K,
    DEFAULT_KEY_JOINTS,
    NTU2PRetrievalBank,
)
from data_loaders.forecasting.ntu2p_xyz_seq_cache import DEFAULT_CACHE_DIR, NTU2PXYZSeqCache


DEFAULT_MANIFEST = "results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json"
BANK_FILE = "train_retrieval_bank.pt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--manifest_path", default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", default=DEFAULT_BANK_DIR)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--key_joints", type=int, default=DEFAULT_KEY_JOINTS)
    parser.add_argument("--velocity_weight", type=float, default=0.0)
    parser.add_argument("--tau_k", type=int, default=DEFAULT_K, help="τ 初值取 train 查询 top-k 距离中位数的 k")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_threads", type=int, default=3)
    return parser.parse_args()


def per_action_summary(bank):
    rows = OrderedDict()
    for action in torch.unique(bank.action).tolist():
        mask = bank.action == action
        rows[int(action)] = OrderedDict(
            [
                ("windows", int(mask.sum().item())),
                ("sequences", int(torch.unique(bank.seq_index[mask]).numel())),
                ("performers", int(torch.unique(bank.performer[mask]).numel())),
            ]
        )
    return rows


def main():
    args = parse_args()
    torch.set_num_threads(int(args.num_threads))
    begin = time.time()
    train = NTU2PXYZSeqCache.load(args.cache_dir, "train", manifest_path=args.manifest_path, device=args.device)
    bank = NTU2PRetrievalBank.build(
        train, stride=args.stride, key_joints=args.key_joints, velocity_weight=args.velocity_weight, device=args.device
    )
    bank.config["median_topk_dist"] = bank.median_neighbor_distance(k=args.tau_k)
    bank.config["median_topk_k"] = int(args.tau_k)
    bank.config["manifest_path"] = args.manifest_path
    bank.config["cache_dir"] = args.cache_dir
    bank.config["created_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, BANK_FILE)
    bank.save(path)
    # 回读一次：确认键校验和与缓存一致。
    NTU2PRetrievalBank.load(path, train, device=args.device)
    summary = OrderedDict(
        [
            ("config", bank.config),
            ("key_dim", int(bank.keys_centered.shape[1])),
            ("file_size_mb", os.path.getsize(path) / 2.0 ** 20),
            ("build_seconds", time.time() - begin),
            ("per_action", per_action_summary(bank)),
        ]
    )
    with open(path + ".json", "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps({key: summary[key] for key in ("config", "key_dim", "file_size_mb", "build_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
