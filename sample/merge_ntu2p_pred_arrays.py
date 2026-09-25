"""把多个 run 用 `eval/eval_ntu2p_v2.py --export_arrays` 导出的预测合并成一个多方法数组文件。

供 `sample/render_ntu2p_review_sheet.py` 与 `eval/analyze_ntu2p_naturalness.py` 在同一批案例上并排比较不同架构。
各输入必须来自同一 split 的同一批窗口：obs/target 与 sample_id 顺序逐项核对，不一致即报错，
避免把不同样本的预测拼到同一张审查图里。

用法：
    python sample/merge_ntu2p_pred_arrays.py --output merged.pt \
        --input A0=save/.../ntu2p_v2_A0_s0_10000/ema/val_arrays_000010000.pt \
        --input A6=save/.../ntu2p_v2_A6_s0_10000/ema/val_arrays_000010000.pt
"""

import argparse
from collections import OrderedDict

import torch


def _sample_ids(data):
    return [str(item["sample_id"]) for item in data["meta"]]


def merge(inputs, method_key="model"):
    merged = None
    methods = OrderedDict()
    for name, path in inputs:
        data = torch.load(path, map_location="cpu")
        if merged is None:
            merged = OrderedDict(
                [
                    ("obs_xyz", data["obs_xyz"]),
                    ("target_xyz", data["target_xyz"]),
                    ("actions", data["actions"]),
                    ("meta", data["meta"]),
                ]
            )
        else:
            if _sample_ids(data) != _sample_ids(merged):
                raise ValueError("{} 的样本顺序与首个输入不一致".format(path))
            for key in ("obs_xyz", "target_xyz"):
                if not torch.equal(data[key], merged[key]):
                    raise ValueError("{} 的 {} 与首个输入不一致".format(path, key))
        methods[name] = data["methods"][method_key]
    merged["methods"] = methods
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="名称=路径，可重复；首个为主方法")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method_key", default="model")
    args = parser.parse_args()
    inputs = []
    for item in args.input:
        if "=" not in item:
            raise ValueError("--input 须为 名称=路径，当前为 {}".format(item))
        name, path = item.split("=", 1)
        inputs.append((name, path))
    merged = merge(inputs, args.method_key)
    torch.save(merged, args.output)
    print("merged {} methods over {} samples -> {}".format(len(merged["methods"]), len(merged["meta"]), args.output))


if __name__ == "__main__":
    main()
