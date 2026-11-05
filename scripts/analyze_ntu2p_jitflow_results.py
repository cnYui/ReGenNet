"""JiT 流候选的 H-val 诊断产物：合并数组、受试者簇配对 bootstrap、全部推人接触/后退曲线、审查图数组。

复用 scripts/analyze_ntu2p_graphdiff_results.py 的 bootstrap 与推人曲线；只读 val，不改变预登记判定。
--methods 形如 FD2=FD2ref:endpoint J2中位数=J2:median J2样本=J2:sample0（名称=候选:读出）。
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

from scripts.analyze_ntu2p_graphdiff_results import paired_subject_bootstrap, plot_contacts


def _arrays_path(directory, candidate, readout):
    if readout.startswith("sample"):
        return directory / "{}_samples_arrays.pt".format(candidate), readout
    if candidate == "FD2ref":
        return directory / "FD2ref_arrays.pt", "model"
    return directory / "{}_{}_arrays.pt".format(candidate, readout), "model"


def load(directory, specs):
    data, methods = None, OrderedDict()
    for spec in specs:
        name, source = spec.split("=")
        candidate, readout = source.split(":")
        path, key = _arrays_path(directory, candidate, readout)
        loaded = torch.load(str(path), map_location="cpu")
        if data is None:
            data = OrderedDict((key_, loaded[key_]) for key_ in ("obs_xyz", "target_xyz", "actions", "meta"))
        else:
            if [row["sample_id"] for row in loaded["meta"]] != [row["sample_id"] for row in data["meta"]]:
                raise ValueError("{} 的样本顺序不一致".format(path))
            if not torch.equal(loaded["target_xyz"], data["target_xyz"]):
                raise ValueError("{} 的 target 不一致".format(path))
        methods[name] = loaded["methods"][key]
    data["methods"] = methods
    data["source"] = {"split": "val", "methods": list(specs)}
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--methods", nargs="+", required=True, help="首个方法必须命名为 FD2（bootstrap 参照）")
    parser.add_argument("--bootstrap_replicates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    opts = parser.parse_args()
    output = Path(opts.output_dir)
    if output.exists():
        raise ValueError("输出目录已存在，须使用新目录保留历史")
    data = load(Path(opts.results_dir), opts.methods)
    if next(iter(data["methods"])) != "FD2":
        raise ValueError("首个方法必须是 FD2 参照")
    output.mkdir(parents=True)
    torch.save(data, str(output / "review_arrays.pt"))
    result = paired_subject_bootstrap(data, opts.bootstrap_replicates, opts.seed)
    (output / "subject_bootstrap.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    plot_contacts(data, output / "all_pushing_contact_retreat.png")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
