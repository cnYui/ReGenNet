"""JiT 流复现（10000 step × 3 seed）的跨 seed 汇总与预登记判定（设计 4.3"采纳后"一节）。

读取 run_ntu2p_jitflow.py 在各 seed tag 下写的 summary.json，按行（候选:读出）求均值 ± 样本标准差；
判定对 3 seed 的均值行套用与筛选相同的规则，参照为 FD2ref:endpoint 的 3 seed 均值。
"""

import argparse
import json
import statistics
from collections import OrderedDict
from pathlib import Path

from scripts.run_ntu2p_jitflow import BASE, decide


NUMERIC_SKIP = ("decision",)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="subjval")
    parser.add_argument("--tags", nargs="+", required=True, help="每个 seed 一个 tag，例如 rep10000_s0 rep10000_s1 rep10000_s2")
    parser.add_argument("--output", required=True)
    opts = parser.parse_args()
    per_seed = [json.loads((BASE / "ntu2p_jitflow" / opts.protocol / tag / "summary.json").read_text())["rows"] for tag in opts.tags]
    names = [name for name in per_seed[0] if all(name in rows for rows in per_seed)]
    mean_rows, std_rows = OrderedDict(), OrderedDict()
    for name in names:
        keys = [key for key, value in per_seed[0][name].items() if key not in NUMERIC_SKIP and isinstance(value, (int, float))]
        mean_rows[name] = OrderedDict((key, statistics.mean(rows[name][key] for rows in per_seed)) for key in keys)
        std_rows[name] = OrderedDict((key, statistics.stdev(rows[name][key] for rows in per_seed) if len(per_seed) > 1 else 0.0) for key in keys)
    ref = mean_rows["FD2ref:endpoint"]
    result = OrderedDict([("tags", opts.tags), ("reference", "FD2ref:endpoint（3 seed 均值）"), ("rows", OrderedDict())])
    for name in names:
        sample = name.endswith(":sample")
        row = OrderedDict(mean_rows[name])
        row["delta_mpjpe_pct"] = 100.0 * (row["mpjpe"] / ref["mpjpe"] - 1.0)
        row["per_seed_delta_mpjpe_pct"] = [100.0 * (rows[name]["mpjpe"] / rows["FD2ref:endpoint"]["mpjpe"] - 1.0) for rows in per_seed]
        result["rows"][name] = OrderedDict([("mean", row), ("std", std_rows[name]), ("decision_on_mean", decide(row, ref, sample))])
    Path(opts.output).parent.mkdir(parents=True, exist_ok=True)
    Path(opts.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    header = "|行|xyz_mse|xyz_mae|mpjpe|Δmpjpe%（逐 seed）|jerk|穿地|excess_antic|漏接触|提分|交互|未过护栏|"
    lines = ["# JiT 流 3 seed 复现汇总", "", "均值 ± 样本标准差；参照 FD2ref:endpoint 的 3 seed 均值。", "", header,
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in result["rows"].items():
        mean, std, decision = item["mean"], item["std"], item["decision_on_mean"]
        lines.append("|{}|{:.5f}±{:.5f}|{:.5f}±{:.5f}|{:.5f}±{:.5f}|{:+.2f}（{}）|{:.2f}|{:.4f}|{:.4f}|{:.1f}|{}|{}|{}|".format(
            name, mean["xyz_mse"], std["xyz_mse"], mean["xyz_mae"], std["xyz_mae"], mean["mpjpe"], std["mpjpe"], mean["delta_mpjpe_pct"],
            "/".join("{:+.2f}".format(v) for v in mean["per_seed_delta_mpjpe_pct"]), mean["jerk_body"], mean["foot_penetration_ratio"],
            mean["primary_excess_antic"], mean["missed_contact_total"], "是" if decision["score_adopt"] else "否",
            "是" if decision["interaction_adopt"] else "否", ",".join(decision["failed_guards"]) or "—"))
    Path(opts.output).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
