"""Summarize saved, completed stage-B runs without refitting or reevaluation."""
import json
import numpy as np

from ecg_data import ROOT, save_json, sha256


def main():
    runs = [("E0 ResNet", "e0_resnet_omi_v1_run"), ("E1 ECG-FM", "e1_ecgfm_frozen_omi_v1"),
            ("E2 CPC query", "e2_ecgcpc_query_omi_v1"), ("E2 CPC mean control", "e2_ecgcpc_mean_omi_v1"),
            ("E3 Local", "e3_local_omi_v1")]
    result = {"scope": "internal_previously_explored_source_not_external_validation",
              "seed_summary": "mean_of_three_predeclared_individual_run_metrics_not_an_ensemble",
              "cad_probability": None, "experiments": {}}
    lines = ["# 阶段 B 固定内部评估汇总", "", "三个预定种子的指标均值不是模型集成的性能。工作点阈值来自选择集，校准参数来自独立校准集。所有实验使用相同的 3,589 条评估 ECG（228 条 OMI 阳性、3,394 位患者）。", "",
             "| 实验 | AUROC 均值 | AUPRC 均值 | Brier 均值 | 灵敏度范围 | 特异度范围 | 漏诊数范围 | 可训练参数 | 8 轮时间/种子 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, folder in runs:
        directory = ROOT / "artifacts" / folder
        metrics = json.loads((directory / "metrics.json").read_text())
        protocol = json.loads((directory / "protocol.json").read_text())
        if json.loads((directory / "status.json").read_text())["status"] != "calibrated_and_internally_evaluated":
            raise ValueError("all predeclared runs must be complete")
        seeds = []
        for seed in protocol["config"]["seeds"]:
            item = metrics["seeds"][str(seed)]
            resource = json.loads((directory / f"seed_{seed}_resources.json").read_text())
            target = item["targets"]["OMI"]
            seeds.append({"seed": seed, "selected_epoch": item["selected_epoch"],
                          **{k: target[k] for k in ("auroc", "auprc_average_precision", "brier", "sensitivity", "specificity", "ppv", "npv", "log_loss")},
                          "confusion": target["confusion"], "ci95": target["intervals"]["ci95"], "resources": resource})
        means = {k: float(np.mean([s[k] for s in seeds])) for k in ("auroc", "auprc_average_precision", "brier")}
        ranges = {k: [float(min(s[k] for s in seeds)), float(max(s[k] for s in seeds))] for k in ("sensitivity", "specificity")}
        fn = [min(s["confusion"]["fn"] for s in seeds), max(s["confusion"]["fn"] for s in seeds)]
        times = [s["resources"]["total_seconds"] for s in seeds]
        result["experiments"][name] = {"run_dir": str(directory), "metric_sha256": sha256(directory / "metrics.json"),
                "seeds": seeds, "mean_metrics": means, "ranges": ranges, "false_negative_range": fn,
                "all_seeds_meet_95_percent_sensitivity": all(s["sensitivity"] >= .95 for s in seeds)}
        lines.append(f"| {name} | {means['auroc']:.4f} | {means['auprc_average_precision']:.4f} | {means['brier']:.4f} | "
                     f"{100*ranges['sensitivity'][0]:.1f}–{100*ranges['sensitivity'][1]:.1f}% | "
                     f"{100*ranges['specificity'][0]:.1f}–{100*ranges['specificity'][1]:.1f}% | {fn[0]}–{fn[1]} | "
                     f"{seeds[0]['resources']['trainable_parameters']:,} | {min(times):.1f}–{max(times):.1f} s |")
    lines += ["", "训练时间包含各轮训练和选择集前向，冻结缓存提取与统计尺度拟合另计。总冻结骨干参数、缓存大小和提取时间见阶段 B 执行记录。完整患者聚类区间、混淆矩阵和亚组描述保存在各实验 metrics.json；本表范围表示三个种子的离散程度。", "",
              "## 患者配对差值", "", "下表是相同患者重采样、三个预定种子平均差值的 95% 区间，条件于这些种子和当前单中心队列，不包含外部队列不确定性。Brier 差值小于零有利于前一模型。", "",
              "| 比较 | AUROC 差值 95% CI | AUPRC 差值 95% CI | Brier 差值 95% CI |", "|---|---:|---:|---:|"]
    result["paired_comparisons"] = {}
    for path in sorted((ROOT / "artifacts/stage_b_summary").glob("paired_*.json")):
        pair = json.loads(path.read_text())
        intervals = pair["mean_over_predeclared_seeds_ci95"]
        result["paired_comparisons"][path.stem] = pair
        formatted = [f"[{intervals[k][0]:.5f}, {intervals[k][1]:.5f}]" for k in ("auroc", "auprc", "brier")]
        lines.append(f"| {path.stem.removeprefix('paired_')} | " + " | ".join(formatted) + " |")
    lines += ["", "当前结果不能作为临床排除 OMI 的依据，也不提供 CAD 总概率。E4 融合尚未训练，是否互补仍待同协议验证。", ""]
    out = ROOT / "artifacts/stage_b_summary"
    save_json(out / "results.json", result)
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved completed stage-B results.json and results.md")


if __name__ == "__main__":
    main()
