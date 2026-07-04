"""
compare.py - 对比报告生成

对比微调模型和基座模型的评测结果，生成 Markdown 报告。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_summary(path: str) -> Dict[str, Any]:
    """加载汇总结果"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_comparison_report(
    finetuned_summary: Dict[str, Any],
    baseline_summary: Dict[str, Any],
    output_path: str,
):
    """生成对比报告"""

    report = []

    # 标题
    report.append("# Model Comparison Report")
    report.append("")
    report.append("## Overview")
    report.append("")

    # 基本信息
    report.append("| Metric | Finetuned Model | Baseline Model | Improvement |")
    report.append("|--------|----------------|----------------|-------------|")

    # 样本数
    ft_samples = finetuned_summary.get("valid_samples", 0)
    bs_samples = baseline_summary.get("valid_samples", 0)
    report.append(f"| Valid Samples | {ft_samples} | {bs_samples} | - |")
    report.append("")

    # 可运行性指标
    report.append("## Execution Metrics")
    report.append("")
    report.append("| Metric | Finetuned | Baseline | Δ (pp) |")
    report.append("|--------|-----------|----------|--------|")

    for metric in ["xpr", "sar", "syr"]:
        ft_val = finetuned_summary.get(metric, 0.0)
        bs_val = baseline_summary.get(metric, 0.0)
        delta = (ft_val - bs_val) * 100  # 百分点差异

        metric_name = metric.upper()
        report.append(f"| {metric_name} | {ft_val:.2%} | {bs_val:.2%} | {delta:+.1f} |")

    report.append("")

    # 忠实度指标
    report.append("## Fidelity Metrics")
    report.append("")
    report.append("| Metric | Finetuned | Baseline | Δ (pp) |")
    report.append("|--------|-----------|----------|--------|")

    for metric, name in [
        ("pcr_avg", "PCR"),
        ("aar_avg", "AAR"),
        ("codebertscore_avg", "CodeBERTScore"),
    ]:
        ft_val = finetuned_summary.get(metric, 0.0)
        bs_val = baseline_summary.get(metric, 0.0)

        if "score" in metric:
            # CodeBERTScore 是 0-1 的浮点数
            delta = (ft_val - bs_val) * 100
            report.append(f"| {name} | {ft_val:.3f} | {bs_val:.3f} | {delta:+.3f} |")
        else:
            # 其他是百分比
            delta = (ft_val - bs_val) * 100
            report.append(f"| {name} | {ft_val:.2%} | {bs_val:.2%} | {delta:+.1f} |")

    report.append("")

    # 综合评分
    report.append("## Overall Scores")
    report.append("")
    report.append("| Score Type | Finetuned | Baseline | Δ (pp) |")
    report.append("|------------|-----------|----------|--------|")

    for score_name in ["execution_score", "fidelity_score", "overall_score"]:
        ft_val = finetuned_summary.get(score_name, 0.0)
        bs_val = baseline_summary.get(score_name, 0.0)
        delta = (ft_val - bs_val) * 100

        name = score_name.replace("_", " ").title()
        report.append(f"| {name} | {ft_val:.2%} | {bs_val:.2%} | {delta:+.1f} |")

    report.append("")

    # 关键发现
    report.append("## Key Findings")
    report.append("")

    overall_delta = (
        finetuned_summary.get("overall_score", 0.0)
        - baseline_summary.get("overall_score", 0.0)
    ) * 100

    if overall_delta >= 20:
        report.append(
            f"✅ **Significant improvement**: +{overall_delta:.1f} pp overall"
        )
    elif overall_delta >= 10:
        report.append(f"✅ **Moderate improvement**: +{overall_delta:.1f} pp overall")
    elif overall_delta >= 0:
        report.append(f"⚠️ **Slight improvement**: +{overall_delta:.1f} pp overall")
    else:
        report.append(f"❌ **Regression**: {overall_delta:.1f} pp overall")

    report.append("")

    # XPR 分析
    xpr_delta = (
        finetuned_summary.get("xpr", 0.0) - baseline_summary.get("xpr", 0.0)
    ) * 100

    if xpr_delta >= 20:
        report.append(
            f"✅ **XPR (Execution Pass Rate)**: Significantly improved (+{xpr_delta:.1f} pp)"
        )
    elif xpr_delta >= 10:
        report.append(
            f"✅ **XPR (Execution Pass Rate)**: Moderately improved (+{xpr_delta:.1f} pp)"
        )
    elif xpr_delta >= 0:
        report.append(
            f"⚠️ **XPR (Execution Pass Rate)**: Slightly improved (+{xpr_delta:.1f} pp)"
        )
    else:
        report.append(
            f"❌ **XPR (Execution Pass Rate)**: Decreased ({xpr_delta:.1f} pp)"
        )

    report.append("")

    # PCR 分析
    pcr_delta = (
        finetuned_summary.get("pcr_avg", 0.0) - baseline_summary.get("pcr_avg", 0.0)
    ) * 100

    if pcr_delta >= 20:
        report.append(
            f"✅ **PCR (Paper Component Coverage)**: Significantly improved (+{pcr_delta:.1f} pp)"
        )
    elif pcr_delta >= 10:
        report.append(
            f"✅ **PCR (Paper Component Coverage)**: Moderately improved (+{pcr_delta:.1f} pp)"
        )
    elif pcr_delta >= 0:
        report.append(
            f"⚠️ **PCR (Paper Component Coverage)**: Slightly improved (+{pcr_delta:.1f} pp)"
        )
    else:
        report.append(
            f"❌ **PCR (Paper Component Coverage)**: Decreased ({pcr_delta:.1f} pp)"
        )

    report.append("")

    # 结论
    report.append("## Conclusion")
    report.append("")

    if overall_delta >= 20:
        report.append(
            "The fine-tuned model demonstrates **significant improvements** over the baseline, "
            "particularly in generating executable and paper-faithful code. "
            "The fine-tuning process successfully enhanced the model's ability to reproduce "
            "academic papers into working implementations."
        )
    elif overall_delta >= 10:
        report.append(
            "The fine-tuned model shows **moderate improvements** over the baseline. "
            "While the gains are notable, there is room for further optimization in the training process."
        )
    elif overall_delta >= 0:
        report.append(
            "The fine-tuned model shows **slight improvements** over the baseline. "
            "The current fine-tuning approach may need refinement to achieve more substantial gains."
        )
    else:
        report.append(
            "⚠️ The fine-tuned model **underperforms** compared to the baseline. "
            "This suggests potential issues with the training data, hyperparameters, or evaluation setup. "
            "Further investigation is required."
        )

    report.append("")

    # 写入文件
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"[Compare] Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")

    parser.add_argument(
        "--finetuned",
        type=str,
        required=True,
        help="Path to fine-tuned model summary.json",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline model summary.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/eval/comparison.md",
        help="Output path for comparison report",
    )

    args = parser.parse_args()

    # 加载结果
    print(f"[Compare] Loading fine-tuned results from {args.finetuned}")
    finetuned_summary = load_summary(args.finetuned)

    print(f"[Compare] Loading baseline results from {args.baseline}")
    baseline_summary = load_summary(args.baseline)

    # 生成报告
    print(f"[Compare] Generating comparison report...")
    generate_comparison_report(
        finetuned_summary=finetuned_summary,
        baseline_summary=baseline_summary,
        output_path=args.output,
    )

    print(f"[Compare] Done!")


if __name__ == "__main__":
    main()
