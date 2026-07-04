# Model Comparison Report

## Overview

| Metric | Finetuned Model | Baseline Model | Improvement |
|--------|----------------|----------------|-------------|
| Valid Samples | 75 | 64 | - |

## Execution Metrics

| Metric | Finetuned | Baseline | Δ (pp) |
|--------|-----------|----------|--------|
| XPR | 14.67% | 9.38% | +5.3 |
| SAR | 14.67% | 9.38% | +5.3 |
| SYR | 34.67% | 39.06% | -4.4 |

## Fidelity Metrics

| Metric | Finetuned | Baseline | Δ (pp) |
|--------|-----------|----------|--------|
| PCR | 16.19% | 9.77% | +6.4 |
| AAR | 5.08% | 3.16% | +1.9 |
| CodeBERTScore | 0.444 | 0.380 | +6.365 |

## Overall Scores

| Score Type | Finetuned | Baseline | Δ (pp) |
|------------|-----------|----------|--------|
| Execution Score | 18.67% | 15.31% | +3.4 |
| Fidelity Score | 22.42% | 16.92% | +5.5 |
| Overall Score | 20.54% | 16.11% | +4.4 |

## Key Findings

⚠️ **Slight improvement**: +4.4 pp overall

⚠️ **XPR (Execution Pass Rate)**: Slightly improved (+5.3 pp)

⚠️ **PCR (Paper Component Coverage)**: Slightly improved (+6.4 pp)

## Conclusion

The fine-tuned model shows **slight improvements** over the baseline. The current fine-tuning approach may need refinement to achieve more substantial gains.
