# Validation Report: Paper 2 Macro Regimes
**Date:** 2026-06-14
**Status:** GREEN

## Summary of Validation Results
## 1. Data Integrity Checks
- Date index sorted: True
- Duplicate dates: False
- Largest date gap: 5 days 00:00:00
- Regime mapping mismatches: 0

## 2. Metric Recalculation (Lookahead Bias Check)
- Max diff in 5D Fwd Return: 9.97e-17
- Max diff in 20D Fwd Return: 9.97e-17
- Max diff in 60D Fwd Return: 9.97e-17
- Successfully computed Forward Volatility and True Forward MDD.

## 3. Non-Overlapping Sample Analysis
- 20D Non-overlapping sample size: 207
- 60D Non-overlapping sample size: 69

## 4. Block Bootstrap Confidence Intervals
- Block Bootstrap complete. Confidence intervals saved.

## 5. Outlier Sensitivity Analysis
- Raw Regime C Mean: 10.1026%
- Winsorized Regime C Mean: 9.5005%

## 6. Subperiod Stability
- Subperiod stability analysis saved.

## 7. Statistical Robustness (Newey-West / HAC)
- OLS with Newey-West (60-lags) Results:
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0230      0.007      3.455      0.001       0.010       0.036
D_B            0.0041      0.009      0.456      0.648      -0.014       0.022
D_C            0.0780      0.015      5.302      0.000       0.049       0.107
D_D            0.1159      0.026      4.481      0.000       0.065       0.167
==============================================================================

## 8. Realistic Data Lag Test
- Original Regime C 20D Return: 4.1967%
- 3-Day Lagged Regime C 20D Return: 4.0796%

## 9. Placebo Test (Block-Shuffle)
- Placebo p-value (Regime C): 0.0000

## 10. Final Conclusion
- Regime D confirmed unreliable with only 6 samples.


## Final Verdict
**GREEN: Conclusion is robust.** Regime C shows persistent outperformance across subperiods, survives Newey-West adjustments for overlap, and passes placebo testing. Regime D is correctly identified as insufficient for inference.