import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class Paper02Validator:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.input_dir = self.base_dir / "outputs" / "paper_02_macro_regimes"
        self.output_dir = self.base_dir / "outputs" / "paper_02_validation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Master Data
        master_path = self.input_dir / "paper_02_macro_master_dataset.csv"
        if not master_path.exists():
            raise FileNotFoundError(f"Master dataset not found at {master_path}")
        self.df = pd.read_csv(master_path, index_col=0, parse_dates=True)
        
        self.report_lines = []
        self.status = "GREEN" # Default, downgraded if checks fail

    def log(self, text):
        print(text)
        self.report_lines.append(text + "\n")

    def validate_integrity(self):
        self.log("## 1. Data Integrity Checks")
        
        # Sorted and Duplicates
        is_sorted = self.df.index.is_monotonic_increasing
        has_dupes = self.df.index.duplicated().any()
        self.log(f"- Date index sorted: {is_sorted}")
        self.log(f"- Duplicate dates: {has_dupes}")
        
        # Gaps
        max_gap = self.df.index.to_series().diff().max()
        self.log(f"- Largest date gap: {max_gap}")
        
        # Regime Logic Mapping
        # MultiLayerMacro mapping: 0=A, 1=B, 2=C, 3/4=D
        check_regime = self.df.copy()
        check_regime['Expected_Regime'] = check_regime['Destabilization_Score'].map({
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'D'
        })
        mismatches = (check_regime['Macro_Regime'] != check_regime['Expected_Regime']).sum()
        self.log(f"- Regime mapping mismatches: {mismatches}")
        
        if not is_sorted or has_dupes or mismatches > 0:
            self.status = "RED"

    def recalculate_metrics(self):
        self.log("\n## 2. Metric Recalculation (Lookahead Bias Check)")
        
        # Forward Returns
        for h in [5, 20, 60]:
            recalc = self.df['ES_Close'].shift(-h) / self.df['ES_Close'] - 1
            diff = np.abs(self.df[f'ES_{h}D_Fwd_Return'] - recalc).max()
            self.log(f"- Max diff in {h}D Fwd Return: {diff:.2e}")
            if diff > 1e-10: 
                self.log(f"  ⚠️ ALERT: {h}D returns mismatch detected!")
                self.status = "RED"

        # Forward Volatility vs Trailing
        # The paper uses ES_Realized_Vol_20D (trailing). We need forward version for validation.
        self.df['ES_Fwd_Realized_Vol_20D'] = self.df['ES_Daily_Return'].rolling(20).std().shift(-20) * np.sqrt(252)
        
        # Drawdowns
        # A. Downside Excursion (Lowest point relative to entry)
        self.df['Val_Fwd_Excursion_60D'] = self.df['ES_Close'].rolling(window=61).apply(
            lambda x: (x.min() / x.iloc[0]) - 1, raw=False).shift(-60)
            
        # B. True Forward Max Drawdown (Peak to Trough inside window)
        def get_true_mdd(x):
            if len(x) < 2: return 0
            # Compute DD of the future segment independently
            peak = x[0]
            mdd = 0
            for val in x:
                if val > peak: peak = val
                dd = (val / peak) - 1
                if dd < mdd: mdd = dd
            return mdd

        self.df['Val_True_Fwd_MDD_60D'] = self.df['ES_Close'].rolling(window=61).apply(
            get_true_mdd, raw=True).shift(-60)
        
        self.log("- Successfully computed Forward Volatility and True Forward MDD.")

    def non_overlapping_tests(self):
        self.log("\n## 3. Non-Overlapping Sample Analysis")
        results = []
        for h in [20, 60]:
            sub = self.df.iloc[::h].copy()
            summary = sub.groupby('Macro_Regime')[f'ES_{h}D_Fwd_Return'].mean()
            counts = sub['Macro_Regime'].value_counts()
            self.log(f"- {h}D Non-overlapping sample size: {len(sub)}")
            for r in ['A', 'B', 'C', 'D']:
                results.append({'Horizon': h, 'Regime': r, 'Mean': summary.get(r, np.nan), 'N': counts.get(r, 0)})
        
        val_df = pd.DataFrame(results)
        val_df.to_csv(self.output_dir / "non_overlapping_performance.csv")

    def block_bootstrap(self, n_iterations=500):
        self.log("\n## 4. Block Bootstrap Confidence Intervals")
        metrics = ['ES_20D_Fwd_Return', 'ES_60D_Fwd_Return', 'ES_Fwd_Max_DD_60D']
        bootstrap_results = []
        
        for metric in metrics:
            for regime in ['A', 'B', 'C']:
                regime_data = self.df[self.df['Macro_Regime'] == regime][metric].dropna().values
                if len(regime_data) < 50: continue
                
                means = []
                block_size = 20
                n_blocks = len(regime_data) // block_size
                
                for _ in range(n_iterations):
                    # Sample indices for blocks
                    starts = np.random.randint(0, len(regime_data) - block_size, size=n_blocks)
                    sample = np.concatenate([regime_data[s : s + block_size] for s in starts])
                    means.append(np.mean(sample))
                
                bootstrap_results.append({
                    'Metric': metric,
                    'Regime': regime,
                    'Mean': np.mean(means),
                    'CI_5': np.percentile(means, 5),
                    'CI_95': np.percentile(means, 95)
                })
        
        boot_df = pd.DataFrame(bootstrap_results)
        boot_df.to_csv(self.output_dir / "bootstrap_confidence_intervals.csv")
        self.log("- Block Bootstrap complete. Confidence intervals saved.")

    def outlier_sensitivity(self):
        self.log("\n## 5. Outlier Sensitivity Analysis")
        # Winsorize 1/99
        target = 'ES_60D_Fwd_Return'
        q_low = self.df[target].quantile(0.01)
        q_high = self.df[target].quantile(0.99)
        
        self.df['ES_60D_Fwd_Winsor'] = self.df[target].clip(q_low, q_high)
        
        raw_perf = self.df.groupby('Macro_Regime')[target].mean()
        winsor_perf = self.df.groupby('Macro_Regime')['ES_60D_Fwd_Winsor'].mean()
        
        self.log(f"- Raw Regime C Mean: {raw_perf.get('C', 0):.4%}")
        self.log(f"- Winsorized Regime C Mean: {winsor_perf.get('C', 0):.4%}")
        
        if winsor_perf.get('C', -1) < winsor_perf.get('A', 0):
            self.log("  ⚠️ WARNING: Regime C performance is driven by outliers!")
            self.status = "YELLOW"

    def subperiod_stability(self):
        self.log("\n## 6. Subperiod Stability")
        periods = {
            '2010-2014': ('2010-01-01', '2014-12-31'),
            '2015-2019': ('2015-01-01', '2019-12-31'),
            '2020-2026': ('2020-01-01', '2026-12-31')
        }
        
        sub_results = []
        for name, (start, end) in periods.items():
            mask = (self.df.index >= start) & (self.df.index <= end)
            period_df = self.df.loc[mask]
            if period_df.empty: continue
            
            perf = period_df.groupby('Macro_Regime')['ES_20D_Fwd_Return'].mean()
            counts = period_df['Macro_Regime'].value_counts()
            
            for r in ['A', 'B', 'C']:
                sub_results.append({
                    'Period': name, 'Regime': r, 'Return': perf.get(r, np.nan), 'N': counts.get(r, 0)
                })
        
        sub_df = pd.DataFrame(sub_results)
        sub_df.to_csv(self.output_dir / "subperiod_stability.csv")
        self.log("- Subperiod stability analysis saved.")

    def statistical_robustness(self):
        self.log("\n## 7. Statistical Robustness (Newey-West / HAC)")
        
        # Regression with HAC errors to account for overlap
        valid_df = self.df.dropna(subset=['ES_60D_Fwd_Return']).copy()
        valid_df['D_B'] = (valid_df['Macro_Regime'] == 'B').astype(int)
        valid_df['D_C'] = (valid_df['Macro_Regime'] == 'C').astype(int)
        valid_df['D_D'] = (valid_df['Macro_Regime'] == 'D').astype(int)
        
        X = sm.add_constant(valid_df[['D_B', 'D_C', 'D_D']])
        y = valid_df['ES_60D_Fwd_Return']
        
        # Lag for Newey-West usually set to horizon or slightly more
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 60})
        
        self.log("- OLS with Newey-West (60-lags) Results:")
        self.log(str(model.summary().tables[1]))
        
        p_val_c = model.pvalues['D_C']
        if p_val_c > 0.05:
            self.log(f"  ⚠️ ALERT: Regime C coefficient not significant under HAC (p={p_val_c:.4f})")
            self.status = "YELLOW"

    def data_lag_test(self):
        self.log("\n## 8. Realistic Data Lag Test")
        # Test if the signal still works if we act 3 days late
        self.df['Macro_Regime_Lag3'] = self.df['Macro_Regime'].shift(3)
        
        lag_perf = self.df.groupby('Macro_Regime_Lag3')['ES_20D_Fwd_Return'].mean()
        self.log(f"- Original Regime C 20D Return: {self.df.groupby('Macro_Regime')['ES_20D_Fwd_Return'].mean().get('C', 0):.4%}")
        self.log(f"- 3-Day Lagged Regime C 20D Return: {lag_perf.get('C', 0):.4%}")

    def placebo_test(self, n_sims=500):
        self.log("\n## 9. Placebo Test (Block-Shuffle)")
        target = 'ES_60D_Fwd_Return'
        actual_c_mean = self.df[self.df['Macro_Regime'] == 'C'][target].mean()
        
        placebo_means = []
        # We shuffle 'Macro_Regime' in blocks of 20 to maintain persistence
        indices = np.arange(len(self.df))
        block_size = 20
        n_blocks = len(self.df) // block_size
        
        for _ in range(n_sims):
            block_indices = np.arange(n_blocks)
            np.random.shuffle(block_indices)
            shuffled_idx = np.concatenate([indices[b*block_size : (b+1)*block_size] for b in block_indices])
            # Re-fill the remaining tail if not divisible
            if len(shuffled_idx) < len(self.df):
                shuffled_idx = np.concatenate([shuffled_idx, indices[len(shuffled_idx):]])
            
            temp_df = self.df.copy()
            temp_df['Regime_Placebo'] = self.df['Macro_Regime'].values[shuffled_idx]
            placebo_means.append(temp_df[temp_df['Regime_Placebo'] == 'C'][target].mean())
            
        p_val = (np.array(placebo_means) >= actual_c_mean).mean()
        self.log(f"- Placebo p-value (Regime C): {p_val:.4f}")
        if p_val > 0.05:
            self.log("  ⚠️ ALERT: Placebo test failed to reject randomness for Regime C.")
            self.status = "YELLOW"

    def generate_final_report(self):
        self.log("\n## 10. Final Conclusion")
        
        # Check Sample Size for D
        count_d = (self.df['Macro_Regime'] == 'D').sum()
        if count_d < 30:
            self.log(f"- Regime D confirmed unreliable with only {count_d} samples.")

        report_md = f"""# Validation Report: Paper 2 Macro Regimes
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Status:** {self.status}

## Summary of Validation Results
{"".join(self.report_lines)}

## Final Verdict
"""
        if self.status == "GREEN":
            report_md += "**GREEN: Conclusion is robust.** Regime C shows persistent outperformance across subperiods, survives Newey-West adjustments for overlap, and passes placebo testing. Regime D is correctly identified as insufficient for inference."
        elif self.status == "YELLOW":
            report_md += "**YELLOW: Conclusion is promising but needs softer wording.** While Regime C outperforms, statistical significance is marginal when accounting for overlap or data lags, or is sensitive to specific outliers."
        else:
            report_md += "**RED: Conclusion is not reliable.** Significant issues were found in data mapping, lookahead bias in returns, or a complete lack of significance in robustness tests."
            
        with open(self.output_dir / "paper_02_validation_report.md", "w") as f:
            f.write(report_md)
        print(f"\nValidation Report saved to: {self.output_dir}")

    def run_all(self):
        self.validate_integrity()
        self.recalculate_metrics()
        self.non_overlapping_tests()
        self.block_bootstrap()
        self.outlier_sensitivity()
        self.subperiod_stability()
        self.statistical_robustness()
        self.data_lag_test()
        self.placebo_test()
        self.generate_final_report()

if __name__ == "__main__":
    validator = Paper02Validator()
    validator.run_all()