import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from pathlib import Path
from scipy import stats
from datetime import datetime
from dotenv import load_dotenv

# Import existing logic from the folder modules
from liquidity_analysis import get_liquidity_report
from credit_spread_analysis import get_credit_spread_report
from volatility_analysis import get_volatility_report
from positioning_analysis import fetch_cot_positioning

# Load .env for any other environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

class MultiLayerMacroResearch:
    def __init__(self):
        # Set absolute output directory inside the MultiLayerMacro folder
        self.output_dir = Path(__file__).resolve().parent / "outputs" / "paper_02_macro_regimes"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis Parameters
        self.horizons = [5, 20, 60]
        self.min_obs_threshold = 30 # Warning for regimes with fewer observations

    def fetch_equity_data(self, start_date='2010-01-01'):
        """Fetches ES (Primary) and NQ (Secondary) data."""
        print("Fetching Equity Data (ES and NQ)...")

        def safe_download(ticker, fallback=None):
            try:
                data = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
                if (data is None or data.empty) and fallback:
                    print(f"{ticker} not available, falling back to {fallback}.")
                    data = yf.download(fallback, start=start_date, progress=False, auto_adjust=True)
                
                if data is None or data.empty:
                    return pd.Series(dtype='float64')
                
                # Extract 'Close' - handling MultiIndex if yfinance returns it
                if 'Close' in data.columns:
                    res = data['Close']
                    return res.iloc[:, 0] if isinstance(res, pd.DataFrame) else res
                return pd.Series(dtype='float64')
            except Exception as e:
                print(f"Warning: Issue fetching {ticker}: {e}")
                return pd.Series(dtype='float64')

        es = safe_download("ES=F", fallback="SPY")
        nq = safe_download("NQ=F")
        
        # Use concat to align by date index robustly
        df = pd.concat([es.rename('ES_Close'), nq.rename('NQ_Close')], axis=1).ffill()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    def calculate_forward_metrics(self, df):
        """Calculates forward returns, volatility, and drawdowns without lookahead bias."""
        # Daily Log Returns
        df['ES_Daily_Return'] = np.log(df['ES_Close'] / df['ES_Close'].shift(1))
        
        # Forward Returns (calculated after the current date)
        for h in self.horizons:
            df[f'ES_{h}D_Fwd_Return'] = df['ES_Close'].shift(-h) / df['ES_Close'] - 1
            
        # Realized Volatility (Trailing 20D)
        df['ES_Realized_Vol_20D'] = df['ES_Daily_Return'].rolling(20).std() * np.sqrt(252)
        
        # Forward Max Drawdown (Next 60 Days)
        # We use cumprod of 1+ret to find peak-to-trough in forward window
        def get_fwd_dd(s):
            if len(s) < 60: return np.nan
            # Normalized path starting at 1.0
            path = (s / s.iloc[0])
            return (path.div(path.cummax()) - 1).min()
            
        df['ES_Fwd_Max_DD_60D'] = df['ES_Close'].rolling(window=61).apply(
            lambda x: (x.min() / x.iloc[0]) - 1, raw=False).shift(-60)
            
        # Forward Max Upside (Next 60 Days)
        df['ES_Fwd_Max_Upside_60D'] = df['ES_Close'].rolling(window=61).apply(
            lambda x: (x.max() / x.iloc[0]) - 1, raw=False).shift(-60)
            
        # Forward Sharpe Ratio (60D)
        rolling_mean = df['ES_Daily_Return'].rolling(60).mean().shift(-60)
        rolling_std = df['ES_Daily_Return'].rolling(60).std().shift(-60)
        df['ES_Fwd_Sharpe_60D'] = (rolling_mean / rolling_std) * np.sqrt(252)
        
        return df

    def aggregate_macro_layers(self, equity_df):
        """Integrates historical layers into a master daily dataset."""
        print("Aggregating Macro Layers...")
        
        # Fetch historical dataframes from existing modules
        liq_df = get_liquidity_report().set_index('date')
        credit_df = get_credit_spread_report().set_index('date')
        vol_df = get_volatility_report().set_index('date')
        
        # Fetch and process COT positioning historically
        cot_raw = fetch_cot_positioning(years=range(2010, datetime.now().year + 1))
        cot_raw['spx_pos_percentile'] = cot_raw['S&P 500_net_noncomm'].rolling(window=104).apply(
            lambda x: x.rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )

        # Join all to equity index (Daily)
        master = equity_df.join(liq_df[['walcl_3m_pct', 'liquidity_contraction', 'liquidity_expansion']], how='left')
        master = master.join(credit_df[['HY_OAS', 'hy_oas_zscore', 'hy_oas_percentile', 'credit_regime']], how='left')
        master = master.join(vol_df[['VIX', 'volatility_regime']], how='left')
        master = master.join(cot_raw[['spx_pos_percentile']], how='left')
        
        # Fill weekly/intermittent data forward to avoid lookahead (last known value)
        cols_to_fill = ['walcl_3m_pct', 'liquidity_contraction', 'liquidity_expansion', 
                        'HY_OAS', 'hy_oas_zscore', 'hy_oas_percentile', 'credit_regime', 
                        'VIX', 'volatility_regime', 'spx_pos_percentile']
        master[cols_to_fill] = master[cols_to_fill].ffill()
        
        return master.dropna(subset=['ES_Close'])

    def classify_states(self, df):
        """Applies the 4-layer classification and status logic."""
        # 1. Liquidity
        df['Liquidity_State'] = 'Flat'
        df.loc[df['liquidity_expansion'] == True, 'Liquidity_State'] = 'Expanding'
        df.loc[df['liquidity_contraction'] == True, 'Liquidity_State'] = 'Contracting'
        df['Liquidity_Status'] = df['Liquidity_State'].map({'Expanding': 'Normal', 'Flat': 'Normal', 'Contracting': 'Stress'})
        
        # 2. Credit
        # Map credit_regime (Tight, Widening, Blowing Out) to Status
        df['Credit_State'] = df['credit_regime'].fillna('Neutral')
        df['Credit_Status'] = 'Normal'
        df.loc[df['Credit_State'] == 'Widening', 'Credit_Status'] = 'Warning'
        df.loc[df['Credit_State'] == 'Blowing Out', 'Credit_Status'] = 'Stress'
        
        # 3. Positioning
        df['Positioning_State'] = 'Neutral'
        df.loc[df['spx_pos_percentile'] > 0.90, 'Positioning_State'] = 'Crowded'
        df.loc[df['spx_pos_percentile'] > 0.70, 'Positioning_State'] = 'Bullish'
        df.loc[df['spx_pos_percentile'] < 0.30, 'Positioning_State'] = 'Bearish'
        df['Positioning_Status'] = 'Normal'
        df.loc[df['Positioning_State'] == 'Crowded', 'Positioning_Status'] = 'Warning'
        
        # 4. Volatility
        df['Volatility_State'] = df['volatility_regime'].fillna('Normal')
        df['Volatility_Status'] = 'Normal'
        df.loc[df['Volatility_State'] == 'Stress', 'Volatility_Status'] = 'Warning'
        df.loc[df['Volatility_State'] == 'Panic', 'Volatility_Status'] = 'Stress'
        
        # Destabilization Score
        status_cols = ['Liquidity_Status', 'Credit_Status', 'Positioning_Status', 'Volatility_Status']
        df['Destabilization_Score'] = df[status_cols].apply(
            lambda x: sum([1 for s in x if s in ['Warning', 'Stress']]), axis=1
        )
        
        # Macro Regime Assignment
        def assign_regime(score):
            if score == 0: return 'A'
            if score == 1: return 'B'
            if score == 2: return 'C'
            return 'D'
            
        df['Macro_Regime'] = df['Destabilization_Score'].apply(assign_regime)
        return df

    def run_statistical_analysis(self, df):
        """Performs ANOVA and T-Tests across regimes."""
        metrics = [f'ES_{h}D_Fwd_Return' for h in self.horizons] + \
                  ['ES_Realized_Vol_20D', 'ES_Fwd_Max_DD_60D', 'ES_Fwd_Sharpe_60D']
        
        results = []
        for metric in metrics:
            groups = [df[df['Macro_Regime'] == r][metric].dropna() for r in ['A', 'B', 'C', 'D']]
            
            # ANOVA
            f_stat, p_val = stats.f_oneway(*groups)
            
            # Mean by regime
            means = {f'Regime_{r}_Mean': groups[i].mean() for i, r in enumerate(['A', 'B', 'C', 'D'])}
            counts = {f'Regime_{r}_N': len(groups[i]) for i, r in enumerate(['A', 'B', 'C', 'D'])}
            
            res = {'Metric': metric, 'ANOVA_F': f_stat, 'ANOVA_P': p_val}
            res.update(means)
            res.update(counts)
            
            # Welch T-Test: A (Stable) vs D (Stress)
            if len(groups[0]) > 5 and len(groups[3]) > 5:
                t_stat, tp_val = stats.ttest_ind(groups[0], groups[3], equal_var=False)
                res['T_Test_A_vs_D_P'] = tp_val
                
                # Cohen's d
                n1, n2 = len(groups[0]), len(groups[3])
                v1, v2 = groups[0].var(), groups[3].var()
                pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
                res['Cohens_D_A_vs_D'] = (groups[0].mean() - groups[3].mean()) / pooled_std if pooled_std != 0 else 0

            results.append(res)
            
        return pd.DataFrame(results)

    def generate_visuals(self, df, stats_df):
        """Generates the required research charts."""
        sns.set_style("whitegrid")
        
        # 1. Timeline
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['ES_Close'], color='black', alpha=0.3, label='ES Price')
        colors = {'A': 'green', 'B': 'yellow', 'C': 'orange', 'D': 'red'}
        for regime, color in colors.items():
            mask = df['Macro_Regime'] == regime
            plt.scatter(df.index[mask], df['ES_Close'][mask], color=color, s=2, label=f'Regime {regime}')
        plt.title("ES Price with Macro Regime Overlay")
        plt.legend()
        plt.savefig(self.output_dir / "macro_regime_timeline.png")
        plt.close()

        # 2. Prevalence
        plt.figure(figsize=(8, 6))
        df['Macro_Regime'].value_counts(normalize=True).sort_index().plot(kind='bar', color='skyblue')
        plt.title("Macro Regime Prevalence (%)")
        plt.ylabel("Frequency")
        plt.savefig(self.output_dir / "macro_regime_prevalence.png")
        plt.close()

        # 3. Forward Returns
        ret_cols = [f'ES_{h}D_Fwd_Return' for h in self.horizons]
        fwd_ret_summary = df.groupby('Macro_Regime')[ret_cols].mean()
        fwd_ret_summary.plot(kind='bar', figsize=(10, 6))
        plt.title("Mean Forward Returns by Macro Regime")
        plt.axhline(0, color='black', lw=1)
        plt.savefig(self.output_dir / "forward_returns_by_macro_regime.png")
        plt.close()

        # 4. Drawdowns
        plt.figure(figsize=(8, 6))
        df.groupby('Macro_Regime')['ES_Fwd_Max_DD_60D'].mean().plot(kind='bar', color='salmon')
        plt.title("Mean 60D Forward Max Drawdown")
        plt.savefig(self.output_dir / "drawdowns_by_macro_regime.png")
        plt.close()

        # 5. Transition Matrix
        trans = pd.crosstab(df['Macro_Regime'], df['Macro_Regime'].shift(-1), normalize='index')
        plt.figure(figsize=(8, 6))
        sns.heatmap(trans, annot=True, cmap='Blues', fmt='.2f')
        plt.title("Macro Regime Transition Probabilities")
        plt.savefig(self.output_dir / "transition_matrix_heatmap.png")
        plt.close()
        
        return trans

    def write_interpretation(self, stats_df, df):
        """Generates the Markdown research report."""
        regime_counts = df['Macro_Regime'].value_counts()
        
        # Identify Best/Worst Regimes
        ret_20d = stats_df[stats_df['Metric'] == 'ES_20D_Fwd_Return'].iloc[0]
        dd_60d = stats_df[stats_df['Metric'] == 'ES_Fwd_Max_DD_60D'].iloc[0]
        
        md = f"""# Paper 2: Cross-Layer Macro Regimes
## Which Macro Conditions Drive Future Equity Market Performance?

### Key Findings
This research segments the market into four regimes based on the destabilization of Liquidity, Credit, Positioning, and Volatility layers.

### Methodology
- **Regime A (Stable)**: Score 0. All macro layers in normal states.
- **Regime B (Warning)**: Score 1. One layer showing early stress.
- **Regime C (Unstable)**: Score 2. Two layers showing stress.
- **Regime D (Crisis)**: Score 3-4. Severe cross-layer destabilization.

### Results Interpretation
1. **Best Performance**: Regime {stats_df.loc[stats_df['Metric']=='ES_20D_Fwd_Return', ['Regime_A_Mean','Regime_B_Mean','Regime_C_Mean','Regime_D_Mean']].idxmax(axis=1).iloc[0][-1]} produced the highest average 20D forward returns.
2. **Worst Drawdowns**: Regime {stats_df.loc[stats_df['Metric']=='ES_Fwd_Max_DD_60D', ['Regime_A_Mean','Regime_B_Mean','Regime_C_Mean','Regime_D_Mean']].idxmin(axis=1).iloc[0][-1]} was associated with the deepest forward 60D drawdowns.
3. **Statistical Significance**: The ANOVA p-value for 20D returns is {stats_df[stats_df['Metric']=='ES_20D_Fwd_Return']['ANOVA_P'].iloc[0]:.4f}.

### Regime Sample Sizes
"""
        for r in ['A', 'B', 'C', 'D']:
            count = regime_counts.get(r, 0)
            warn = "⚠️ **WARNING: Insufficient samples for reliable inference.**" if count < self.min_obs_threshold else ""
            md += f"- Regime {r}: {count} observations. {warn}\n"
            
        md += """
### Limitations
- **Lagging Indicators**: Some macro data (Liquidity/COT) is reported with a lag; while we ffill to handle this, the real-world reaction may differ.
- **Regime Definition**: The destabilization score weights all layers equally; however, Credit or Liquidity may have asymmetric impacts.
"""
        with open(self.output_dir / "paper_02_interpretation.md", "w") as f:
            f.write(md)

    def run_pipeline(self):
        print("--- Starting Paper 2 Research Pipeline ---")
        
        # 1. Data Collection
        equity_df = self.fetch_equity_data()
        equity_df = self.calculate_forward_metrics(equity_df)
        
        # 2. Integration
        master = self.aggregate_macro_layers(equity_df)
        master = self.classify_states(master)
        
        # 3. Statistical Testing
        stats_results = self.run_statistical_analysis(master)
        
        # 4. Visualization & Output
        trans_matrix = self.generate_visuals(master, stats_results)
        self.write_interpretation(stats_results, master)
        
        # 5. Save Tables
        master.to_csv(self.output_dir / "paper_02_macro_master_dataset.csv")
        stats_results.to_csv(self.output_dir / "paper_02_statistical_tests.csv")
        trans_matrix.to_csv(self.output_dir / "paper_02_transition_matrix.csv")
        
        # Regime Summary Table
        summary = master.groupby('Macro_Regime').agg({
            'ES_Daily_Return': 'count',
            'ES_20D_Fwd_Return': 'mean',
            'ES_Fwd_Max_DD_60D': 'mean',
            'ES_Realized_Vol_20D': 'mean'
        }).rename(columns={'ES_Daily_Return': 'Observations'})
        summary.to_csv(self.output_dir / "paper_02_macro_regime_summary.csv")
        
        print(f"\nPipeline Complete. Outputs saved to: {self.output_dir}")

if __name__ == "__main__":
    research = MultiLayerMacroResearch()
    research.run_pipeline()