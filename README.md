# Multi-Layer Macro Regime Research Framework

## Abstract
This project provides a comprehensive quantitative research engine designed to identify and validate market regimes using a multi-layered macro approach. By synthesizing signals across Liquidity, Credit, Positioning, and Volatility, the framework categorizes market environments into distinct regimes (A through D) to analyze their impact on equity performance (ES Futures) and systemic risk.

## Methodology
- **Layered Analysis**: Aggregates data across four critical macro pillars:
    - **Liquidity**: Tracks Central Bank balance sheet trends (WALCL) and money supply (M2).
    - **Credit**: Monitors corporate spreads (HY OAS, BAA) to detect systemic stress or "blowing out" conditions.
    - **Positioning**: Evaluates market sentiment and "crowdedness" via COT reports and Put/Call ratios.
    - **Volatility**: Analyzes VIX and VXN levels to identify complacency, stress, or panic.
- **Destabilization Scoring**: Quantifies systemic risk by counting "Warning" or "Stress" signals across the four layers.
- **Regime Classification**: Segments the market into four regimes based on the Destabilization Score:
    - **Regime A**: Risk-on / Stable environment.
    - **Regime B**: Early warning / Mild deterioration.
    - **Regime C**: Unstable / Mixed signals.
    - **Regime D**: Crisis / Destabilization.
- **Statistical Analysis**: Evaluates forward returns (5D, 20D, 60D), realized volatility, and maximum drawdowns using robust statistical tests (ANOVA, Welch T-Tests, Newey-West HAC errors).
- **Robustness Validation**: A secondary validation engine uses block bootstrapping, outlier sensitivity tests, and subperiod stability checks to ensure findings are not results of lookahead bias.

## How to Run

### 1. API Key Setup
The project relies on FRED (Federal Reserve Economic Data) for macro indicators.
1. Obtain a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html).
2. Open `config.py`.
3. Replace the `FRED_API_KEY` value with your personal key.
   *Note: Ensure you do not share your private API key when distributing your own version of this code.*

### 2. Installation
Install the required dependencies using pip:
```bash
pip install pandas numpy yfinance scipy matplotlib seaborn fredapi python-dotenv statsmodels cot-reports
```

### 3. Execution
Run the primary research pipeline to generate datasets, charts, and reports:
```bash
python MultiLayerMacro.py
```

Run the validation suite to stress-test the findings:
```bash
python validation_paper_02.py
```

## Outputs
All results, including master datasets, statistical summaries, and publication-quality charts, are saved to the `outputs/` directory within this folder.

---
**Disclaimer**: This research is for educational and informational purposes only and does not constitute financial advice.