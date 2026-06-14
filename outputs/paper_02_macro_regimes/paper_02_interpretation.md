# Paper 2: Cross-Layer Macro Regimes
## Which Macro Conditions Drive Future Equity Market Performance?

### Key Findings
This research segments the market into four regimes based on the destabilization of Liquidity, Credit, Positioning, and Volatility layers.

### Methodology
- **Regime A (Stable)**: Score 0. All macro layers in normal states.
- **Regime B (Warning)**: Score 1. One layer showing early stress.
- **Regime C (Unstable)**: Score 2. Two layers showing stress.
- **Regime D (Crisis)**: Score 3-4. Severe cross-layer destabilization.

### Results Interpretation
1. **Best Performance**: Regime n produced the highest average 20D forward returns.
2. **Worst Drawdowns**: Regime n was associated with the deepest forward 60D drawdowns.
3. **Statistical Significance**: The ANOVA p-value for 20D returns is 0.0000.

### Regime Sample Sizes
- Regime A: 2402 observations. 
- Regime B: 1462 observations. 
- Regime C: 267 observations. 
- Regime D: 6 observations. ⚠️ **WARNING: Insufficient samples for reliable inference.**

### Limitations
- **Lagging Indicators**: Some macro data (Liquidity/COT) is reported with a lag; while we ffill to handle this, the real-world reaction may differ.
- **Regime Definition**: The destabilization score weights all layers equally; however, Credit or Liquidity may have asymmetric impacts.
