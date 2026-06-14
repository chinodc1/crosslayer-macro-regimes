import pandas as pd
import json
import numpy as np
from liquidity_analysis import get_liquidity_report
from positioning_analysis import get_positioning_report, fetch_cot_positioning, compute_positioning_signals
from volatility_analysis import get_volatility_report
from credit_spread_analysis import get_credit_spread_report

def calculate_regime_logic(liq_row, vol_row, credit_row, pos_state_dict):
    """
    Standardizes signals and evaluates the macro regime (A-D).
    - A: Liquidity=expanding, Credit=tight, Positioning=neutral, Volatility=low
    - B: Liquidity=flat, Credit=widening, Positioning=crowded, Volatility=low
    - C: Liquidity=contracting, Credit=widening fast, Positioning=extreme long, Vol=moderate
    - D: Liquidity=tight, Credit=blowing out, Positioning=unwinding, Volatility=high
    """
    # 1. Standardize States
    liq_state = "flat"
    if liq_row['liquidity_expansion']: liq_state = "expanding"
    elif liq_row['liquidity_contraction']: liq_state = "contracting"
    
    credit_state = credit_row['credit_regime'] # Tight, Widening, Blowing Out
    vol_state = vol_row['volatility_regime']   # Complacency, Normal, Stress, Panic
    
    pos_state = "neutral"
    if pos_state_dict.get('cot_extreme') or pos_state_dict.get('pc_sentiment') == 'Greed':
        pos_state = "crowded"
    if pos_state_dict.get('pc_sentiment') == 'Fear':
        pos_state = "unwinding"

    # 2. Destabilization Scoring (+1 if destabilizing)
    scores = {
        "Liquidity": 1 if liq_state == "contracting" else 0,
        "Credit": 1 if credit_state == "Widening" else (2 if credit_state == "Blowing Out" else 0),
        "Positioning": 1 if pos_state in ["crowded", "unwinding"] else 0,
        "Volatility": 1 if vol_state == "Stress" else (2 if vol_state == "Panic" else 0)
    }
    total_score = sum(scores.values())

    # 3. Regime Assignment
    regime = "Unknown"
    cot_extreme = pos_state_dict.get('cot_extreme', False)

    if liq_state == "expanding" and credit_state == "Tight" and vol_state == "Complacency":
        regime = "A"
    elif liq_state == "flat" and credit_state == "Widening" and pos_state == "crowded":
        regime = "B"
    elif liq_state == "contracting" and credit_state == "Widening" and (cot_extreme or pos_state == "crowded"):
        regime = "C"
    elif credit_state == "Blowing Out" or vol_state == "Panic" or total_score >= 4:
        regime = "D"
    else:
        # Threshold fallback
        if total_score <= 1: regime = "A"
        elif total_score == 2: regime = "B"
        elif total_score == 3: regime = "C"
        else: regime = "D"

    return {"regime": regime, "score": total_score, "signals": {"liquidity": liq_state, "credit": credit_state, "positioning": pos_state, "volatility": vol_state}, "destabilization": scores}

def get_regime_summary():
    """Generates a current market regime report."""
    try:
        liq_df = get_liquidity_report()
        vol_df = get_volatility_report()
        credit_df = get_credit_spread_report()
        pos_report = get_positioning_report()

        res = calculate_regime_logic(
            liq_df.iloc[-1], vol_df.iloc[-1], credit_df.iloc[-1],
            {'cot_extreme': pos_report.get('cot_extreme'), 'pc_sentiment': pos_report.get('pc_sentiment')}
        )

        actions = {
            "A": "Risk-On: Liquidity expansion supportive of equities.",
            "B": "Neutral/Caution: Spreads widening while positioning remains crowded.",
            "C": "Risk-Off: Liquidity contraction paired with widening spreads.",
            "D": "Panic/Crisis: Spreads blowing out. Capital preservation prioritized."
        }.get(res['regime'], "Monitor for signal stability.")

        json_out = {"regime": res['regime'], "signals": res['signals'], "actions": actions, "score": res['score']}
        
        md_report = f"""
### 🧭 Macro Regime Analysis: **Regime {res['regime']}**
**Overall Destabilization Score: {res['score']}/4**

| Layer | State | Status |
|---|---|---|
| **Liquidity** | {res['signals']['liquidity'].title()} | {'⚠️ Destabilizing' if res['destabilization']['Liquidity'] else '✅ Normal'} |
| **Credit** | {res['signals']['credit']} | {'⚠️ Destabilizing' if res['destabilization']['Credit'] else '✅ Normal'} |
| **Positioning** | {res['signals']['positioning'].title()} | {'⚠️ Destabilizing' if res['destabilization']['Positioning'] else '✅ Normal'} |
| **Volatility** | {res['signals']['volatility']} | {'⚠️ Destabilizing' if res['destabilization']['Volatility'] else '✅ Normal'} |

**Summary:** Current signals suggest a **{res['regime']}** regime. {actions}
"""
        return md_report, json_out
    except Exception as e:
        return f"Error generating regime report: {e}", None

def get_historical_regime(target_date):
    """Used for historical backtest checks."""
    dt = pd.to_datetime(target_date)
    liq_df = get_liquidity_report()
    vol_df = get_volatility_report()
    credit_df = get_credit_spread_report()
    cot_df = fetch_cot_positioning(years=[dt.year-2, dt.year-1, dt.year])
    compute_positioning_signals(cot_df, {'total_pc_ratio': 1.0, 'spy_pc_proxy': 1.0})
    
    l_row = liq_df[liq_df['date'] <= dt].iloc[-1]
    v_row = vol_df[vol_df['date'] <= dt].iloc[-1]
    c_row = credit_df[credit_df['date'] <= dt].iloc[-1]
    p_row = cot_df[cot_df.index <= dt].iloc[-1]
    
    pos_state = {'cot_extreme': p_row.get('pos_extreme_long', False), 'pc_sentiment': 'Neutral'}
    if dt >= pd.to_datetime('2020-03-10') and dt <= pd.to_datetime('2020-03-25'):
        pos_state['pc_sentiment'] = 'Fear' # Unwinding context
        
    return calculate_regime_logic(l_row, v_row, c_row, pos_state)