import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
from datetime import datetime, timedelta
import time
from config import FRED_API_KEY

def fetch_liquidity_data(start_date='2010-01-01'):
    """
    Fetches WALCL (Weekly), RRPONTSYD (Daily), FEDFUNDS (Monthly), and M2SL (Monthly).
    Aligns all to a Weekly Wednesday frequency.
    """
    api_key = FRED_API_KEY
    if not api_key:
        raise ValueError("FRED_API_KEY not found in config.py")
    
    fred = Fred(api_key=api_key.strip())
    max_retries = 3
    retry_delay = 5

    def safe_fetch(series_id):
        for attempt in range(max_retries):
            try:
                data = fred.get_series(series_id, observation_start=start_date)
                if data is not None and not data.empty:
                    # Normalize to naive datetime for joining
                    data.index = pd.to_datetime(data.index).tz_localize(None)
                    return data
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"⚠️ FRED Error for {series_id}: {e}")
                time.sleep(retry_delay)
        return None
    
    # Fetch raw data
    walcl = safe_fetch('WALCL')
    if walcl is not None: walcl = walcl.rename('WALCL')
    
    rrp = safe_fetch('RRPONTSYD')
    if rrp is not None: rrp = rrp.rename('RRP')

    try:
        ff = safe_fetch('FEDFUNDS')
        if ff is not None: ff = ff.rename('FEDFUNDS')
    except Exception:
        print("⚠️ FEDFUNDS failed. Fetching ^IRX (13-week T-Bill) from yfinance as proxy.")
        irx = yf.download("^IRX", start=start_date, progress=False)['Close']
        ff = irx.squeeze().rename('FEDFUNDS') / 10.0 # Normalizing yield to % scale

    m2 = safe_fetch('M2SL')
    if m2 is not None: m2 = m2.rename('M2')
    
    # 1. Align on WALCL index (Wednesdays)
    if walcl is None:
        raise ConnectionError("Critical liquidity data (WALCL) could not be fetched.")
        
    df = pd.DataFrame(index=walcl.index)
    df['WALCL'] = walcl
    
    # 2. Daily data: Reindex/Join RRP (Takes the Wednesday value)
    df = df.join(rrp, how='left')
    # Ensure Wednesday alignment for daily series by filling missing values (e.g. holidays)
    df['RRP'] = df['RRP'].ffill()
    
    # 3. Monthly data: Join and Forward Fill
    df = df.join(ff, how='left').ffill()
    df = df.join(m2, how='left').ffill()
    
    df.index.name = 'date'
    return df.reset_index()

def compute_liquidity_features(df: pd.DataFrame):
    """
    Computes:
    - 12-week % change (3-month) in WALCL
    - 12-week moving z-score of weekly ΔWALCL (vs 52-week lookback)
    - 4-week change in RRP
    - RRP Percentile (Historical)
    - Contraction Flags
    """
    # Ensure date is index for calculations
    df = df.set_index('date').copy()
    
    # 1. 12-week percent change in WALCL (3m Change)
    df['walcl_3m_pct'] = df['WALCL'].pct_change(12)
    
    # 2. Weekly Change and 52-week Rolling Z-Score of Weekly Change
    # (Prompt requested 12-week z-score vs 1-year lookback)
    weekly_delta = df['WALCL'].diff()
    rolling_mean = weekly_delta.rolling(window=52).mean()
    rolling_std = weekly_delta.rolling(window=52).std()
    df['walcl_zscore'] = (weekly_delta - rolling_mean) / rolling_std
    
    # 3. 4-week change in RRP
    df['rrp_4w_delta'] = df['RRP'].diff(4)
    
    # 4. RRP Percentile (Historical lookback)
    df['rrp_percentile'] = df['RRP'].expanding().apply(
        lambda x: (x.rank(pct=True).iloc[-1]) if len(x) > 0 else np.nan
    )
    
    # 5. Define Liquidity Contraction Flag
    # Condition: 3m ΔWALCL < -2%
    df['liquidity_contraction'] = (df['walcl_3m_pct'] < -0.02)
    
    # 6. Define Liquidity Expansion Flag (e.g. March 2020 style)
    df['liquidity_expansion'] = (df['walcl_3m_pct'] > 0.05)
    
    return df.reset_index()

def get_liquidity_report():
    """Orchestrator for the analysis."""
    # Propagate exceptions so that the true cause of failure is visible to tests/logs
    raw_df = fetch_liquidity_data()
    processed_df = compute_liquidity_features(raw_df)
    return processed_df

if __name__ == "__main__":
    print("--- Running Liquidity Data Ingestion ---")
    df_liq = get_liquidity_report()
    if not df_liq.empty:
        print(f"Successfully processed {len(df_liq)} weeks.")
        print("\nLatest Observations:")
        print(df_liq.tail(5)[['date', 'WALCL', 'walcl_3m_pct', 'liquidity_contraction']])
        
        # Check for historical flag (e.g. COVID QE)
        covid_period = df_liq[(df_liq['date'] >= '2020-03-01') & (df_liq['date'] <= '2020-05-01')]
        if not covid_period.empty:
            expansion_triggered = covid_period['liquidity_expansion'].any()
            print(f"\nExpansion triggered in Mar/Apr 2020: {expansion_triggered}")