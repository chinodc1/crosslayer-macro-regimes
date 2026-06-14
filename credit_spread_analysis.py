import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
from datetime import datetime
import time # For retry mechanism
from config import FRED_API_KEY

def fetch_credit_spread_data(start_date='2005-01-01'):
    """
    Fetches BAA10Y (daily), BAMLH0A0HYM2 (daily) from FRED,
    and HYG, LQD (daily close) from yfinance.
    Merges and cleans the data, aligning on dates.
    Includes retry logic for FRED API.
    """
    api_key = FRED_API_KEY
    if not api_key:
        raise ValueError("FRED_API_KEY not found in config.py")
    
    fred = Fred(api_key=api_key.strip())
    max_retries = 3
    retry_delay = 5 # seconds
    
    # --- Fetch FRED Data ---
    fred_series_ids = {
        'BAA10Y': 'BAA10Y',
        'BAMLH0A0HYM2': 'HY_OAS'
    }
    fred_data_dict = {}
    for series_id, col_name in fred_series_ids.items():
        for attempt in range(max_retries):
            try:
                data = fred.get_series(series_id, observation_start=start_date)
                if data is not None and not data.empty:
                    # Normalize index to naive datetime immediately for robust alignment
                    data.index = pd.to_datetime(data.index).tz_localize(None)
                    fred_data_dict[col_name] = data
                    break # Success, break retry loop
            except Exception as e:
                print(f"FRED API error for {series_id} on attempt {attempt + 1}: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
    
    if len(fred_data_dict) < 2:
        raise ConnectionError(f"Failed to fetch required FRED series. Found: {list(fred_data_dict.keys())}")
    
    # Create DataFrame from dict for cleaner alignment
    fred_df = pd.DataFrame(fred_data_dict)
    
    # --- Fetch yfinance Data ---
    yf_symbols = ['HYG', 'LQD']
    yf_data = yf.download(yf_symbols, start=start_date, progress=False, auto_adjust=True)
    
    if yf_data.empty:
        raise ValueError(f"No yfinance data returned for {yf_symbols}")
        
    # Extract 'Close' prices and rename columns
    hyg_lqd_df = yf_data['Close'].rename(columns={'HYG': 'HYG_Close', 'LQD': 'LQD_Close'})
    
    # --- Merge Data ---
    # Indices for fred_df are already normalized during dict construction
    hyg_lqd_df.index = pd.to_datetime(hyg_lqd_df.index).tz_localize(None)
    
    # Merge FRED and yfinance data on their indices
    # Use outer merge to keep all dates, then forward-fill and back-fill
    # Ensure FRED data is not empty before merging
    if fred_df.empty:
        df = hyg_lqd_df
    elif hyg_lqd_df.empty:
        df = fred_df
    else:
        df = pd.merge(fred_df, hyg_lqd_df, left_index=True, right_index=True, how='outer')
    
    # Sort and perform robust cleaning
    df = df.sort_index()
    
    # Robust Drop: Only drop if we have NO spread data at all (FRED or yfinance)
    df = df.dropna(subset=['HYG_Close', 'LQD_Close'], how='all')

    # Fill logic
    for col in ['BAA10Y', 'HY_OAS']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=5).ffill(limit=3)
    
    # Create synthetic HY_OAS if FRED failed
    synthetic_spread = (1 - (df['HYG_Close'] / df['LQD_Close'])) * 50
    if 'HY_OAS' not in df.columns:
        df['HY_OAS'] = synthetic_spread
    else:
        # Fill missing values with synthetic spread where FRED data is null
        df['HY_OAS'] = df['HY_OAS'].fillna(synthetic_spread)

    if 'BAA10Y' not in df.columns:
        df['BAA10Y'] = np.nan # Placeholder to avoid schema breaks
    
    df.index.name = 'date'
    return df.reset_index()

def compute_credit_spread_features(df: pd.DataFrame):
    """
    Computes credit spread signals and defines credit regimes.
    """
    df = df.set_index('date').copy()
    
    # 1. 6-month percent change of BAA10Y (approx 120 trading days)
    df['baa10y_6m_pct_change'] = df['BAA10Y'].pct_change(120)
    
    # 2. Daily z-score of HY OAS (BAMLH0A0HYM2) (180-day rolling window)
    window_zscore = 180
    # Use min_periods=1 to ensure results are generated as soon as possible, for testing early crisis
    df['hy_oas_rolling_mean'] = df['HY_OAS'].rolling(window=window_zscore, min_periods=1).mean()
    df['hy_oas_rolling_std'] = df['HY_OAS'].rolling(window=window_zscore, min_periods=1).std()

    # Check if the series is valid (not constant)
    zero_std_count = (df['hy_oas_rolling_std'] == 0).sum()
    if len(df) > window_zscore and zero_std_count > (len(df) * 0.9):
         print("CRITICAL WARNING: HY_OAS series appears to be constant/flat. Check API data.")
         # Force z-score to NaN if the data is junk so tests fail clearly
         df['hy_oas_rolling_std'] = df['hy_oas_rolling_std'].replace(0, np.nan)

    if zero_std_count > 0:
        print(f"DEBUG: Found {zero_std_count} instances of 0 rolling standard deviation.")
    # -------------------------------------------------------------------------
    
    # Handle cases where rolling standard deviation is zero to prevent division by zero.
    # If std is 0, z-score should be 0 as there's no deviation from the mean.
    df['hy_oas_zscore'] = np.where(
        df['hy_oas_rolling_std'] == 0,
        0,
        (df['HY_OAS'] - df['hy_oas_rolling_mean']) / df['hy_oas_rolling_std']
    )
    
    # 3. Yield Ratio = 1 - (HYG_Close / LQD_Close)
    df['yield_ratio'] = 1 - (df['HYG_Close'] / df['LQD_Close'])
    
    # 4. Define Credit Regimes based on HY_OAS (BAMLH0A0HYM2)
    # Use a 5-year rolling window for percentiles (approx 5*252 trading days)
    window_percentile = 5 * 252
    
    # Calculate rolling percentile rank
    df['hy_oas_percentile'] = df['HY_OAS'].rolling(window=window_percentile, min_periods=1).apply(
        lambda x: x.rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    )
    
    # Initialize regime column
    df['credit_regime'] = 'Neutral'
    
    # "Tight" if spreads are <50th percentile of last 5 years
    df.loc[df['hy_oas_percentile'] < 0.50, 'credit_regime'] = 'Tight'
    
    # "Widening" > 70th percentile
    df.loc[df['hy_oas_percentile'] > 0.70, 'credit_regime'] = 'Widening'
    
    # "Blowing out" if 3σ move (z-score > 3)
    df.loc[df['hy_oas_zscore'] > 3, 'credit_regime'] = 'Blowing Out'
    
    # Clean up intermediate columns
    df = df.drop(columns=['hy_oas_rolling_mean', 'hy_oas_rolling_std'])
    
    return df.reset_index()

def get_credit_spread_report():
    """Orchestrates the credit spread data pipeline."""
    raw_df = fetch_credit_spread_data()
    processed_df = compute_credit_spread_features(raw_df)
    return processed_df

if __name__ == "__main__":
    print("--- Running Credit Spread Data Pipeline ---")
    df_credit = get_credit_spread_report()
    
    if not df_credit.empty:
        print(f"Successfully processed {len(df_credit)} days of credit spread data.")
        print("\nLatest Observations:")
        print(df_credit.tail(5)[['date', 'BAA10Y', 'HY_OAS', 'hy_oas_zscore', 'yield_ratio', 'credit_regime']])
        
        # --- Backtest Check: 2008 Financial Crisis ---
        print("\n--- 2008 Financial Crisis Check ---")
        crisis_2008 = df_credit[(df_credit['date'] >= '2008-01-01') & (df_credit['date'] <= '2009-03-31')]
        
        if not crisis_2008.empty:
            max_oas_zscore_2008 = crisis_2008['hy_oas_zscore'].max()
            min_oas_zscore_2008 = crisis_2008['hy_oas_zscore'].min()
            
            print(f"Max HY_OAS Z-score during 2008-2009 crisis: {max_oas_zscore_2008:.2f}")
            print(f"Min HY_OAS Z-score during 2008-2009 crisis: {min_oas_zscore_2008:.2f}")
            
            # Verify that in 2008 HY_OAS z-score > 3
            if (crisis_2008['hy_oas_zscore'] > 3).any():
                print("✅ Confirmed: HY_OAS Z-score exceeded 3 during the 2008-2009 crisis.")
            else:
                print("❌ Warning: HY_OAS Z-score did NOT exceed 3 during the 2008-2009 crisis. Check data or logic.")
            
            # Check for 'Blowing Out' regime during the crisis
            blowing_out_days = crisis_2008[crisis_2008['credit_regime'] == 'Blowing Out']
            if not blowing_out_days.empty:
                print(f"✅ Confirmed: 'Blowing Out' regime detected on {len(blowing_out_days)} days during 2008-2009 crisis.")
                print("Example 'Blowing Out' dates:")
                print(blowing_out_days[['date', 'HY_OAS', 'hy_oas_zscore']].head())
            else:
                print("❌ Warning: 'Blowing Out' regime was not detected during the 2008-2009 crisis.")
        else:
            print("No data available for the 2008-2009 crisis period.")
            
        # --- Recent Credit Regime Check ---
        print("\n--- Recent Credit Regime Check ---")
        recent_regime = df_credit.tail(1)['credit_regime'].iloc[0]
        recent_oas = df_credit.tail(1)['HY_OAS'].iloc[0]
        recent_oas_zscore = df_credit.tail(1)['hy_oas_zscore'].iloc[0]
        print(f"Current Credit Regime: {recent_regime}")
        print(f"Latest HY OAS: {recent_oas:.2f}")
        print(f"Latest HY OAS Z-score: {recent_oas_zscore:.2f}")

    else:
        print("Failed to generate credit spread report.")