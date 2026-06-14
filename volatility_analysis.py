import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
import time # For retry mechanism
from config import FRED_API_KEY

def fetch_volatility_data(start_date='2005-01-01'):
    """
    Fetches VIXCLS (CBOE VIX) and VXNCLS (CBOE NASDAQ-100 Volatility Index) from FRED.
    Aligns daily values to end-of-week (Friday) for reporting.
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
        'VIXCLS': 'VIX',
        'VXNCLS': 'VXN'
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
    
    if not fred_data_dict:
        raise ConnectionError("Failed to fetch any required FRED volatility series.")
    
    # Create DataFrame from dict for cleaner alignment
    df = pd.DataFrame(fred_data_dict)
    
    # Sort and perform initial cleaning (forward-fill to handle daily gaps like holidays)
    df = df.sort_index().ffill()
    
    # Align to end-of-week (Friday) and take the last available value of the week
    # Use .ffill() again to carry forward values for weeks where Friday might be missing
    df_weekly = df.resample('W-FRI').last().ffill()
    
    df_weekly.index.name = 'date'
    return df_weekly.reset_index()

def compute_volatility_features(df: pd.DataFrame):
    """
    Computes volatility signals and defines volatility regimes.
    """
    df = df.set_index('date').copy()
    
    # 1. 1-month change in VIX (approx 4 weeks for weekly data)
    df['vix_1m_change'] = df['VIX'].pct_change(4)
    
    # 2. Define Volatility Regimes
    df['volatility_regime'] = 'Normal' # Default regime
    
    # "Complacency" if VIX < 15
    df.loc[df['VIX'] < 15, 'volatility_regime'] = 'Complacency'
    
    # "Stress" if 25 < VIX <= 40
    df.loc[(df['VIX'] > 25) & (df['VIX'] <= 40), 'volatility_regime'] = 'Stress'
    
    # "Panic" if VIX > 40
    df.loc[df['VIX'] > 40, 'volatility_regime'] = 'Panic'
    
    return df.reset_index()

def get_volatility_report():
    """Orchestrates the volatility data pipeline."""
    raw_df = fetch_volatility_data()
    processed_df = compute_volatility_features(raw_df)
    return processed_df

if __name__ == "__main__":
    print("--- Running Volatility Regime Data Pipeline ---")
    df_vol = get_volatility_report()
    
    if not df_vol.empty:
        print(f"Successfully processed {len(df_vol)} weeks of volatility data.")
        print("\nLatest Observations:")
        print(df_vol.tail(5)[['date', 'VIX', 'vix_1m_change', 'volatility_regime']])
        
        # --- Backtest Check: Known Turbulent Days ---
        print("\n--- Historical Volatility Check ---")
        
        # Oct 2008 Financial Crisis
        crisis_2008 = df_vol[(df_vol['date'] >= '2008-10-01') & (df_vol['date'] <= '2008-10-31')]
        if not crisis_2008.empty:
            max_vix_2008 = crisis_2008['VIX'].max()
            print(f"Max VIX during Oct 2008 crisis: {max_vix_2008:.2f}")
            if (crisis_2008['VIX'] > 40).any():
                print("✅ Confirmed: VIX exceeded 40 during Oct 2008 crisis.")
            else:
                print("❌ Warning: VIX did NOT exceed 40 during Oct 2008 crisis. Check data or logic.")
            if (crisis_2008['volatility_regime'] == 'Panic').any():
                print("✅ Confirmed: 'Panic' regime detected during Oct 2008 crisis.")
            
        # Mar 2020 COVID-19 Pandemic
        crisis_2020 = df_vol[(df_vol['date'] >= '2020-03-01') & (df_vol['date'] <= '2020-03-31')]
        if not crisis_2020.empty:
            max_vix_2020 = crisis_2020['VIX'].max()
            print(f"Max VIX during Mar 2020 crisis: {max_vix_2020:.2f}")
            if (crisis_2020['VIX'] > 40).any():
                print("✅ Confirmed: VIX exceeded 40 during Mar 2020 crisis.")
            else:
                print("❌ Warning: VIX did NOT exceed 40 during Mar 2020 crisis. Check data or logic.")
            if (crisis_2020['volatility_regime'] == 'Panic').any():
                print("✅ Confirmed: 'Panic' regime detected during Mar 2020 crisis.")
        
        # --- Recent Volatility Regime Check ---
        print("\n--- Recent Volatility Regime Check ---")
        recent_regime = df_vol.tail(1)['volatility_regime'].iloc[0]
        recent_vix = df_vol.tail(1)['VIX'].iloc[0]
        print(f"Current Volatility Regime: {recent_regime}")
        print(f"Latest VIX: {recent_vix:.2f}")

    else:
        print("Failed to generate volatility report.")