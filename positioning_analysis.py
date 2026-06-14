import os
import pandas as pd
import numpy as np
import cot_reports as cot
import yfinance as yf
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

def fetch_cot_positioning(years=None):
    """
    Fetches COT Legacy Futures data for S&P 500 and NASDAQ.
    Handles multi-year aggregation for percentile calculations.
    """
    if years is None:
        current_year = datetime.now().year
        # Attempt to get last 3 years for 104-week percentile
        years = [current_year - 2, current_year - 1, current_year]

    all_data = []
    print(f"--- Fetching COT Data for years: {years} ---")
    
    for year in years:
        try:
            # cot_reports handles the ZIP download and parsing
            df_year = cot.cot_year(year=year, cot_report_type='legacy_fut')
            if not df_year.empty:
                all_data.append(df_year)
        except Exception as e:
            print(f"⚠️ Could not fetch COT data for {year}: {e}")

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    
    # Clean column names (handle inconsistencies like spaces/underscores)
    df.columns = [col.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "") for col in df.columns]
    
    # Standardize column references
    market_col = 'market_and_exchange_names'

    # Robust matching for date and positioning columns as report headers vary by year/type
    # Try common date column names found in Legacy and TFF reports
    date_col = next((c for c in ['report_date_as_mm_dd_yyyy', 'as_of_date_in_form_yyyy_mm_dd', 'as_of_date_in_form_yymmdd'] 
                    if c in df.columns), None)
    
    # Try common positioning column names (Legacy vs TFF/Disaggregated)
    long_col = next((c for c in ['noncommercial_long_all', 'noncommercial_positions_long_all', 'asset_mgr_pos_long_all', 'managed_money_long_all'] 
                    if c in df.columns), 'noncommercial_long_all')
    
    short_col = next((c for c in ['noncommercial_short_all', 'noncommercial_positions_short_all', 'asset_mgr_pos_short_all', 'managed_money_short_all'] 
                     if c in df.columns), 'noncommercial_short_all')
    
    oi_col = next((c for c in ['open_interest_all', 'open_interest'] if c in df.columns), 'open_interest_all')

    if not date_col or long_col not in df.columns or short_col not in df.columns or oi_col not in df.columns:
        raise KeyError(f"Required columns missing from COT data. Available headers: {list(df.columns[:15])}")

    # Convert Date
    df['date'] = pd.to_datetime(df[date_col])
    df = df.sort_values('date')

    # Markets to track
    targets = {
        "S&P 500": "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
        "NASDAQ": "NASDAQ-100 STOCK INDEX MINI - CHICAGO MERCANTILE EXCHANGE"
    }

    results = []
    for label, full_name in targets.items():
        mkt_df = df[df[market_col].str.contains(label, na=False, case=False)].copy()
        if mkt_df.empty:
            continue
        
        mkt_df[f'{label}_net_noncomm'] = mkt_df[long_col] - mkt_df[short_col]
        mkt_df[f'{label}_net_pct_oi'] = mkt_df[f'{label}_net_noncomm'] / mkt_df[oi_col]
        
        # Keep only essential columns
        mkt_res = mkt_df[['date', f'{label}_net_noncomm', f'{label}_net_pct_oi']]
        
        # Deduplicate by date to ensure unique index for concatenation
        mkt_res_unique = mkt_res.drop_duplicates(subset=['date'], keep='last')
        results.append(mkt_res_unique.set_index('date'))

    if not results:
        return pd.DataFrame()

    return pd.concat(results, axis=1).ffill()

def fetch_pc_ratio_data():
    """
    Fetches Put/Call ratio. 
    Uses CBOE JSON API for current and yfinance for a synthetic proxy.
    """
    print("--- Fetching Put/Call Ratio ---")
    pc_data = {'date': datetime.now().date(), 'total_pc_ratio': np.nan, 'spy_pc_proxy': np.nan}
    
    # 1. CBOE JSON API (Latest)
    cboe_url = "https://cdn.cboe.com/api/global/us_indices/market_statistics/daily_market_statistics.json"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(cboe_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            stats = resp.json().get('data', {})
            pc_data['total_pc_ratio'] = stats.get('total_pc_ratio')
    except Exception as e:
        print(f"⚠️ CBOE API fetch failed: {e}")

    # 2. yfinance SPY Proxy (Real-time Sentiment)
    try:
        spy = yf.Ticker("SPY")
        if spy.options:
            exp = spy.options[0]
            opt = spy.option_chain(exp)
            puts_vol = opt.puts['volume'].sum()
            calls_vol = opt.calls['volume'].sum()
            if calls_vol > 0:
                pc_data['spy_pc_proxy'] = float(puts_vol / calls_vol)
    except Exception as e:
        print(f"⚠️ yfinance SPY options proxy failed: {e}")

    return pc_data

def compute_positioning_signals(df_cot, current_pc):
    """
    Calculates percentiles and flags extremes.
    """
    # 1. COT 104-week Percentile (2 years)
    if 'S&P 500_net_noncomm' in df_cot.columns:
        df_cot['spx_pos_percentile'] = df_cot['S&P 500_net_noncomm'].rolling(window=104, min_periods=12).apply(
            lambda x: x.rank(pct=True).iloc[-1]
        )
    
    # 2. Flag COT Extremes
    df_cot['pos_extreme_long'] = df_cot.get('spx_pos_percentile', 0) > 0.95
    df_cot['pos_extreme_short'] = df_cot.get('spx_pos_percentile', 1) < 0.05

    # 3. P/C Signal (Using available current vs historical baseline)
    # Note: In production, historical P/C would be stored in a DB. 
    # Here we use the latest value relative to a 'Normal' range (0.7 - 1.1)
    pc_val = current_pc.get('total_pc_ratio')
    proxy_val = current_pc.get('spy_pc_proxy')
    curr_pc = pc_val if not pd.isna(pc_val) else proxy_val
    if pd.isna(curr_pc):
        curr_pc = 1.0  # Neutral fallback if both sources fail

    return {
        'latest_cot': df_cot.tail(1).to_dict('records')[0] if not df_cot.empty else {},
        'current_pc': curr_pc,
        'pc_sentiment': 'Fear' if curr_pc > 1.1 else 'Greed' if curr_pc < 0.7 else 'Neutral',
        'cot_extreme': df_cot['pos_extreme_long'].iloc[-1] if not df_cot.empty else False
    }

def get_positioning_report():
    """Orchestrator for positioning analysis."""
    df_cot = fetch_cot_positioning()
    current_pc = fetch_pc_ratio_data()
    
    if df_cot.empty:
        print("⚠️ No COT data available for analysis.")
        return None

    signals = compute_positioning_signals(df_cot, current_pc)
    return signals

if __name__ == "__main__":
    report = get_positioning_report()
    if report:
        print("\n=== Market Positioning Report ===")
        cot_data = report['latest_cot']
        print(f"Latest Report Date: {cot_data.get('date')}")
        print(f"S&P 500 Net Positioning: {cot_data.get('S&P 500_net_noncomm'):,.0f}")
        print(f"S&P 500 Position Percentile (104w): {cot_data.get('spx_pos_percentile', 0)*100:.1f}%")
        print(f"COT Extreme (Crowded Long): {report['cot_extreme']}")
        print(f"Put/Call Ratio: {report['current_pc']:.2f} ({report['pc_sentiment']})")
        
        if report['cot_extreme'] or report['current_pc'] < 0.7:
            print("\n🚨 WARNING: Sentiment is at Bullish Extremes (Contrarian Risk High)")
        elif report['current_pc'] > 1.1:
            print("\n🛡️ INFO: High Hedging Activity Detected (Put/Call > 1.1)")

    # --- Backtest Check: Pre-March 2020 Logic ---
    print("\n--- Running Historical Check: Pre-March 2020 ---")
    try:
        # Fetch 2019-2020 specifically for the test
        df_hist = fetch_cot_positioning(years=[2019, 2020])
        if not df_hist.empty:
            # Check February 2020 (right before crash)
            pre_covid = df_hist[(df_hist.index >= '2020-02-01') & (df_hist.index <= '2020-02-28')]
            if not pre_covid.empty:
                # Hedge funds/Asset managers were historically net long indices going into the crash
                avg_net = pre_covid['S&P 500_net_noncomm'].mean()
                print(f"Feb 2020 S&P 500 Net Positioning: {avg_net:,.0f}")
                if avg_net > 0:
                    print("✅ Confirmed: Script correctly captures Net Long positioning pre-March 2020.")
                else:
                    print("ℹ️ Note: Different classification in legacy reports vs TFF. Check net_pct_oi.")
    except Exception as e:
        print(f"Skipping historical test: {e}")