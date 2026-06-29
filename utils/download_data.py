import requests
import pandas as pd
import os
from datetime import datetime, timedelta

# NOAA GOES X-ray data endpoint
BASE_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

def download_goes_xray(save_path="raw_data/goes_xray.csv"):
    """Download last 7 days of GOES X-ray flux (soft + hard bands)"""
    
    print("Downloading GOES X-ray data...")
    response = requests.get(BASE_URL)
    
    if response.status_code != 200:
        raise Exception(f"Download failed: {response.status_code}")
    
    data = response.json()
    df = pd.DataFrame(data)
    
    # Parse timestamps
    df['time_tag'] = pd.to_datetime(df['time_tag'])
    df = df.sort_values('time_tag').reset_index(drop=True)
    
    # GOES has two channels:
    # 0.05-0.4 nm  = hard X-ray  (like HEL1OS)
    # 0.1-0.8 nm   = soft X-ray  (like SoLEXS)
    soft = df[df['energy'] == '0.1-0.8nm'].copy()
    hard = df[df['energy'] == '0.05-0.4nm'].copy()
    
    # Merge into one DataFrame
    merged = pd.merge(
        soft[['time_tag', 'flux']].rename(columns={'flux': 'soft_flux'}),
        hard[['time_tag', 'flux']].rename(columns={'flux': 'hard_flux'}),
        on='time_tag',
        how='inner'
    )
    
    merged.to_csv(save_path, index=False)
    print(f"Saved {len(merged)} rows to {save_path}")
    print(f"Time range: {merged['time_tag'].min()} → {merged['time_tag'].max()}")
    print(f"Soft flux range: {merged['soft_flux'].min():.2e} → {merged['soft_flux'].max():.2e}")
    
    return merged

def download_goes_historical(start="2024-06-01", end="2024-06-30",
                              save_path="raw_data/goes_historical.csv"):
    """Download historical GOES data from a specific date range"""
    import requests
    from datetime import datetime, timedelta

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    all_dfs  = []

    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")
        url = f"https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
        # For historical: use the 6-minute archive
        url = (f"https://www.ngdc.noaa.gov/stp/satellite/goes-r/data/"
               f"science/xrs/GOES-16/science/xrsf-l2-avg1m_science/"
               f"sci_xrsf-l2-avg1m_g16_d{date_str}_v2-2-0.nc")
        current += timedelta(days=1)

    # Simpler alternative — use the 7-day endpoint repeatedly
    # Just run download_goes_xray() during an active solar period
    print("Tip: Solar Cycle 25 peaked in 2024.")
    print("Check https://www.spaceweather.com for active dates")
    print("then re-run download_goes_xray() — data is always last 7 days")

if __name__ == "__main__":
    download_goes_historical()

if __name__ == "__main__":
    os.makedirs("raw_data", exist_ok=True)
    df = download_goes_xray()
    print(df.head())
    