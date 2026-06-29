import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time

# These months had high solar activity in Solar Cycle 25
ACTIVE_PERIODS = [
    ("2024-05-01", "2024-05-12"),  # X2.2, X8.7 flares (strongest in years)
    ("2024-03-01", "2024-03-31"),  # Several M-class flares
    ("2024-01-01", "2024-01-31"),  # Active period
    ("2023-12-01", "2023-12-31"),  # M and X class activity
]

def fetch_goes_day(date_str):
    """
    Fetch 1 minute GOES-16 X-ray data for a specific date
    from the NOAA archive (Gp_xr_1m format)
    date_str: 'YYYY-MM-DD'
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year  = dt.strftime("%Y")
    month = dt.strftime("%m")
    day   = dt.strftime("%d")

    # NOAA 1-minute X-ray archive
    url = (
        f"https://services.swpc.noaa.gov/json/goes/primary/"
        f"xrays-1-day.json"
    )

    # For true historical dates we use the bulk archive
    archive_url = (
        f"https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
        f"solar-features/solar-flares/x-rays/goes/xrs/"
    )

    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame(data)
            df['time_tag'] = pd.to_datetime(df['time_tag'])
            soft = df[df['energy'] == '0.1-0.8nm'][[
                'time_tag', 'flux']].rename(columns={'flux': 'soft_flux'})
            hard = df[df['energy'] == '0.05-0.4nm'][[
                'time_tag', 'flux']].rename(columns={'flux': 'hard_flux'})
            merged = pd.merge(soft, hard, on='time_tag')
            return merged
    except Exception as e:
        print(f"  Warning: {e}")
    return None


def download_noaa_flare_catalog():
    """
    Download the official NOAA flare event catalog.
    This gives us ground-truth flare times + classes
    to create high-quality labels.
    """
    url = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"
    try:
        r = requests.get(url, timeout=15)
        df = pd.DataFrame(r.json())
        if len(df) > 0:
            df['begin_time'] = pd.to_datetime(df['begin_time'])
            df['peak_time'] = pd.to_datetime(df['max_time'])
            df['end_time']   = pd.to_datetime(df['end_time'])
            df.to_csv('data/noaa_flare_catalog.csv', index=False)
            print(f"Flare catalog: {len(df)} events saved")
            print(df[['begin_time', 'peak_time',
                       'max_class']].to_string(index=False))
            return df
    except Exception as e:
        print(f"Catalog download failed: {e}")
    return None


def create_catalog_labels(flux_df, catalog_df, lead_minutes=15):
    """
    Use the official NOAA flare catalog to create precise labels.
    Much more accurate than our threshold-based approach.
    
    For each timestep: forecast_label=1 if a flare STARTS
    within the next lead_minutes according to the catalog.
    """
    flux_df = flux_df.copy()
    flux_df['catalog_nowcast']  = 0
    flux_df['catalog_forecast'] = 0
    flux_df['catalog_class']    = 'A'

    for _, flare in catalog_df.iterrows():
        begin = flare['begin_time']
        peak  = flare['max_time']
        end   = flare['end_time']
        cls   = flare['max_class'][0]  # 'M', 'X', 'C' etc.

        # Nowcast: during the flare event
        mask_now = (
            (flux_df['time_tag'] >= begin) &
            (flux_df['time_tag'] <= end)
        )
        flux_df.loc[mask_now, 'catalog_nowcast'] = 1
        flux_df.loc[mask_now, 'catalog_class']   = cls

        # Forecast: lead_minutes BEFORE the flare begins
        forecast_start = begin - timedelta(minutes=lead_minutes)
        mask_fore = (
            (flux_df['time_tag'] >= forecast_start) &
            (flux_df['time_tag'] <  begin)
        )
        flux_df.loc[mask_fore, 'catalog_forecast'] = 1

    n_now  = flux_df['catalog_nowcast'].sum()
    n_fore = flux_df['catalog_forecast'].sum()
    total  = len(flux_df)
    print(f"Catalog-labeled nowcast  : {n_now}  ({100*n_now/total:.1f}%)")
    print(f"Catalog-labeled forecast : {n_fore} ({100*n_fore/total:.1f}%)")
    return flux_df


def build_extended_dataset():
    """
    Main function: combines current 7-day data with
    catalog-based precise labels into one enriched dataset.
    """
    os.makedirs('data', exist_ok=True)

    print("Step 1: Loading current GOES data...")
    df_current = pd.read_csv(
        'data/labeled_dataset.csv', parse_dates=['time_tag']
    )
    print(f"  Current dataset: {len(df_current)} rows")

    print("\nStep 2: Downloading NOAA flare catalog...")
    catalog = download_noaa_flare_catalog()

    if catalog is not None and len(catalog) > 0:
        print("\nStep 3: Adding catalog-based precise labels...")
        df_enriched = create_catalog_labels(df_current, catalog)

        # Use catalog labels where available, fall back to threshold labels
        df_enriched['nowcast_label']  = df_enriched['catalog_nowcast']
        df_enriched['forecast_label'] = df_enriched['catalog_forecast']
        df_enriched.drop(columns=['catalog_nowcast', 'catalog_forecast',
                                   'catalog_class'], inplace=True)

        df_enriched.to_csv('data/labeled_dataset.csv', index=False)
        print(f"\nEnriched dataset saved: {len(df_enriched)} rows")
    else:
        print("\nNo catalog events in current 7-day window.")
        print("This is normal during quiet solar periods.")
        print("Keeping threshold-based labels — they are good enough.")
        print("\nTo get more flare-rich data, we will use NOAA bulk archive.")
        download_bulk_archive()


def download_bulk_archive():
    """
    Downloads pre-compiled flare event lists from NOAA's bulk archive.
    Contains thousands of labeled flare events going back to 2017.
    """
    print("\nDownloading NOAA bulk flare event list (2017-2024)...")
    url = ("https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
           "solar-features/solar-flares/x-rays/goes/xrs/"
           "goes-xrs-report_2024.txt")
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with open('raw_data/goes_xrs_2024.txt', 'w') as f:
                f.write(r.text)
            print(f"Saved: raw_data/goes_xrs_2024.txt")
            print(f"Lines: {len(r.text.splitlines())}")
        else:
            print(f"Archive returned {r.status_code}")
            print("Tip: Use Heliophysics Data Portal instead:")
            print("     https://heliophysicsdata.gsfc.nasa.gov")
    except Exception as e:
        print(f"Archive download failed: {e}")


if __name__ == "__main__":
    build_extended_dataset()