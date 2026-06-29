import pandas as pd
import numpy as np

# ── Flare class thresholds (soft X-ray peak flux W/m²) ─────────────
FLARE_CLASSES = {
    'X': 1e-4,
    'M': 1e-5,
    'C': 1e-6,
    'B': 1e-7,
    'A': 1e-8,
}

def get_flare_class(flux):
    """Return flare class label for a given soft X-ray flux value"""
    if flux >= 1e-4:   return 'X'
    elif flux >= 1e-5: return 'M'
    elif flux >= 1e-6: return 'C'
    elif flux >= 1e-7: return 'B'
    else:              return 'A'

def add_features(df):
    """
    Engineer all features from raw soft + hard X-ray flux.
    These are the inputs your ML model will learn from.
    """
    df = df.copy().sort_values('time_tag').reset_index(drop=True)

    # ── Log transform (flux spans many orders of magnitude) ────────────
    df['log_soft'] = np.log10(df['soft_flux'].clip(lower=1e-12))
    df['log_hard'] = np.log10(df['hard_flux'].clip(lower=1e-12))

    # ── Flux ratio soft/hard (key precursor signal) ────────────────────
    df['flux_ratio'] = df['soft_flux'] / (df['hard_flux'] + 1e-12)
    df['log_ratio']  = np.log10(df['flux_ratio'].clip(lower=1e-3))

    # ── Rolling statistics (1, 5, 15, 30 min windows) ─────────────────
    # Data is 1-minute cadence, so window=5 means 5 minutes
    for w in [5, 15, 30]:
        df[f'soft_mean_{w}'] = df['log_soft'].rolling(w, min_periods=1).mean()
        df[f'soft_std_{w}']  = df['log_soft'].rolling(w, min_periods=1).std().fillna(0)
        df[f'hard_mean_{w}'] = df['log_hard'].rolling(w, min_periods=1).mean()

    # ── Derivatives (rate of change — rises sharply before flares) ─────
    df['soft_deriv_1']  = df['log_soft'].diff(1)   # 1-min change
    df['soft_deriv_5']  = df['log_soft'].diff(5)   # 5-min change
    df['hard_deriv_1']  = df['log_hard'].diff(1)
    df['hard_deriv_5']  = df['log_hard'].diff(5)

    # ── Rolling max (captures peak tendency) ──────────────────────────
    df['soft_max_15'] = df['log_soft'].rolling(15, min_periods=1).max()
    df['soft_max_30'] = df['log_soft'].rolling(30, min_periods=1).max()

    # ── Fill any NaN from diffs at the start ───────────────────────────
    df = df.fillna(0)
    return df


def label_flares(df, lead_minutes=15):
    df = df.copy()

    # ── Use RELATIVE threshold instead of absolute ─────────────────
    # Flare = flux rises more than 3x above its own 30-min rolling baseline
    # This works regardless of current solar activity level
    baseline = df['soft_flux'].rolling(30, min_periods=1).mean()
    ratio_to_baseline = df['soft_flux'] / (baseline + 1e-12)

    # Nowcast: current flux is 3x above recent baseline (active flare)
    df['nowcast_label'] = (ratio_to_baseline >= 1.3).astype(int)

    # Flare class still uses absolute thresholds (standard classification)
    df['flare_class'] = df['soft_flux'].apply(get_flare_class)

    # Forecast: will flux exceed 3x baseline in next N minutes?
    future_max_ratio = (
        ratio_to_baseline
        .rolling(window=lead_minutes, min_periods=1)
        .max()
        .shift(-lead_minutes)
        .fillna(0)
    )
    df['forecast_label'] = (future_max_ratio >= 1.3).astype(int)

    total = len(df)
    n_now  = df['nowcast_label'].sum()
    n_fore = df['forecast_label'].sum()
    print(f"Total samples   : {total}")
    print(f"Nowcast flares  : {n_now}  ({100*n_now/total:.1f}%)")
    print(f"Forecast flares : {n_fore} ({100*n_fore/total:.1f}%)")
    print(f"Class distribution:\n{df['flare_class'].value_counts()}")
    return df


if __name__ == "__main__":
    df_raw = pd.read_csv("raw_data/goes_xray.csv", parse_dates=['time_tag'])

    print("Adding features...")
    df_feat = add_features(df_raw)

    print("\nLabeling flares (15-min forecast window)...")
    df_labeled = label_flares(df_feat, lead_minutes=15)

    df_labeled.to_csv("data/labeled_dataset.csv", index=False)
    print(f"\nSaved to data/labeled_dataset.csv")
    print(f"Feature columns: {[c for c in df_labeled.columns if c not in ['time_tag','soft_flux','hard_flux']]}")