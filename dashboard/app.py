import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from models.forecast_model import FlareLSTM, SEQ_LEN, FEATURE_COLS
from utils.feature_engineering import add_features

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="SolarSentinel — Aditya-L1",
    page_icon="🛰️",
    layout="wide"
)

# ── Load models (cached so they don't reload every interaction) ───
@st.cache_resource
def load_models():
    # Nowcast: Random Forest
    rf     = joblib.load('models/nowcast_rf.pkl')
    rf_scaler = joblib.load('models/nowcast_scaler.pkl')

    # Forecast: LSTM
    device = torch.device('mps') if torch.backends.mps.is_available() \
             else torch.device('cpu')
    lstm = FlareLSTM(input_size=len(FEATURE_COLS))
    lstm.load_state_dict(torch.load('models/forecast_lstm.pt',
                                    map_location=device))
    lstm.to(device).eval()
    lstm_scaler = joblib.load('models/forecast_scaler.pkl')

    return rf, rf_scaler, lstm, lstm_scaler, device

@st.cache_data(ttl=300)   # refresh data every 5 minutes
def load_latest_data():
    import requests
    url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
    r = requests.get(url, timeout=10)
    data = r.json()
    df = pd.DataFrame(data)
    df['time_tag'] = pd.to_datetime(df['time_tag'])
    soft = df[df['energy'] == '0.1-0.8nm'][
               ['time_tag', 'flux']].rename(columns={'flux': 'soft_flux'})
    hard = df[df['energy'] == '0.05-0.4nm'][
               ['time_tag', 'flux']].rename(columns={'flux': 'hard_flux'})
    merged = pd.merge(soft, hard, on='time_tag').sort_values('time_tag')
    merged = add_features(merged)
    return merged

def get_flare_class(flux):
    if   flux >= 1e-4: return 'X', '#FF0000'
    elif flux >= 1e-5: return 'M', '#FF6600'
    elif flux >= 1e-6: return 'C', '#FFAA00'
    elif flux >= 1e-7: return 'B', '#00AA00'
    else:              return 'A', '#0066FF'

def run_nowcast(df, rf, rf_scaler):
    X = rf_scaler.transform(df[FEATURE_COLS].values)
    proba = rf.predict_proba(X)[:, 1]
    return proba

def run_forecast(df, lstm, lstm_scaler, device):
    X = lstm_scaler.transform(df[FEATURE_COLS].values)
    probas = np.zeros(len(X))
    for i in range(SEQ_LEN, len(X)):
        seq = torch.tensor(
            X[i-SEQ_LEN:i][np.newaxis], dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            probas[i] = lstm(seq).item()
    return probas

# ── UI ────────────────────────────────────────────────────────────
st.title("☀️ SolarSentinel — Aditya-L1 Flare Monitor")
st.caption("SolarSentinel — Real-time solar flare nowcasting & 15-minute forecasting using ML")

# Sidebar controls
st.sidebar.header("Settings")
nowcast_thresh  = st.sidebar.slider("Nowcast alert threshold",
                                    0.1, 0.9, 0.5, 0.05)
forecast_thresh = st.sidebar.slider("Forecast alert threshold",
                                    0.1, 0.9, 0.4, 0.05)
hours_shown     = st.sidebar.slider("Hours of data to show",
                                    6, 72, 24)
auto_refresh    = st.sidebar.checkbox("Auto-refresh every 5 min", True)

# Load everything
with st.spinner("Loading models and fetching live data..."):
    rf, rf_scaler, lstm, lstm_scaler, device = load_models()
    df = load_latest_data()

# Trim to requested hours
cutoff = df['time_tag'].max() - pd.Timedelta(hours=hours_shown)
df_plot = df[df['time_tag'] >= cutoff].copy().reset_index(drop=True)

# Run models
nowcast_proba  = run_nowcast(df_plot, rf, rf_scaler)
forecast_proba = run_forecast(df_plot, lstm, lstm_scaler, device)

# Current status
latest_flux   = df_plot['soft_flux'].iloc[-1]
latest_now    = nowcast_proba[-1]
latest_fore   = forecast_proba[-1]
flare_cls, cls_color = get_flare_class(latest_flux)

# ── Status cards ──────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current Flux", f"{latest_flux:.2e} W/m²")
with col2:
    st.metric("Flare Class", flare_cls)
with col3:
    alert = "🚨 FLARE" if latest_now >= nowcast_thresh else "✅ Quiet"
    st.metric("Nowcast", alert, f"{latest_now*100:.0f}% confidence")
with col4:
    warn = "⚠️ LIKELY" if latest_fore >= forecast_thresh else "🔵 Low risk"
    st.metric("Forecast (15 min)", warn, f"{latest_fore*100:.0f}% probability")

st.divider()

# ── Main chart ────────────────────────────────────────────────────
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    subplot_titles=('X-ray Flux (soft + hard)',
                    'Nowcast Probability',
                    'Forecast Probability (15-min ahead)'),
    vertical_spacing=0.08,
    row_heights=[0.5, 0.25, 0.25]
)

t = df_plot['time_tag']

# Soft X-ray
fig.add_trace(go.Scatter(x=t, y=df_plot['soft_flux'],
    name='Soft X-ray', line=dict(color='#E8593C', width=1)), row=1, col=1)
# Hard X-ray
fig.add_trace(go.Scatter(x=t, y=df_plot['hard_flux'],
    name='Hard X-ray', line=dict(color='#3B8BD4', width=1)), row=1, col=1)

# Flare class lines
for label, val in {'B':1e-7,'C':1e-6,'M':1e-5,'X':1e-4}.items():
    fig.add_hline(y=val, line_dash='dot', line_color='gray',
                  line_width=0.8, annotation_text=label,
                  annotation_position='left', row=1, col=1)

# Nowcast probability
fig.add_trace(go.Scatter(x=t, y=nowcast_proba,
    name='Nowcast prob', fill='tozeroy',
    line=dict(color='#E8593C', width=1),
    fillcolor='rgba(232,89,60,0.2)'), row=2, col=1)
fig.add_hline(y=nowcast_thresh, line_dash='dash',
              line_color='red', line_width=1, row=2, col=1)

# Forecast probability
fig.add_trace(go.Scatter(x=t, y=forecast_proba,
    name='Forecast prob', fill='tozeroy',
    line=dict(color='#F5A623', width=1),
    fillcolor='rgba(245,166,35,0.2)'), row=3, col=1)
fig.add_hline(y=forecast_thresh, line_dash='dash',
              line_color='orange', line_width=1, row=3, col=1)

fig.update_yaxes(type='log', row=1, col=1)
fig.update_yaxes(range=[0,1], row=2, col=1)
fig.update_yaxes(range=[0,1], row=3, col=1)
fig.update_layout(height=600, showlegend=True,
                  margin=dict(l=60, r=20, t=40, b=20))

st.plotly_chart(fig, use_container_width=True)

# ── Model performance summary ──────────────────────────────────────
st.subheader("Model Performance")
perf_col1, perf_col2 = st.columns(2)
with perf_col1:
    st.markdown("**Nowcast Model (Random Forest)**")
    st.table(pd.DataFrame({
        'Metric': ['TSS', 'HSS', 'AUC', 'TPR', 'FPR'],
        'Score':  ['0.9457', '0.8547', '0.9957', '96.7%', '2.2%']
    }))
with perf_col2:
    st.markdown("**Forecast Model (LSTM — 15 min lead)**")
    st.table(pd.DataFrame({
        'Metric': ['TSS', 'HSS', 'AUC', 'TPR', 'FPR'],
        'Score':  ['0.4447', '0.4938', '0.8105', '50.2%', '5.8%']
    }))

# Auto-refresh
if auto_refresh:
    import time
    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    time.sleep(300)
    st.rerun()