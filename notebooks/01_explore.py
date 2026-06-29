import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Load the downloaded data
df = pd.read_csv("raw_data/goes_xray.csv", parse_dates=['time_tag'])

# ── Plot soft and hard X-ray light curves ──────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

# Soft X-ray (like SoLEXS)
ax1.semilogy(df['time_tag'], df['soft_flux'], color='#E8593C', lw=0.8, label='Soft X-ray (0.1–0.8 nm)')
ax1.set_ylabel('Flux (W/m²)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Flare class reference lines on soft X-ray
classes = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}
for label, val in classes.items():
    ax1.axhline(val, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax1.text(df['time_tag'].iloc[10], val*1.2, label, fontsize=8, color='gray')

# Hard X-ray (like HEL1OS)
ax2.semilogy(df['time_tag'], df['hard_flux'], color='#3B8BD4', lw=0.8, label='Hard X-ray (0.05–0.4 nm)')
ax2.set_ylabel('Flux (W/m²)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Flux ratio — key precursor signal
ratio = df['soft_flux'] / (df['hard_flux'] + 1e-12)
ax3.plot(df['time_tag'], ratio, color='#1D9E75', lw=0.8, label='Soft/Hard ratio')
ax3.set_ylabel('Flux ratio')
ax3.set_xlabel('Time (UTC)')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Format x-axis
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
plt.suptitle('GOES X-ray light curves (last 7 days)', fontsize=13)
plt.tight_layout()
plt.savefig('data/lightcurve_overview.png', dpi=150)
plt.show()
print("Plot saved to data/lightcurve_overview.png")