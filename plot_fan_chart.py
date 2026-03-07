#!/usr/bin/env python3
"""
Monte Carlo Fan Chart
Reconstructs per-campaign Michaelis-Menten response curves via regression,
then plots each simulation as a cumulative-budget path so the fan of outcomes
looks like a classic Monte Carlo price-path chart.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import least_squares

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading simulation results...")
df = pd.read_csv('/home/user/monte_carlo_results.csv')

spend_cols = [c for c in df.columns if c.startswith('spend_')]
campaigns  = [c.replace('spend_', '') for c in spend_cols]
n_camps    = len(campaigns)

X = df[spend_cols].values           # (10000, 14)
y = df['total_conversions'].values  # (10000,)
total_budget = float(X[0].sum())    # ~$11k (constant across sims)

# ── Fit Michaelis-Menten response curves ───────────────────────────────────────
# total_conv ≈ Σ_i  L_i · spend_i / (spend_i + K_i)
# Log-parameterise so L, K stay positive during optimisation.

def predict(X_mat, params):
    out = np.zeros(len(X_mat))
    for i in range(n_camps):
        L = np.exp(params[2 * i])
        K = np.exp(params[2 * i + 1])
        out += L * X_mat[:, i] / (X_mat[:, i] + K)
    return out

def residuals(params):
    return predict(X, params) - y

# Initial guess from baseline spend proportions
baseline = df[df['sim_id'] == 0].iloc[0]
p0 = []
for camp in campaigns:
    spend_0  = max(float(baseline[f'spend_{camp}']), 1.0)
    frac     = spend_0 / total_budget
    conv_est = max(float(baseline['total_conversions']) * frac, 0.01)
    L_guess  = conv_est * 2.5
    K_guess  = max(spend_0 * (L_guess / conv_est - 1.0), 1.0)
    p0.extend([np.log(L_guess), np.log(K_guess)])

print(f"Fitting {n_camps}-campaign response curves on {len(df):,} simulations...")
result     = least_squares(residuals, p0, method='lm', max_nfev=30_000)
params_fit = result.x
r2 = 1 - np.var(residuals(params_fit)) / np.var(y)
print(f"  Fit R² = {r2:.4f}")

L_fit = np.exp(params_fit[0::2])   # shape (14,)
K_fit = np.exp(params_fit[1::2])   # shape (14,)

def campaign_conv_vec(spend_matrix):
    """Vectorised MM conversion estimate: (N, 14) spend → (N, 14) conversions."""
    return L_fit * spend_matrix / (spend_matrix + K_fit)

# ── Build simulation paths ──────────────────────────────────────────────────────
# For each simulation: sort its campaigns by spend DESC (highest first),
# then walk through them accumulating (cumulative_budget, cumulative_conversions).
# This produces 15 points per sim: step 0 = origin, steps 1–14 = one campaign added.

print("Building simulation paths...")
spend_mat = df[spend_cols].values                          # (10000, 14)
conv_mat  = campaign_conv_vec(spend_mat)                   # (10000, 14)
sort_idx  = np.argsort(spend_mat, axis=1)[:, ::-1]        # sort DESC per sim

# Reorder per simulation
spend_sorted = np.take_along_axis(spend_mat, sort_idx, axis=1)
conv_sorted  = np.take_along_axis(conv_mat,  sort_idx, axis=1)

# Prepend zeros for the origin point
zero_col         = np.zeros((len(df), 1))
cum_budget_paths = np.cumsum(np.hstack([zero_col, spend_sorted]), axis=1)  # (N, 15)
cum_conv_raw     = np.cumsum(np.hstack([zero_col, conv_sorted]),  axis=1)  # (N, 15)

# Scale each path's y-axis so its endpoint matches the ACTUAL simulated total
# conversions, not the noisy MM estimate.  This preserves the true fan spread
# while using the MM shape for intermediate steps.
est_total  = conv_mat.sum(axis=1)                          # (N,)
scale      = y / np.maximum(est_total, 1e-6)               # actual / estimated
cum_conv_paths = cum_conv_raw * scale[:, np.newaxis]       # (N, 15)

# ── Identify key scenarios ─────────────────────────────────────────────────────
cap          = df['total_cpa'].quantile(0.97)
clean        = df[df['total_cpa'] < cap]
baseline_cpa = float(baseline['total_cpa'])

max_conv_idx = int(clean['total_conversions'].idxmax())
min_cpa_idx  = int(clean['total_cpa'].idxmin())
constrained  = clean[clean['total_cpa'] <= baseline_cpa * 1.10]
balanced_idx = int(constrained['total_conversions'].idxmax()) if len(constrained) else max_conv_idx

key_scenarios = {
    0:            ('Baseline',        '#ffffff'),
    max_conv_idx: ('Max Conversions', '#00ff88'),
    min_cpa_idx:  ('Min CPA',         '#ff6b35'),
    balanced_idx: ('Balanced',        '#4ecdc4'),
}

# ── Sample paths to plot (3 000 random + all key scenarios) ───────────────────
rng   = np.random.default_rng(42)
all_idx  = np.arange(len(df))
key_idx  = np.array(list(key_scenarios.keys()))
rest_idx = np.setdiff1d(all_idx, key_idx)
sample   = rng.choice(rest_idx, size=min(3_000, len(rest_idx)), replace=False)
plot_idx = np.concatenate([sample, key_idx])

total_convs = df['total_conversions'].values
vmin, vmax  = total_convs.min(), total_convs.max()
norm        = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap        = plt.cm.plasma

# ── Plot ────────────────────────────────────────────────────────────────────────
print("Rendering chart...")
fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')

# Draw background paths sorted so high-conversion paths are on top
order = np.argsort(total_convs[sample])
for rank in order:
    i = sample[rank]
    color = cmap(norm(total_convs[i]))
    ax.plot(cum_budget_paths[i], cum_conv_paths[i],
            color=color, alpha=0.10, linewidth=0.5, zorder=1)

# Draw key scenario paths on top — skip exact duplicates
drawn_idx = set()
label_offsets = {'Baseline': -22, 'Max Conversions': 8, 'Min CPA': -8, 'Balanced': 20}
for orig_idx, (label, color) in key_scenarios.items():
    is_dup = orig_idx in drawn_idx
    drawn_idx.add(orig_idx)
    ax.plot(cum_budget_paths[orig_idx], cum_conv_paths[orig_idx],
            color=color, linewidth=2.5, zorder=10,
            label=None if is_dup else label,
            solid_capstyle='round')
    if not is_dup:
        ax.scatter(cum_budget_paths[orig_idx, -1], cum_conv_paths[orig_idx, -1],
                   color=color, s=80, zorder=11, edgecolors='white', linewidths=0.5)
        y_off = label_offsets.get(label, 0)
        ax.annotate(
            f"{label}  {cum_conv_paths[orig_idx, -1]:.1f} conv",
            xy=(cum_budget_paths[orig_idx, -1], cum_conv_paths[orig_idx, -1]),
            xytext=(12, y_off), textcoords='offset points',
            color=color, fontsize=8.5, va='center', fontweight='bold',
        )

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.85)
cbar.set_label('Total Conversions per Simulation', color='#cccccc', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='#aaaaaa', labelcolor='#aaaaaa')
cbar.outline.set_edgecolor('#444444')

# Axes style
ax.set_xlabel('Cumulative Budget Allocated', color='#cccccc', fontsize=12)
ax.set_ylabel('Estimated Cumulative Conversions', color='#cccccc', fontsize=12)
ax.set_title(
    'Monte Carlo Budget Simulation — 10,000 Allocation Paths\n'
    'Campaigns sorted by spend size; paths coloured by conversion outcome',
    color='white', fontsize=13, pad=16
)
ax.tick_params(colors='#aaaaaa', labelsize=9)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#444444')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

ax.legend(
    facecolor='#1a1a2e', edgecolor='#555555', labelcolor='white',
    fontsize=10, loc='upper left', framealpha=0.9,
)

plt.tight_layout()
out_path = '/home/user/monte_carlo_fan_chart.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
print(f"Saved → {out_path}")
