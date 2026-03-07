#!/usr/bin/env python3
"""
Monte Carlo Budget Optimizer for Google Ads
============================================
Simulates thousands of budget allocations across campaigns or ad groups to
recommend the most effective spend distribution for two objectives:
  1. Maximum Conversion Volume
  2. Minimum Cost Per Acquisition (CPA)

Supports both flat CSVs and 52-week historical data for seasonality-aware
projections. Works at campaign level OR ad group level (auto-detected from
the CSV columns, or forced with --level).

A graph visualising all simulation outcomes is always saved by default.

Usage:
  python monte_carlo_ads.py                              # built-in sample data
  python monte_carlo_ads.py --csv campaigns.csv          # local file (campaigns)
  python monte_carlo_ads.py --csv ad_groups.csv          # auto-detects ad groups
  python monte_carlo_ads.py --csv data.csv --level ad_group   # force ad group level
  python monte_carlo_ads.py --csv https://example.com/data.csv  # remote URL
  python monte_carlo_ads.py --csv weekly.csv --week 47   # plan for week 47
  python monte_carlo_ads.py --generate-sample            # write sample_weekly.csv
  python monte_carlo_ads.py --sims 20000                 # higher precision
  python monte_carlo_ads.py --no-plot                    # skip chart (plot is default)

CSV formats accepted (flat):
  Campaign level : name/campaign, current_spend, impressions, clicks, conversions, avg_cpc
  Ad group level : ad_group, current_spend, impressions, clicks, conversions, avg_cpc

CSV formats accepted (52-week weekly):
  Campaign level : campaign, week, spend, impressions, clicks, conversions, avg_cpc
  Ad group level : ad_group, week, spend, impressions, clicks, conversions, avg_cpc
"""

import argparse
import datetime
import io
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Optional rich terminal output ───────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TimeElapsedColumn
    from rich.table import Table
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


# ══════════════════════════════════════════════════════════════════════════════
# Data Model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Campaign:
    """Holds one campaign's (monthly-average) historical performance metrics."""
    name: str
    current_spend: float   # Monthly spend ($)
    impressions: int
    clicks: int
    conversions: int
    avg_cpc: float

    @property
    def cvr(self) -> float:
        return self.conversions / self.clicks if self.clicks else 0.0

    @property
    def cpa(self) -> float:
        return self.current_spend / self.conversions if self.conversions else float("inf")

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Seasonality Model
# ══════════════════════════════════════════════════════════════════════════════

class SeasonalityModel:
    """
    Derives per-campaign weekly seasonality from 52 weeks of historical data.

    For each campaign and each week:
        seasonal_index[w] = weekly_cvr[w] / mean_annual_cvr

    An index > 1 means that week converts better than average; < 1 worse.

    Also provides:
        cvr_std[campaign]  – empirical week-to-week CVR std dev (replaces the
                             fixed 15% assumption used in the flat-data path)
        trend[campaign]    – linear slope of weekly conversions (positive = growing)
        peak_week[campaign]– the week with the highest seasonal index
    """

    WEEKS = 52

    def __init__(self, weekly_df: pd.DataFrame, group_col: str = "campaign"):
        """
        weekly_df must have columns:
            <group_col>, week (1-52), spend, impressions, clicks, conversions, avg_cpc
        group_col is either "campaign" or "ad_group".
        """
        self.group_col = group_col
        self.indices: Dict[str, np.ndarray] = {}   # entity -> (52,) array
        self.cvr_std: Dict[str, float] = {}
        self.trend:   Dict[str, float] = {}
        self._fit(weekly_df)

    def _fit(self, df: pd.DataFrame) -> None:
        for campaign, grp in df.groupby(self.group_col):
            grp = grp.sort_values("week").copy()
            grp["weekly_cvr"] = np.where(
                grp["clicks"] > 0,
                grp["conversions"] / grp["clicks"],
                np.nan,
            )
            mean_cvr = grp["weekly_cvr"].mean(skipna=True)

            indices = np.ones(self.WEEKS)
            for _, row in grp.iterrows():
                w = int(row["week"]) - 1          # convert to 0-indexed
                if 0 <= w < self.WEEKS and mean_cvr > 0 and not np.isnan(row["weekly_cvr"]):
                    indices[w] = row["weekly_cvr"] / mean_cvr

            self.indices[campaign] = indices
            self.cvr_std[campaign] = float(grp["weekly_cvr"].std(skipna=True) or 0.001)

            # Linear trend over the 52 weeks
            valid = grp.dropna(subset=["weekly_cvr"])
            if len(valid) >= 4:
                slope, *_ = stats.linregress(valid["week"], valid["conversions"])
                self.trend[campaign] = float(slope)
            else:
                self.trend[campaign] = 0.0

    def index_for(self, campaign: str, week: int) -> float:
        """Seasonal index for `campaign` during `week` (1-52)."""
        arr = self.indices.get(campaign)
        if arr is None:
            return 1.0
        return float(arr[(week - 1) % self.WEEKS])

    def noise_sigma(self, campaign: str, base_cvr: float) -> float:
        """
        Log-normal sigma to use in the Monte Carlo noise draw.
        Uses empirical CVR std dev when available, otherwise falls back to 15%.
        """
        std = self.cvr_std.get(campaign, 0.0)
        relative = std / max(base_cvr, 1e-6) if base_cvr > 0 else 0.15
        return float(np.sqrt(np.log1p(relative ** 2)))

    def peak_week(self, campaign: str) -> int:
        """Week number (1-52) with the highest seasonal index."""
        arr = self.indices.get(campaign, np.ones(self.WEEKS))
        return int(np.argmax(arr)) + 1

    def summary_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising key seasonality stats per entity."""
        rows = []
        for camp, idx in self.indices.items():
            rows.append({
                self.group_col: camp,
                "peak_week": int(np.argmax(idx)) + 1,
                "peak_index": float(idx.max()),
                "trough_week": int(np.argmin(idx)) + 1,
                "trough_index": float(idx.min()),
                "cvr_std": self.cvr_std.get(camp, 0.0),
                "trend_slope": self.trend.get(camp, 0.0),
            })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Response Curve  (Michaelis-Menten saturation model)
# ══════════════════════════════════════════════════════════════════════════════

class ResponseCurve:
    """
    Models spend → conversions with diminishing returns:

        conversions(spend) = L · spend / (spend + K)

    L = asymptotic maximum conversions
    K = spend at which half the maximum is reached

    When a SeasonalityModel is provided, the noise sigma is drawn from
    empirical weekly CVR variance rather than a fixed 15% assumption.
    """

    def __init__(
        self,
        campaign: Campaign,
        saturation_multiplier: float = 2.5,
        seasonality: Optional[SeasonalityModel] = None,
    ):
        self.campaign = campaign
        self.seasonality = seasonality

        self.L = max(campaign.conversions * saturation_multiplier, 1.0)

        if campaign.conversions > 0 and campaign.current_spend > 0:
            self.K = campaign.current_spend * (self.L / campaign.conversions - 1.0)
        else:
            self.K = max(campaign.current_spend, 1.0)

        # Noise sigma: empirical if we have weekly data, otherwise ±15%
        if seasonality:
            self._sigma = seasonality.noise_sigma(campaign.name, campaign.cvr)
        else:
            rel = max(campaign.cvr * 0.15, 0.001) / max(campaign.cvr, 1e-6)
            self._sigma = float(np.sqrt(np.log1p(rel ** 2)))

    def predict(self, spend: float, seasonal_multiplier: float = 1.0) -> float:
        """Deterministic prediction (no noise), optionally season-adjusted."""
        base = self.L * spend / (spend + self.K) if (spend + self.K) > 0 else 0.0
        return max(base * seasonal_multiplier, 0.0)

    def sample(
        self,
        spend: float,
        rng: np.random.Generator,
        seasonal_multiplier: float = 1.0,
    ) -> float:
        """
        Stochastic prediction for Monte Carlo draws.
        Log-normal noise keeps results positive and right-skewed (realistic).
        """
        base = self.predict(spend, seasonal_multiplier)
        noise = rng.lognormal(mean=-0.5 * self._sigma ** 2, sigma=self._sigma)
        return max(base * noise, 0.0)

    def marginal_conversion_per_dollar(
        self, spend: float, delta: float = 50.0, seasonal_multiplier: float = 1.0
    ) -> float:
        return (
            self.predict(spend + delta, seasonal_multiplier)
            - self.predict(spend, seasonal_multiplier)
        ) / delta


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo Engine
# ══════════════════════════════════════════════════════════════════════════════

class MonteCarloOptimizer:
    """
    Samples random budget allocations and evaluates each for conversion
    volume and blended CPA.

    When a SeasonalityModel + target_week are provided, every conversion
    prediction is scaled by that week's seasonal index, making the
    recommendations specific to that point in the calendar year.
    """

    def __init__(
        self,
        campaigns: List[Campaign],
        total_budget: float,
        n_simulations: int = 10_000,
        seed: int = 42,
        seasonality: Optional[SeasonalityModel] = None,
        target_week: Optional[int] = None,
    ):
        self.campaigns = campaigns
        self.total_budget = total_budget
        self.n_simulations = n_simulations
        self.seasonality = seasonality
        self.target_week = target_week
        self.n = len(campaigns)
        self.rng = np.random.default_rng(seed)

        self.curves = [ResponseCurve(c, seasonality=seasonality) for c in campaigns]

        # Pre-compute seasonal multipliers for the target week
        if seasonality and target_week:
            self.seasonal_multipliers = [
                seasonality.index_for(c.name, target_week) for c in campaigns
            ]
        else:
            self.seasonal_multipliers = [1.0] * self.n

    def _sample_allocation(self) -> np.ndarray:
        """Dirichlet draw — non-negative splits that sum to total_budget."""
        return self.rng.dirichlet(np.ones(self.n)) * self.total_budget

    def _evaluate(self, allocation: np.ndarray) -> Tuple[float, float]:
        total_conv = sum(
            curve.sample(spend, self.rng, sm)
            for curve, spend, sm in zip(self.curves, allocation, self.seasonal_multipliers)
        )
        cpa = self.total_budget / total_conv if total_conv > 0 else 1e9
        return total_conv, cpa

    def run(self) -> pd.DataFrame:
        """
        Run all simulations. Simulation 0 is always the current allocation
        so it serves as the baseline in comparisons.
        """
        records = []
        current = np.array([c.current_spend for c in self.campaigns])

        for i in range(self.n_simulations):
            alloc = current if i == 0 else self._sample_allocation()
            convs, cpa = self._evaluate(alloc)
            row: Dict = {"sim_id": i, "total_conversions": convs, "total_cpa": cpa}
            for camp, spend in zip(self.campaigns, alloc):
                row[f"spend_{camp.name}"] = spend
                row[f"pct_{camp.name}"] = spend / self.total_budget * 100
            records.append(row)

        return pd.DataFrame(records)

    @staticmethod
    def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
        """O(n log n) Pareto front: maximise conversions & minimise CPA."""
        sorted_df = df.sort_values("total_conversions", ascending=False).reset_index(drop=True)
        pareto, min_cpa = [], float("inf")
        for _, row in sorted_df.iterrows():
            if row["total_cpa"] < min_cpa:
                min_cpa = row["total_cpa"]
                pareto.append(row)
        return pd.DataFrame(pareto).sort_values("total_conversions")

    def find_optima(self, df: pd.DataFrame) -> Dict:
        """
        Returns four allocation scenarios:
            baseline        – current spend (sim_id == 0)
            max_conversions – maximises conversion volume
            min_cpa         – minimises blended CPA
            balanced        – maximises conversions subject to CPA ≤ baseline + 10%
        """
        cap = df["total_cpa"].quantile(0.97)
        clean = df[df["total_cpa"] < cap].copy()
        baseline = df[df["sim_id"] == 0].iloc[0]
        cpa_cap  = baseline["total_cpa"] * 1.10

        max_conv_row = clean.loc[clean["total_conversions"].idxmax()]
        min_cpa_row  = clean.loc[clean["total_cpa"].idxmin()]

        constrained  = clean[clean["total_cpa"] <= cpa_cap]
        balanced_row = (
            constrained.loc[constrained["total_conversions"].idxmax()]
            if len(constrained) else max_conv_row
        )

        return {
            "baseline":        baseline,
            "max_conversions": max_conv_row,
            "min_cpa":         min_cpa_row,
            "balanced":        balanced_row,
            "pareto":          self.pareto_frontier(clean),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

DARK_BG    = "#0f1117"
PANEL_BG   = "#1a1a2e"
GRID_COLOR = "#2a2a3a"
TEXT_COLOR = "#e0e0e0"
GREEN      = "#00ff88"
RED        = "#ff6b6b"
GOLD       = "#ffd700"
BLUE       = "#4fc3f7"
PURPLE     = "#c084fc"


def _ax_style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(True, color=GRID_COLOR, alpha=0.6, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def create_visualizations(
    campaigns: List[Campaign],
    df: pd.DataFrame,
    optima: Dict,
    total_budget: float,
    seasonality: Optional[SeasonalityModel] = None,
    target_week: Optional[int] = None,
    output_file: str = "monte_carlo_results.png",
    entity_label: str = "Campaign",
) -> str:

    has_seasonality = seasonality is not None
    n_rows = 4 if has_seasonality else 3

    fig = plt.figure(figsize=(20, 5 * n_rows))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.45, wspace=0.38)

    names  = [c.name for c in campaigns]
    curves = [ResponseCurve(c, seasonality=seasonality) for c in campaigns]
    sm_map = {c.name: (seasonality.index_for(c.name, target_week) if seasonality and target_week else 1.0)
              for c in campaigns}

    baseline = optima["baseline"]
    max_conv = optima["max_conversions"]
    min_cpa  = optima["min_cpa"]
    balanced = optima["balanced"]
    pareto   = optima["pareto"]

    q97 = df["total_cpa"].quantile(0.97)
    vis = df[df["total_cpa"] < q97]

    # ── Row 0, col 0-1: Monte Carlo scatter ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    _ax_style(ax1)
    sc = ax1.scatter(
        vis["total_conversions"], vis["total_cpa"],
        c=vis["total_conversions"], cmap="viridis",
        alpha=0.25, s=6, rasterized=True,
    )
    if len(pareto):
        ax1.plot(pareto["total_conversions"], pareto["total_cpa"],
                 "w--", lw=1.5, alpha=0.7, label="Pareto frontier", zorder=3)
    for row, color, marker, size, label in [
        (baseline, "white", "D", 120, "Current"),
        (max_conv, GREEN,   "*", 220, "Max Conversions"),
        (min_cpa,  RED,     "*", 220, "Min CPA"),
        (balanced, GOLD,    "*", 220, "Balanced"),
    ]:
        ax1.scatter(row["total_conversions"], row["total_cpa"],
                    color=color, s=size, marker=marker, zorder=5,
                    label=label, edgecolors="none")
    week_label = f" — Week {target_week}" if target_week else ""
    ax1.set_xlabel("Total Conversions", color=TEXT_COLOR, fontsize=9)
    ax1.set_ylabel("Blended CPA ($)",   color=TEXT_COLOR, fontsize=9)
    ax1.set_title(f"Monte Carlo: Conversion Volume vs CPA Trade-off{week_label}",
                  color=TEXT_COLOR, fontsize=12, pad=8)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    cb = plt.colorbar(sc, ax=ax1)
    cb.set_label("Conversions", color=TEXT_COLOR, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)

    # ── Row 0, col 2: Conversion histogram ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    _ax_style(ax2)
    ax2.hist(vis["total_conversions"], bins=70, color=GREEN, alpha=0.75, edgecolor="none")
    ax2.axvline(baseline["total_conversions"], color="white", ls="--", lw=1.5, label="Current")
    ax2.axvline(max_conv["total_conversions"], color=GREEN,   ls="-",  lw=2,   label="Max Conv")
    ax2.axvline(balanced["total_conversions"], color=GOLD,    ls="-",  lw=1.5, label="Balanced")
    ax2.set_xlabel("Total Conversions", color=TEXT_COLOR, fontsize=8)
    ax2.set_ylabel("Frequency",          color=TEXT_COLOR, fontsize=8)
    ax2.set_title("Conversion Distribution", color=TEXT_COLOR, fontsize=11)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=7)

    # ── Row 1: Budget allocation bars ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    _ax_style(ax3)
    x, w = np.arange(len(names)), 0.20
    for offset, (label, row, color) in zip(
        [-1.5, -0.5, 0.5, 1.5],
        [("Current", baseline, "white"), ("Max Conv", max_conv, GREEN),
         ("Min CPA", min_cpa, RED),      ("Balanced", balanced, GOLD)],
    ):
        ax3.bar(x + offset * w, [row[f"spend_{n}"] for n in names],
                w, label=label, color=color, alpha=0.82)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, color=TEXT_COLOR, rotation=20, ha="right", fontsize=8)
    ax3.set_ylabel("Monthly Spend ($)", color=TEXT_COLOR, fontsize=9)
    ax3.set_title(f"Budget Allocation by Optimisation Scenario ({entity_label} level)",
                  color=TEXT_COLOR, fontsize=12)
    ax3.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    # ── Row 2, col 0-1: Response curves ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :2])
    _ax_style(ax4)
    spend_range = np.linspace(0, total_budget * 0.75, 300)
    palette = plt.cm.tab10(np.linspace(0, 1, len(campaigns)))
    for camp, curve, color in zip(campaigns, curves, palette):
        sm = sm_map[camp.name]
        preds = [curve.predict(s, sm) for s in spend_range]
        ax4.plot(spend_range, preds, color=color, lw=2, label=camp.name)
        ax4.axvline(camp.current_spend, color=color, ls=":", lw=1, alpha=0.5)
        ax4.scatter([camp.current_spend], [curve.predict(camp.current_spend, sm)],
                    color=color, s=50, zorder=4, edgecolors="white", lw=0.5)
    curve_title = "Response Curves — Diminishing Returns"
    if target_week:
        curve_title += f" (Week {target_week} seasonal adjustment)"
    ax4.set_xlabel("Spend ($)", color=TEXT_COLOR, fontsize=9)
    ax4.set_ylabel("Expected Conversions", color=TEXT_COLOR, fontsize=9)
    ax4.set_title(curve_title, color=TEXT_COLOR, fontsize=11)
    ax4.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
               fontsize=7, ncol=2)

    # ── Row 2, col 2: CPA histogram ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    _ax_style(ax5)
    ax5.hist(vis["total_cpa"], bins=70, color=RED, alpha=0.75, edgecolor="none")
    ax5.axvline(baseline["total_cpa"], color="white", ls="--", lw=1.5, label="Current")
    ax5.axvline(min_cpa["total_cpa"],  color=RED,     ls="-",  lw=2,   label="Min CPA")
    ax5.axvline(balanced["total_cpa"], color=GOLD,    ls="-",  lw=1.5, label="Balanced")
    ax5.set_xlabel("Blended CPA ($)", color=TEXT_COLOR, fontsize=8)
    ax5.set_ylabel("Frequency",        color=TEXT_COLOR, fontsize=8)
    ax5.set_title("CPA Distribution",  color=TEXT_COLOR, fontsize=11)
    ax5.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=7)

    # ── Row 3 (seasonal only): 52-week seasonality index chart ────────────────
    if has_seasonality:
        ax6 = fig.add_subplot(gs[3, :2])
        _ax_style(ax6)
        weeks = np.arange(1, 53)
        for camp, color in zip(campaigns, palette):
            idx = seasonality.indices[camp.name]
            ax6.plot(weeks, idx, color=color, lw=1.8, label=camp.name, alpha=0.9)
        ax6.axhline(1.0, color="white", ls="--", lw=1, alpha=0.4, label="Baseline (1.0)")
        if target_week:
            ax6.axvline(target_week, color=PURPLE, ls="-", lw=2, alpha=0.8,
                        label=f"Target week {target_week}")
        ax6.set_xlabel("Week of Year", color=TEXT_COLOR, fontsize=9)
        ax6.set_ylabel("Seasonal Index", color=TEXT_COLOR, fontsize=9)
        ax6.set_title("Weekly Seasonality Indices (1.0 = annual average)",
                      color=TEXT_COLOR, fontsize=12)
        ax6.set_xlim(1, 52)
        ax6.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                   fontsize=7, ncol=2)

        # ── Row 3, col 2: per-campaign seasonal index for target week ──────────
        ax7 = fig.add_subplot(gs[3, 2])
        _ax_style(ax7)
        if target_week:
            indices = [seasonality.index_for(c.name, target_week) for c in campaigns]
            colors  = [GREEN if i >= 1.0 else RED for i in indices]
            bars = ax7.barh(names, indices, color=colors, alpha=0.8)
            ax7.axvline(1.0, color="white", ls="--", lw=1.2)
            for bar, idx_val in zip(bars, indices):
                ax7.text(idx_val + 0.01, bar.get_y() + bar.get_height() / 2,
                         f"{idx_val:.2f}×", va="center", color=TEXT_COLOR, fontsize=8)
            ax7.set_xlabel("Seasonal Index", color=TEXT_COLOR, fontsize=8)
            ax7.set_title(f"Week {target_week} Index per {entity_label}",
                          color=TEXT_COLOR, fontsize=10)
        else:
            # Show peak index per entity
            peaks = [seasonality.indices[c.name].max() for c in campaigns]
            peak_weeks = [seasonality.peak_week(c.name) for c in campaigns]
            ax7.barh(names, peaks, color=GOLD, alpha=0.8)
            ax7.axvline(1.0, color="white", ls="--", lw=1.2)
            for i, (bar, pw) in enumerate(zip(ax7.patches, peak_weeks)):
                ax7.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                         f"wk {pw}", va="center", color=TEXT_COLOR, fontsize=7)
            ax7.set_xlabel("Peak Seasonal Index", color=TEXT_COLOR, fontsize=8)
            ax7.set_title(f"Peak Seasonal Index per {entity_label}", color=TEXT_COLOR, fontsize=10)

    title = f"Google Ads — Monte Carlo Budget Optimisation  ({entity_label} level)"
    if target_week:
        title += f"  |  Projected for Week {target_week}"
    fig.suptitle(title, color=TEXT_COLOR, fontsize=15, fontweight="bold", y=0.998)
    plt.savefig(output_file, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return output_file


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def print_report(
    campaigns: List[Campaign],
    df: pd.DataFrame,
    optima: Dict,
    total_budget: float,
    seasonality: Optional[SeasonalityModel] = None,
    target_week: Optional[int] = None,
    entity_label: str = "Campaign",
) -> None:
    if HAS_RICH:
        _rich_report(campaigns, df, optima, total_budget, seasonality, target_week, entity_label)
    else:
        _plain_report(campaigns, df, optima, total_budget, entity_label)


def _rich_report(campaigns, df, optima, total_budget, seasonality, target_week, entity_label="Campaign"):
    baseline = optima["baseline"]
    max_conv = optima["max_conversions"]
    min_cpa  = optima["min_cpa"]
    balanced = optima["balanced"]
    names    = [c.name for c in campaigns]

    subtitle = f"Total budget: [bold]${total_budget:,.0f}[/bold]   Simulations: [bold]{len(df):,}[/bold]"
    if target_week:
        subtitle += f"   Target week: [bold]{target_week}[/bold]"
    if seasonality:
        subtitle += "   [green]Seasonality: ON[/green]"

    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Google Ads — Monte Carlo Budget Optimiser  ({entity_label} level)[/bold cyan]\n[dim]{subtitle}[/dim]",
        border_style="cyan",
    ))

    # ── Seasonality summary ───────────────────────────────────────────────────
    if seasonality:
        console.print("\n[bold yellow]52-Week Seasonality Summary[/bold yellow]")
        st = seasonality.summary_table()
        t0 = Table(header_style="bold white", border_style="dim")
        t0.add_column(entity_label,    style="cyan", min_width=22)
        t0.add_column("Peak Week",     justify="right")
        t0.add_column("Peak Index",    justify="right")
        t0.add_column("Trough Week",   justify="right")
        t0.add_column("Trough Index",  justify="right")
        t0.add_column("CVR Std Dev",   justify="right")
        t0.add_column("Trend",         justify="right")
        if target_week:
            t0.add_column(f"Wk {target_week} Index", justify="right", style="magenta")

        for _, row in st.iterrows():
            trend_str = f"{'↑' if row['trend_slope'] > 0 else '↓'}{abs(row['trend_slope']):.1f}/wk"
            extra = []
            if target_week:
                wi = seasonality.index_for(row[seasonality.group_col], target_week)
                color = "green" if wi >= 1.0 else "red"
                extra = [f"[{color}]{wi:.2f}×[/{color}]"]
            t0.add_row(
                row[seasonality.group_col],
                str(int(row["peak_week"])),
                f"{row['peak_index']:.2f}×",
                str(int(row["trough_week"])),
                f"{row['trough_index']:.2f}×",
                f"{row['cvr_std']:.4f}",
                trend_str,
                *extra,
            )
        console.print(t0)

    # ── Current performance ───────────────────────────────────────────────────
    console.print(f"\n[bold yellow]Current {entity_label} Performance[/bold yellow]")
    t1 = Table(header_style="bold white", border_style="dim")
    for col, just in [(entity_label,"left"),("Spend","right"),("Impressions","right"),
                       ("Clicks","right"),("Conversions","right"),
                       ("CVR","right"),("CPA","right"),("Avg CPC","right")]:
        t1.add_column(col, justify=just)
    for c in campaigns:
        t1.add_row(c.name, f"${c.current_spend:,.0f}", f"{c.impressions:,}",
                   f"{c.clicks:,}", f"{c.conversions:,}",
                   f"{c.cvr:.2%}", f"${c.cpa:.2f}", f"${c.avg_cpc:.2f}")
    tc  = sum(c.conversions for c in campaigns)
    tk  = sum(c.clicks for c in campaigns)
    t1.add_row("[bold]TOTAL[/bold]", f"[bold]${total_budget:,.0f}[/bold]",
               f"[bold]{sum(c.impressions for c in campaigns):,}[/bold]",
               f"[bold]{tk:,}[/bold]", f"[bold]{tc:,}[/bold]",
               f"[bold]{tc/tk:.2%}[/bold]" if tk else "—",
               f"[bold]${total_budget/tc:.2f}[/bold]" if tc else "—", "")
    console.print(t1)

    # ── Simulation stats ──────────────────────────────────────────────────────
    console.print("\n[bold yellow]Monte Carlo Simulation Statistics[/bold yellow]")
    cap = df["total_cpa"].quantile(0.97)
    vis = df[df["total_cpa"] < cap]
    t2 = Table(header_style="bold white", border_style="dim")
    for col in ["Metric","Min","P25","Median","P75","Max","Std Dev"]:
        t2.add_column(col, justify="right" if col != "Metric" else "left",
                      style="cyan" if col == "Metric" else "")
    for col, label, fmt in [
        ("total_conversions", "Conversions", lambda v: f"{v:.0f}"),
        ("total_cpa",         "CPA ($)",     lambda v: f"${v:.2f}"),
    ]:
        s = vis[col]
        t2.add_row(label, fmt(s.min()), fmt(s.quantile(.25)), fmt(s.median()),
                   fmt(s.quantile(.75)), fmt(s.max()), fmt(s.std()))
    console.print(t2)

    # ── Optimal allocations ───────────────────────────────────────────────────
    console.print("\n[bold yellow]Recommended Budget Allocations[/bold yellow]")
    t3 = Table(header_style="bold white", border_style="dim", show_lines=True)
    t3.add_column(entity_label,   style="cyan", min_width=22)
    t3.add_column("Current ($)",  justify="right")
    t3.add_column("Max Conv ($)", justify="right", style="green")
    t3.add_column("Δ Max Conv",   justify="right", style="green")
    t3.add_column("Min CPA ($)",  justify="right", style="red")
    t3.add_column("Δ Min CPA",    justify="right", style="red")
    t3.add_column("Balanced ($)", justify="right", style="yellow")
    t3.add_column("Δ Balanced",   justify="right", style="yellow")

    def _d(new, old):
        d = new - old
        return f"{'+'if d>=0 else ''}${d:,.0f}"

    def _pct(new, old):
        if old == 0: return "—"
        d = (new - old) / old * 100
        return f"{'+'if d>=0 else ''}{d:.1f}%"

    for name in names:
        cur, mc, mp, ba = (baseline[f"spend_{name}"], max_conv[f"spend_{name}"],
                           min_cpa[f"spend_{name}"],  balanced[f"spend_{name}"])
        t3.add_row(name, f"${cur:,.0f}",
                   f"${mc:,.0f}", _d(mc,cur),
                   f"${mp:,.0f}", _d(mp,cur),
                   f"${ba:,.0f}", _d(ba,cur))

    for label, b_val, mc_val, mp_val, ba_val, fmt in [
        ("Expected Conversions",
         baseline["total_conversions"], max_conv["total_conversions"],
         min_cpa["total_conversions"],  balanced["total_conversions"],
         lambda v: f"{v:.0f}"),
        ("Blended CPA ($)",
         baseline["total_cpa"], max_conv["total_cpa"],
         min_cpa["total_cpa"],  balanced["total_cpa"],
         lambda v: f"${v:.2f}"),
    ]:
        t3.add_row(f"[bold]{label}[/bold]",
                   f"[bold]{fmt(b_val)}[/bold]",
                   f"[bold green]{fmt(mc_val)}[/bold green]",
                   f"[bold green]{_pct(mc_val, b_val)}[/bold green]",
                   f"[bold red]{fmt(mp_val)}[/bold red]",
                   f"[bold red]{_pct(mp_val, b_val)}[/bold red]",
                   f"[bold yellow]{fmt(ba_val)}[/bold yellow]",
                   f"[bold yellow]{_pct(ba_val, b_val)}[/bold yellow]")
    console.print(t3)

    # ── Insights ──────────────────────────────────────────────────────────────
    console.print("\n[bold yellow]Key Insights & Recommendations[/bold yellow]")
    b_conv, b_cpa = baseline["total_conversions"], baseline["total_cpa"]
    console.print(
        f"  [green]● Max Conversion[/green] "
        f"+{(max_conv['total_conversions']-b_conv)/b_conv*100:.1f}% volume "
        f"({b_conv:.0f} → {max_conv['total_conversions']:.0f}) "
        f"at CPA ${max_conv['total_cpa']:.2f}"
    )
    console.print(
        f"  [red]● Min CPA[/red] "
        f"{(b_cpa-min_cpa['total_cpa'])/b_cpa*100:.1f}% cheaper "
        f"(${b_cpa:.2f} → ${min_cpa['total_cpa']:.2f}) "
        f"with {min_cpa['total_conversions']:.0f} conversions"
    )
    console.print(
        f"  [yellow]● Balanced[/yellow] "
        f"+{(balanced['total_conversions']-b_conv)/b_conv*100:.1f}% conversions "
        f"while holding CPA ≤ 10% above current (${balanced['total_cpa']:.2f})"
    )
    if seasonality and target_week:
        console.print(f"\n  [magenta]● Seasonality note:[/magenta] "
                      f"Week {target_week} indices applied — "
                      f"campaigns above 1.0× are in peak season, below 1.0× are in trough.")
    console.print("\n  [dim]Per-campaign shift (current → Max Conv):[/dim]")
    for name in names:
        cur  = baseline[f"spend_{name}"]
        new  = max_conv[f"spend_{name}"]
        diff = new - cur
        pct  = diff / cur * 100 if cur else 0
        color = "green" if diff > 0 else "red"
        console.print(
            f"    [{color}]{'↑' if diff>0 else '↓'} {name}[/{color}]: "
            f"${cur:,.0f} → ${new:,.0f} ([bold]{pct:+.1f}%[/bold])"
        )
    console.print()


def _plain_report(campaigns, df, optima, total_budget, entity_label="Campaign"):
    baseline, max_conv, min_cpa, balanced = (
        optima["baseline"], optima["max_conversions"],
        optima["min_cpa"],  optima["balanced"],
    )
    names = [c.name for c in campaigns]
    print("\n" + "=" * 72)
    print(f"  GOOGLE ADS MONTE CARLO BUDGET OPTIMISER  ({entity_label.upper()} LEVEL)")
    print(f"  Budget: ${total_budget:,.0f}   Simulations: {len(df):,}")
    print("=" * 72)
    print(f"\n{entity_label:<25} {'Spend':>10} {'Convs':>8} {'CPA':>9} {'CVR':>7}")
    print("-" * 62)
    for c in campaigns:
        print(f"{c.name:<25} ${c.current_spend:>9,.0f} {c.conversions:>8,}"
              f" ${c.cpa:>8.2f} {c.cvr:>6.2%}")
    print(f"\n{entity_label:<25} {'Current':>10} {'MaxConv':>10} {'MinCPA':>10} {'Balanced':>10}")
    print("-" * 68)
    for name in names:
        print(f"{name:<25} ${baseline[f'spend_{name}']:>9,.0f}"
              f" ${max_conv[f'spend_{name}']:>9,.0f}"
              f" ${min_cpa[f'spend_{name}']:>9,.0f}"
              f" ${balanced[f'spend_{name}']:>9,.0f}")
    print(f"\n{'Expected Conversions':<25} {baseline['total_conversions']:>10.0f}"
          f" {max_conv['total_conversions']:>10.0f}"
          f" {min_cpa['total_conversions']:>10.0f}"
          f" {balanced['total_conversions']:>10.0f}")
    print(f"{'Blended CPA ($)':<25} ${baseline['total_cpa']:>9.2f}"
          f" ${max_conv['total_cpa']:>9.2f}"
          f" ${min_cpa['total_cpa']:>9.2f}"
          f" ${balanced['total_cpa']:>9.2f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def sample_campaigns() -> List[Campaign]:
    """Built-in flat sample data (6 campaigns, ~$55k/month)."""
    return [
        Campaign("Brand Search",      8_000,  250_000,   15_000,  900, 0.53),
        Campaign("Generic Search",   15_000,  800_000,   12_000,  480, 1.25),
        Campaign("Competitor Search", 6_000,  150_000,    4_500,  135, 1.33),
        Campaign("Display Remarketing",5_000,2_000_000,   6_000,  180, 0.83),
        Campaign("Performance Max",  12_000,  500_000,    8_000,  400, 1.50),
        Campaign("Shopping",          9_000,  300_000,    9_000,  360, 1.00),
    ]


def _base_seasonal_index(week: int) -> float:
    """
    Retail-like annual seasonality curve.
    Peak ≈ weeks 47-52 (holiday), trough ≈ weeks 1-4 (January) and 26-28 (summer).
    """
    w = week
    # Smooth cosine background (peak at week 50)
    bg = 1.0 + 0.12 * np.cos(2 * np.pi * (w - 50) / 52)
    # Holiday ramp (weeks 44-52)
    if 44 <= w <= 52:
        bg *= 1.0 + 0.07 * (w - 43)
    # Back-to-school (weeks 32-36)
    if 32 <= w <= 36:
        bg *= 1.10
    # Valentine's (week 7)
    if w == 7:
        bg *= 1.12
    # Memorial Day (week 21)
    if 20 <= w <= 22:
        bg *= 1.08
    # Summer dip (weeks 25-30)
    if 25 <= w <= 30:
        bg *= 0.87
    # January lull (weeks 1-4)
    if 1 <= w <= 4:
        bg *= 0.88
    return max(bg, 0.1)


def generate_weekly_sample_data(path: str = "sample_campaigns_weekly.csv", seed: int = 0) -> None:
    """
    Writes a realistic 52-week × 6-campaign dataset to `path`.

    Seasonality profiles per campaign:
        Brand Search        — low seasonality  (mostly stable, slight holiday lift)
        Generic Search      — moderate          (holiday + back-to-school)
        Competitor Search   — moderate          (holiday driven)
        Display Remarketing — high              (very holiday-driven)
        Performance Max     — moderate-high
        Shopping            — very high         (extreme holiday peak)
    """
    rng = np.random.default_rng(seed)

    # (base_monthly_spend, impressions/wk, clicks/wk, conversions/wk, avg_cpc, amplitude)
    campaign_defs = {
        "Brand Search":        (8_000,  58_000, 3_460, 208, 0.53, 0.06),
        "Generic Search":      (15_000, 185_000, 2_770, 111, 1.25, 0.14),
        "Competitor Search":   (6_000,  34_600, 1_040,  31, 1.33, 0.16),
        "Display Remarketing": (5_000,  462_000, 1_385,  42, 0.83, 0.22),
        "Performance Max":     (12_000, 115_000, 1_846,  92, 1.50, 0.15),
        "Shopping":            (9_000,  69_200,  2_077,  83, 1.00, 0.28),
    }

    rows = []
    for camp_name, (monthly_spend, wk_imp, wk_clicks, wk_conv, avg_cpc, amplitude) in campaign_defs.items():
        weekly_spend_base = monthly_spend * 12 / 52

        for week in range(1, 53):
            si = _base_seasonal_index(week)
            # Scale amplitude: Shopping swings more, Brand less
            si = 1.0 + (si - 1.0) * (amplitude / 0.14)

            noise = rng.normal(1.0, 0.04)

            spend       = round(weekly_spend_base * si * noise, 2)
            impressions = max(1, int(wk_imp    * si * rng.normal(1.0, 0.05)))
            clicks      = max(1, int(wk_clicks * si * rng.normal(1.0, 0.06)))
            conversions = max(0, int(wk_conv   * si * rng.normal(1.0, 0.08)))
            cpc         = round(avg_cpc * rng.normal(1.0, 0.05), 3)

            rows.append({
                "campaign":    camp_name,
                "week":        week,
                "spend":       spend,
                "impressions": impressions,
                "clicks":      clicks,
                "conversions": conversions,
                "avg_cpc":     cpc,
            })

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Sample weekly data written to: {path}  ({len(rows)} rows)")


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """
    Parse CSV/TSV bytes, auto-detecting:
      - Encoding: UTF-16 BOM, UTF-8 BOM, UTF-8, Latin-1
      - Separator: tab vs comma
      - Metadata header rows (Google Ads exports prepend title + date-range lines)
      - Thousands-comma formatting in numeric cells
    """
    # Decode
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        text = raw.decode('utf-16')
    else:
        for enc in ('utf-8-sig', 'utf-8', 'latin-1'):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not decode CSV — unsupported encoding.")

    lines = text.splitlines()

    # Skip metadata rows: advance until a line with ≥3 delimited fields
    skip = 0
    for i, line in enumerate(lines):
        sep_candidate = '\t' if '\t' in line else ','
        if len(line.split(sep_candidate)) >= 3:
            skip = i
            break

    header_line = lines[skip]
    sep = '\t' if '\t' in header_line else ','
    buf = '\n'.join(lines[skip:])
    return pd.read_csv(io.StringIO(buf), sep=sep, thousands=',')


def load_campaigns_from_csv(
    path: str, level: str = "auto"
) -> Tuple[List[Campaign], Optional[SeasonalityModel], str]:
    """
    Auto-detects flat vs weekly CSV format, and campaign vs ad group level.
    Accepts a local file path or an HTTP/HTTPS URL.

    Flat format:
        name/campaign/ad_group, current_spend, impressions, clicks, conversions, avg_cpc

    Weekly format (52-week):
        campaign/ad_group, week, spend, impressions, clicks, conversions, avg_cpc

    level: "auto" (default), "campaign", or "ad_group".

    Returns (campaigns, seasonality_model, entity_label).
    seasonality_model is None for flat CSVs.
    entity_label is "Campaign" or "Ad Group".
    """
    if path.startswith("http://") or path.startswith("https://"):
        with urllib.request.urlopen(path) as resp:
            raw = resp.read()
        df = _read_csv_bytes(raw)
    else:
        df = _read_csv_bytes(open(path, "rb").read())
    df.columns = df.columns.str.strip().str.lower()

    # Normalise Google Ads export column names → internal names
    df = df.rename(columns={
        "cost":             "spend",
        "impr.":            "impressions",
        "impressions.":     "impressions",
        "avg. cpc":         "avg_cpc",
        "avg cpc":          "avg_cpc",
        "ad group":         "ad_group",
        "ad group name":    "ad_group",
        "adgroup":          "ad_group",
        "ad_group_name":    "ad_group",
        "campaign name":    "campaign",
    })
    # Drop non-essential Google Ads columns we don't use
    df = df.drop(columns=[c for c in ("currency code",) if c in df.columns])

    # Convert date-based week column (e.g. "2026-01-12") to ISO week number
    if "week" in df.columns and not pd.api.types.is_integer_dtype(df["week"]):
        parsed = pd.to_datetime(df["week"], errors="coerce")
        if parsed.notna().any():
            df["week"] = parsed.dt.isocalendar().week.astype(int)

    # Round conversions to int if they came in as floats (Google Ads uses decimals)
    if "conversions" in df.columns:
        df["conversions"] = df["conversions"].round().astype(int)

    # Resolve group column: explicit level flag > auto-detect from columns
    if level == "ad_group":
        group_col = "ad_group"
    elif level == "campaign":
        group_col = "campaign"
    else:  # auto
        group_col = "ad_group" if "ad_group" in df.columns else "campaign"

    if "week" in df.columns:
        campaigns, seasonality, entity_label = _load_weekly(df, group_col=group_col)
        return campaigns, seasonality, entity_label
    else:
        campaigns, entity_label = _load_flat(df, group_col=group_col)
        return campaigns, None, entity_label


def _load_flat(df: pd.DataFrame, group_col: Optional[str] = None) -> Tuple[List[Campaign], str]:
    """
    Load a flat (non-weekly) CSV.  The entity name column is auto-detected as the
    first present column from: group_col hint → "ad_group" → "campaign" → "name".
    Returns (campaigns, entity_label).
    """
    # Resolve the name column
    for candidate in ([group_col] if group_col else []) + ["ad_group", "campaign", "name"]:
        if candidate and candidate in df.columns:
            name_col = candidate
            break
    else:
        raise ValueError(
            "CSV must have a 'name', 'campaign', or 'ad_group' column."
        )

    entity_label = "Ad Group" if name_col == "ad_group" else "Campaign"

    required = {name_col, "current_spend", "impressions", "clicks", "conversions", "avg_cpc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    campaigns = []
    for _, row in df.iterrows():
        campaigns.append(Campaign(
            name=str(row[name_col]).strip(),
            current_spend=float(row["current_spend"]),
            impressions=int(row["impressions"]),
            clicks=int(row["clicks"]),
            conversions=int(row["conversions"]),
            avg_cpc=float(row["avg_cpc"]),
        ))
    return campaigns, entity_label


def _load_weekly(
    df: pd.DataFrame, group_col: str = "campaign"
) -> Tuple[List[Campaign], SeasonalityModel, str]:
    """
    Load a 52-week weekly CSV grouped by `group_col` ("campaign" or "ad_group").
    Returns (campaigns, seasonality, entity_label).
    """
    entity_label = "Ad Group" if group_col == "ad_group" else "Campaign"
    required = {group_col, "week", "spend", "impressions", "clicks", "conversions", "avg_cpc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Weekly CSV missing columns: {missing}")

    seasonality = SeasonalityModel(df, group_col=group_col)

    campaigns = []
    for name, grp in df.groupby(group_col):
        # Use the last 4 weeks as the "current" operating point
        recent = grp.sort_values("week").tail(4)
        weekly_spend = recent["spend"].mean()
        monthly_spend = weekly_spend * (52 / 12)  # annualise then monthly

        campaigns.append(Campaign(
            name=str(name).strip(),
            current_spend=round(monthly_spend, 2),
            impressions=int(grp["impressions"].sum() / 52 * (52 / 12)),
            clicks=int(grp["clicks"].sum() / 52 * (52 / 12)),
            conversions=int(grp["conversions"].sum() / 52 * (52 / 12)),
            avg_cpc=float(grp["avg_cpc"].mean()),
        ))

    return campaigns, seasonality, entity_label


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Monte Carlo budget optimiser for Google Ads campaigns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--csv",    metavar="FILE_OR_URL", help="Campaign/ad-group CSV path or HTTP(S) URL (flat or 52-week weekly)")
    p.add_argument("--level",  metavar="LEVEL", default="auto",
                   choices=["auto", "campaign", "ad_group"],
                   help="Grouping level: 'campaign' (default), 'ad_group', or 'auto' (detect from CSV columns)")
    p.add_argument("--week",   metavar="N",    type=int,
                   help="Target week of year to plan for (1-52). "
                        "Requires weekly CSV. Defaults to current ISO week.")
    p.add_argument("--sims",   metavar="N",    type=int,    default=10_000,
                   help="Number of simulations (default: 10000)")
    p.add_argument("--seed",   metavar="N",    type=int,    default=42)
    p.add_argument("--out",    metavar="FILE",              default="monte_carlo_results.png",
                   help="Output chart filename")
    p.add_argument("--no-plot",  action="store_true", help="Skip chart generation")
    p.add_argument("--generate-sample", action="store_true",
                   help="Write sample_campaigns_weekly.csv and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.generate_sample:
        generate_weekly_sample_data()
        return

    seasonality: Optional[SeasonalityModel] = None
    target_week: Optional[int] = None
    entity_label: str = "Campaign"

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.csv:
        print(f"Loading: {args.csv}")
        campaigns, seasonality, entity_label = load_campaigns_from_csv(args.csv, level=args.level)
        if seasonality:
            print(f"Detected 52-week format — seasonality model fitted for "
                  f"{len(seasonality.indices)} {entity_label.lower()}s.")
            target_week = args.week or datetime.date.today().isocalendar()[1]
            print(f"Projecting for week {target_week}.")
        else:
            print(f"Detected flat format ({entity_label} level) — no seasonality modelling.")
    else:
        print("Using built-in sample data (6 campaigns, ~$55k/month).")
        print("Tip: run with --generate-sample to create a 52-week CSV.")
        campaigns = sample_campaigns()
        entity_label = "Campaign"

    total_budget = sum(c.current_spend for c in campaigns)

    status = (f"{entity_label}s: {len(campaigns)}   Budget: ${total_budget:,.0f}   "
              f"Simulations: {args.sims:,}")
    if target_week:
        status += f"   Week: {target_week}"
    (console.print(f"\n[dim]{status}[/dim]") if HAS_RICH else print(status))

    # ── Simulate ──────────────────────────────────────────────────────────────
    optimizer = MonteCarloOptimizer(
        campaigns, total_budget,
        n_simulations=args.sims, seed=args.seed,
        seasonality=seasonality, target_week=target_week,
    )

    if HAS_RICH:
        with Progress(SpinnerColumn(), "[progress.description]{task.description}",
                      BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%",
                      TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task("[cyan]Running Monte Carlo simulations…", total=args.sims)
            records = []
            current = np.array([c.current_spend for c in optimizer.campaigns])
            for i in range(optimizer.n_simulations):
                alloc = current if i == 0 else optimizer._sample_allocation()
                convs, cpa = optimizer._evaluate(alloc)
                row: Dict = {"sim_id": i, "total_conversions": convs, "total_cpa": cpa}
                for camp, spend in zip(optimizer.campaigns, alloc):
                    row[f"spend_{camp.name}"] = spend
                    row[f"pct_{camp.name}"] = spend / optimizer.total_budget * 100
                records.append(row)
                if i % 500 == 0:
                    prog.update(task, completed=i)
            prog.update(task, completed=args.sims)
            df = pd.DataFrame(records)
    else:
        print("Running simulations…", flush=True)
        df = optimizer.run()
        print("Done.")

    # ── Analyse & report ──────────────────────────────────────────────────────
    optima = optimizer.find_optima(df)
    print_report(campaigns, df, optima, total_budget, seasonality, target_week, entity_label)

    df.to_csv("monte_carlo_results.csv", index=False)
    print("Simulation data  → monte_carlo_results.csv")

    if not args.no_plot:
        print("Generating visualisation…")
        img = create_visualizations(
            campaigns, df, optima, total_budget,
            seasonality=seasonality, target_week=target_week,
            output_file=args.out,
            entity_label=entity_label,
        )
        print(f"Chart saved      → {img}")


if __name__ == "__main__":
    main()
