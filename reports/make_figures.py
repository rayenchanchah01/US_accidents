"""Charts and numbers for the PDF report.

Run from the repo root, after reports/model_check.py:   python reports/make_figures.py
Writes reports/figures/*.png and reports/figures/facts.json (every number quoted in the report).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cleaning import run_all, step1_types  # noqa: E402
from cleaning.common import CleaningLog, RAW_PATH  # noqa: E402
from cleaning.step2_duplicates import EVENT_KEY  # noqa: E402
from cleaning.step3_columns import DROP_CATEGORY  # noqa: E402
from cleaning.step5_outliers import LIMITS  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
MODEL_CHECK = ROOT / "reports" / "model_check.csv"

# Colours: validated colour-blind-safe categorical slots + one-hue ordinal ramp for Severity 1-4
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEVERITY = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SOURCE_COLORS = {"Source1": BLUE, "Source2": ORANGE, "Source3": AQUA}
INK, INK2, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
WIDTH = 6.3  # inches: the text width of the A4 report, so fonts print at their real size

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 8.5, "text.color": INK,
    "axes.titlesize": 9.5, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")
YEARS = list(range(2016, 2024))


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def value_grid(ax, axis):
    """Gridlines only along the value axis."""
    ax.grid(False)
    ax.grid(True, axis=axis)


# ---------------------------------------------------------------- dataset

def fig_severity(raw):
    counts = raw["Severity"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    labels = [f"Severity {s}" for s in counts.index]
    ax.bar(labels, counts.values, width=0.45, color=BLUE)
    for x, v in zip(labels, counts.values):
        ax.text(x, v + 8000, f"{v:,}  ({v / len(raw):.1%})", ha="center", va="bottom", color=INK2)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.yaxis.set_major_formatter(thousands)
    ax.set_ylabel("Accidents")
    ax.set_title("Accidents per Severity level (original 500K rows)")
    value_grid(ax, "y")
    save(fig, "01_severity")


# ---------------------------------------------------------------- step 3: leakage

def fig_leakage_evidence(typed):
    duration = (typed["End_Time"] - typed["Start_Time"]).dt.total_seconds() / 60
    panels = [(duration.groupby(typed["Severity"]).median(), "Median duration (End_Time - Start_Time)", "minutes", "{:.0f} min"),
              (typed.groupby("Severity")["Distance(mi)"].median(), "Median Distance(mi)", "miles", "{:.2f} mi")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.1))
    for ax, (s, title, unit, fmt) in zip(axes, panels):
        labels = [str(i) for i in s.index]
        ax.bar(labels, s.values, width=0.45, color=[BLUE] * 3 + [SEVERITY[3]])
        for x, v in zip(labels, s.values):
            ax.text(x, v + s.max() * 0.03, fmt.format(v), ha="center", va="bottom", color=INK2)
        ax.set_ylim(0, s.max() * 1.22)
        ax.set_title(title)
        ax.set_xlabel("Severity")
        ax.set_ylabel(unit)
        value_grid(ax, "y")
    fig.tight_layout(w_pad=3)
    save(fig, "03_leakage_evidence")


def fig_leakage_model(summary):
    rows = [("Cleaned set (baseline)", summary["baseline_mean"], BLUE)]
    names = {"End_Time": "+ End_Time (as duration)", "Distance(mi)": "+ Distance(mi)",
             "Description": "+ Description mentions 'closed'/'blocked'",
             "All three leakage columns": "+ all three"}
    rows += [(names[k], summary["leakage"][k]["mean_f1"], ORANGE) for k in names]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], height=0.5, color=[r[2] for r in rows])
    for yi, (_, v, _) in zip(y, rows):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 on the hold-out set (mean of 3 splits)")
    ax.set_title("Adding the leakage columns inflates the score")
    value_grid(ax, "x")
    save(fig, "04_leakage_model")


# ---------------------------------------------------------------- step 3: redundant / artefacts / rare

def fig_redundant(typed):
    both = typed[["Temperature(F)", "Wind_Chill(F)"]].dropna()
    both = both[both["Temperature(F)"].between(-40, 120)]
    r = typed["Temperature(F)"].corr(typed["Wind_Chill(F)"])
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#3987e5", "#0d366b"])
    ax.hexbin(both["Temperature(F)"], both["Wind_Chill(F)"], gridsize=45, bins="log", cmap=cmap, mincnt=1, linewidths=0)
    ax.plot([-40, 120], [-40, 120], color=INK2, linewidth=0.8)
    ax.set_xlabel("Temperature (°F)")
    ax.set_ylabel("Wind chill (°F)")
    ax.set_title(f"Wind_Chill vs Temperature (r = {r:.3f})")
    ax.text(100, 70, "y = x", color=INK2, fontsize=7.5)
    ax.grid(False)

    ax = axes[1]
    cols = ["Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"]
    agree = [(typed[c] == typed["Sunrise_Sunset"]).mean() * 100 for c in cols]
    y = np.arange(3)[::-1]
    ax.barh(y, agree, height=0.45, color=BLUE)
    for yi, v in zip(y, agree):
        ax.text(v + 1.5, yi, f"{v:.1f}%", va="center", color=INK2)
    ax.set_yticks(y, [c.replace("_", " ") for c in cols])
    ax.set_xlim(0, 112)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of rows with the same value as Sunrise_Sunset")
    ax.set_title("Twilight columns repeat Sunrise_Sunset")
    value_grid(ax, "x")
    fig.tight_layout(w_pad=2.5)
    save(fig, "05_redundant")
    return round(r, 4), dict(zip(cols, [round(a, 1) for a in agree]))


def _year_lines(ax, series, title, ylabel):
    for (name, (s, color)), (x_label, align) in zip(series.items(), [(2016, "left"), (2023, "right")]):
        s = s.reindex(YEARS)
        ax.plot(s.index, s.values, color=color, linewidth=1.6, marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=1, solid_capstyle="round")
        ax.annotate(name, (x_label, s[x_label]), xytext=(0, 7), textcoords="offset points",
                    ha=align, color=INK2, fontsize=7.5)
    ax.set_xticks([2016, 2018, 2020, 2022])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    value_grid(ax, "y")


def fig_artefacts(typed):
    year = typed["Start_Time"].dt.year
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3))
    ax = axes[0]
    miss = typed["Precipitation(in)"].isna().groupby(year).mean() * 100
    ax.bar(miss.index, miss.values, width=0.55, color=BLUE)
    ax.set_xticks([2016, 2018, 2020, 2022])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_title("Precipitation: % missing")
    value_grid(ax, "y")

    share = lambda col, value: (typed[col] == value).groupby(year).mean() * 100  # noqa: E731
    _year_lines(axes[1], {'"Clear"': (share("Weather_Condition", "Clear"), BLUE),
                          '"Fair"': (share("Weather_Condition", "Fair"), ORANGE)},
                "Weather_Condition label", "% of rows")
    _year_lines(axes[2], {'"Calm"': (share("Wind_Direction", "Calm"), BLUE),
                          '"CALM"': (share("Wind_Direction", "CALM"), ORANGE)},
                "Wind_Direction spelling", "")
    for ax in axes[1:]:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.legend(handles=[plt.Line2D([], [], color=BLUE, linewidth=1.6, marker="o", markersize=4),
                        plt.Line2D([], [], color=ORANGE, linewidth=1.6, marker="o", markersize=4)],
               labels=["Old wording (until 2019)", "New wording (from 2019)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.62, -0.06))
    fig.tight_layout(w_pad=1.5, rect=(0, 0.06, 1, 1))
    save(fig, "06_artefacts")
    return miss.round(1).to_dict()


def fig_rare_flags(typed):
    flags = step1_types.ROAD_COLS
    pct = (typed[flags].mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    y = np.arange(len(pct))
    plotted = pct.clip(lower=0.001)
    colors = [ORANGE if c in DROP_CATEGORY else BLUE for c in pct.index]
    ax.barh(y, plotted, height=0.55, color=colors)
    ax.set_xscale("log")
    ax.set_xlim(0.001, 60)
    for yi, (c, v) in zip(y, pct.items()):
        n = int(typed[c].sum())
        text = "never True (0 rows)" if n == 0 else (f"{n} rows" if n < 1000 else f"{v:.2f}%")
        ax.text(max(v, 0.001) * 1.25, yi, text, va="center", color=INK2, fontsize=7.5, zorder=3,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    ax.axvline(1, color=INK2, linewidth=0.8, zorder=1)
    ax.text(1.08, len(pct) - 0.2, "1% of rows", color=INK2, fontsize=7.5)
    ax.set_ylim(-0.6, len(pct) + 0.4)
    ax.set_yticks(y, pct.index)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlabel("% of rows where the flag is True (log scale)")
    ax.set_title("Road-feature flags: how often each one is True")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["Kept", "Dropped"], loc="lower right")
    value_grid(ax, "x")
    save(fig, "07_rare_flags")


# ---------------------------------------------------------------- step 3: model check

def model_summary():
    """Every test run uses seed 0, so it is compared with the seed-0 baseline on the same split.
    The noise band is the largest change that retraining the baseline with another seed caused."""
    mc = pd.read_csv(MODEL_CHECK)
    noise = mc[mc["kind"] == "noise"]
    ref = noise[noise["seed"] == 0].set_index("split")["macro_f1"]
    seed_effect = noise["macro_f1"] - noise["split"].map(ref)
    out = {"baseline_mean": float(ref.mean()), "baseline_by_split": ref.round(4).to_dict(),
           "noise_band": float(seed_effect.abs().max()), "n_seeds": int(noise["seed"].nunique()),
           "n_splits": int(noise["split"].nunique()),
           "baseline_recall_sev4": float(noise.loc[noise["seed"] == 0, "recall_sev4"].mean()),
           "dropped": {}, "leakage": {}}
    for kind in ("dropped", "leakage"):
        part = mc[mc["kind"] == kind].copy()
        part["delta"] = part["macro_f1"] - part["split"].map(ref)
        for name, g in part.groupby("column", sort=False):
            out[kind][name] = {"mean_f1": float(g["macro_f1"].mean()), "mean_delta": float(g["delta"].mean()),
                               "min_delta": float(g["delta"].min()), "max_delta": float(g["delta"].max()),
                               "recall_sev4": float(g["recall_sev4"].mean())}
    return out


def fig_model_check(summary):
    d = pd.DataFrame(summary["dropped"]).T
    together = d.loc[["All dropped columns together"]]
    single = d.drop(index="All dropped columns together").sort_values("mean_delta")
    d = pd.concat([single, together])
    band = summary["noise_band"]
    fig, ax = plt.subplots(figsize=(WIDTH, 4.6))
    y = np.arange(len(d))
    lim = max(0.012, float(np.abs(d[["min_delta", "max_delta"]]).max().max()) * 1.15)
    ax.axvspan(-band, band, color=BAND, zorder=0, linewidth=0)
    ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
    for yi, (name, r) in zip(y, d.iterrows()):
        color = SEVERITY[3] if name == "All dropped columns together" else BLUE
        ax.plot([r["min_delta"], r["max_delta"]], [yi, yi], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot(r["mean_delta"], yi, "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    ax.axhline(len(d) - 1.5, color=GRID, linewidth=0.8)
    labels = [(f"All {len(single)} together" if n == "All dropped columns together" else n) for n in d.index]
    ax.set_yticks(y, labels)
    for tick, n in zip(ax.get_yticklabels(), d.index):
        if n == "All dropped columns together":
            tick.set_fontweight("semibold")
            tick.set_color(INK)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.3f}" if v else "0"))
    ax.set_xlabel(f"Change in macro F1 when the column is put back (dot = mean of {summary['n_splits']} splits, line = min to max)")
    ax.set_title("Putting each dropped column back changes nothing beyond retraining noise")
    ax.text(band, -0.55, f"  grey band = retraining noise (±{band:.3f})", color=INK2, fontsize=7.5, va="center")
    value_grid(ax, "x")
    save(fig, "08_model_check")


# ---------------------------------------------------------------- steps 4-6

def fig_missing(raw):
    miss = (raw.isna().mean() * 100).loc[lambda s: s > 0].sort_values()

    def action(c):
        if c in DROP_CATEGORY:
            return "Column dropped in step 3", ORANGE
        if c in ("City", "Sunrise_Sunset", "Weather_Condition"):
            return "Kept: rows removed or filled", AQUA
        return "Kept: left empty (imputed later, on train only)", BLUE

    fig, ax = plt.subplots(figsize=(WIDTH, 4.3))
    y = np.arange(len(miss))
    ax.barh(y, miss.values, height=0.6, color=[action(c)[1] for c in miss.index])
    for yi, (c, v) in zip(y, miss.items()):
        n = int(raw[c].isna().sum())
        text = f"{n} rows" if v < 0.05 else (f"{v:.2f}%" if v < 1 else f"{v:.1f}%")
        ax.text(v + 0.6, yi, text, va="center", color=INK2, fontsize=7.5)
    ax.set_yticks(y, miss.index)
    ax.set_xlim(0, 52)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("% of rows missing (original data)")
    ax.set_title("Missing values per column and what we did")
    seen = {}
    for c in miss.index:
        seen.setdefault(*action(c))
    order = ["Column dropped in step 3", "Kept: left empty (imputed later, on train only)", "Kept: rows removed or filled"]
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=seen[k]) for k in order], labels=order, loc="lower right")
    value_grid(ax, "x")
    save(fig, "09_missing")
    return miss.round(2).to_dict()


def fig_light_missing(raw):
    groups = {f"Rows missing Sunrise_Sunset ({raw['Sunrise_Sunset'].isna().sum():,})": raw[raw["Sunrise_Sunset"].isna()],
              f"All rows ({len(raw):,})": raw}
    fig, ax = plt.subplots(figsize=(WIDTH, 1.5))
    shares = {}
    for yi, (name, g) in zip([1, 0], groups.items()):
        s = g["Severity"].value_counts(normalize=True).reindex([1, 2, 3, 4], fill_value=0) * 100
        shares[name] = s.round(1).to_dict()
        left = 0
        for sev, v in s.items():
            ax.barh(yi, v, left=left, height=0.5, color=SEVERITY[sev - 1], edgecolor="white", linewidth=1.5)
            left += v
        ax.text(101, yi, f"Severity 4: {s[4]:.1f}%", va="center", color=INK, fontweight="semibold")
    ax.set_yticks([1, 0], list(groups))
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_title("Severity mix: rows missing the light columns vs all rows")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in SEVERITY],
              labels=[f"Severity {i}" for i in range(1, 5)], ncol=4, loc="upper left", bbox_to_anchor=(0, -0.28))
    value_grid(ax, "x")
    save(fig, "10_light_missing")
    return shares


def fig_outliers(typed, fixed):
    cols = ["Temperature(F)", "Wind_Speed(mph)", "Visibility(mi)", "Pressure(in)"]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.9))
    for ax, col in zip(axes.flat, cols):
        v = typed[col].dropna()
        lo, hi = LIMITS[col]
        ax.hist(v, bins=80, color=BLUE, log=True)
        x0, x1 = min(v.min(), lo), max(v.max(), hi)
        pad = (x1 - x0) * 0.03
        ax.set_xlim(x0 - pad, x1 + pad)
        for a, b in [(x0 - pad, lo), (hi, x1 + pad)]:
            if b > a:
                ax.axvspan(a, b, color=ORANGE, alpha=0.12, linewidth=0)
        for edge in (lo, hi):
            ax.axvline(edge, color=ORANGE, linewidth=1)
        ax.set_title(f"{col}: valid {lo:g} to {hi:g}")
        ax.text(0.98, 0.92, f"{fixed[col]} values set to NaN", transform=ax.transAxes, ha="right", color=INK, fontsize=7.5)
        ax.set_ylabel("rows (log)")
        value_grid(ax, "y")
    fig.tight_layout(h_pad=1.6, w_pad=2)
    save(fig, "11_outliers")


def fig_weather(clean):
    counts = clean["Weather_Group"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    y = np.arange(len(counts))
    ax.barh(y, counts.values, height=0.55, color=BLUE)
    for yi, v in zip(y, counts.values):
        ax.text(v + 3000, yi, f"{v:,}  ({v / len(clean):.1%})", va="center", color=INK2, fontsize=7.5)
    ax.set_yticks(y, counts.index)
    ax.set_xlim(0, counts.max() * 1.3)
    ax.xaxis.set_major_formatter(thousands)
    ax.set_xlabel("Accidents (cleaned data)")
    ax.set_title("Weather_Group: 108 labels folded into 8 groups")
    value_grid(ax, "x")
    save(fig, "12_weather_groups")


def fig_drift(clean):
    year = clean["Start_Time"].dt.year
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), sharey=False)
    for ax, sev in zip(axes, [3, 1]):
        for source, color in SOURCE_COLORS.items():
            g = clean[clean["Source"] == source]
            s = (g["Severity"] == sev).groupby(year[g.index]).mean().mul(100).reindex(YEARS)
            ax.plot(s.index, s.values, color=color, linewidth=1.6, marker="o", markersize=4,
                    markeredgecolor="white", markeredgewidth=1, label=source)
            last = s.dropna()
            ax.text(last.index[-1] + 0.25, last.iloc[-1], source, color=INK2, fontsize=7.5, va="center")
        ax.set_xlim(2015.6, 2024.3)
        ax.set_xticks([2016, 2018, 2020, 2022])
        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_title(f"% of accidents labelled Severity {sev}")
        value_grid(ax, "y")
    axes[0].legend(loc="upper right", handlelength=1.2)
    fig.tight_layout(w_pad=2.5)
    save(fig, "13_drift")


# ---------------------------------------------------------------- facts for the text

def collect_facts(raw, typed, clean, log, summary, extra):
    sev_raw = raw["Severity"].value_counts().sort_index()
    sev_clean = clean["Severity"].value_counts().sort_index()
    example = typed[typed["ID"].isin(["A-3422801", "A-3422802"])]
    assert len(example) == 2 and example[EVENT_KEY].nunique().max() == 1
    return {
        "raw_shape": list(raw.shape), "clean_shape": list(clean.shape),
        "date_min": str(typed["Start_Time"].min()), "date_max": str(typed["Start_Time"].max()),
        "start_time_formats": {str(k): int(v) for k, v in raw["Start_Time"].str.len().value_counts().items()},
        "n_states": int(raw["State"].nunique()), "source_counts": raw["Source"].value_counts().to_dict(),
        "severity_raw": {int(k): int(v) for k, v in sev_raw.items()},
        "severity_clean": {int(k): int(v) for k, v in sev_clean.items()},
        "steps": log.to_frame().to_dict(orient="records"),
        "repeated_event_rows": log.notes["repeated_event_rows"],
        "repeated_event_conflicting_severity": log.notes["repeated_event_conflicting_severity"],
        "impossible_values": log.notes["impossible_values"],
        "weather_mapping": log.notes["weather_mapping"],
        "weather_label_counts": raw["Weather_Condition"].value_counts().to_dict(),
        "weather_group_counts": clean["Weather_Group"].value_counts().to_dict(),
        "raw_missing_pct": (raw.isna().mean() * 100).round(2).to_dict(),
        "raw_nunique": raw.nunique().to_dict(),
        "raw_dtypes": raw.dtypes.astype(str).to_dict(),
        "clean_missing_pct": (clean.isna().mean() * 100).round(2).to_dict(),
        "clean_dtypes": clean.dtypes.astype(str).to_dict(),
        "duplicate_example": example[["ID", "Source", "Severity", "Start_Time", "End_Time", "Description"]]
                             .astype(str).to_dict(orient="records"),
        "model": summary,
        **extra,
    }


def main():
    raw = pd.read_csv(RAW_PATH)
    typed = step1_types.run(raw, CleaningLog())
    clean, log = run_all(save=False)
    summary = model_summary()

    fig_severity(raw)
    fig_leakage_evidence(typed)
    fig_leakage_model(summary)
    corr, twilight = fig_redundant(typed)
    precip_missing = fig_artefacts(typed)
    fig_rare_flags(typed)
    fig_model_check(summary)
    missing = fig_missing(raw)
    light = fig_light_missing(raw)
    fig_outliers(typed, log.notes["impossible_values"])
    fig_weather(clean)
    fig_drift(clean)

    facts = collect_facts(raw, typed, clean, log, summary,
                          {"wind_chill_corr": corr, "twilight_agreement": twilight,
                           "precip_missing_by_year": precip_missing, "light_missing_severity": light,
                           "missing_plotted": missing})
    (FIG_DIR / "facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print("figures and facts written to", FIG_DIR)


if __name__ == "__main__":
    main()
