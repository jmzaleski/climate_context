"""
climate_context.py
------------------
Hourly forecast curves (GFS, ECMWF, GEM, ICON) plotted against
ERA5 historical climatology for Golden BC and Vancouver BC.

Layout per location: two panels side by side
  LEFT  — near term (today + NEAR_DAYS), tight auto-scale
  RIGHT — medium range (remaining days), outlier-clipped scale
           with ⚠ annotation showing peak exceedance per model only

Also saves:  climate_context_<location>.csv  (wide format, one col per model)

Requirements: pip install requests numpy pandas matplotlib
"""

import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Configuration ─────────────────────────────────────────────────────────────

LOCATIONS = {
    "Vancouver, BC": {"lat": 49.25, "lon": -123.12},
    "Golden, BC":    {"lat": 51.30, "lon": -116.98},
}

FORECAST_MODELS = {
    "GFS (NOAA)":   "gfs_seamless",
    "ECMWF IFS":    "ecmwf_ifs025",
    "GEM (Canada)": "gem_seamless",
    "ICON (DWD)":   "icon_seamless",
}

MODEL_COLOURS = {
    "GFS (NOAA)":   "#E05C2A",
    "ECMWF IFS":    "#8B2BE0",
    "GEM (Canada)": "#2AB05C",
    "ICON (DWD)":   "#2A7BE0",
}

TIMEZONE          = "America/Vancouver"
FORECAST_DAYS     = 10
NEAR_DAYS         = 2
ERA5_START_YEAR   = 1981
ERA5_END_YEAR     = 2023
CLIMATE_WINDOW    = 10
TIME_SLOTS        = [0, 6, 12, 18]
MAX_WORKERS       = 8
OUTLIER_THRESHOLD = 5.0   # °C above ERA5 p90 → flag + clip

# ── API fetches ───────────────────────────────────────────────────────────────

def fetch_forecast_hourly(lat, lon, model_id):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m",
        "models": model_id,
        "timezone": TIMEZONE,
        "forecast_days": FORECAST_DAYS,
    }, timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.Series(
        data=h["temperature_2m"],
        index=pd.to_datetime(h["time"]),
        dtype=float,
    ).dropna()


def fetch_sunrise_sunset(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "daily": ["sunrise", "sunset"],
        "timezone": TIMEZONE,
        "forecast_days": FORECAST_DAYS,
    }, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({
        "date":    pd.to_datetime(d["time"]).date,
        "sunrise": pd.to_datetime(d["sunrise"]),
        "sunset":  pd.to_datetime(d["sunset"]),
    })


def fetch_era5_daily_min(lat, lon):
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon,
        "start_date": f"{ERA5_START_YEAR}-01-01",
        "end_date":   f"{ERA5_END_YEAR}-12-31",
        "daily": "temperature_2m_min",
        "timezone": TIMEZONE,
    }, timeout=90)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]).date,
        "tmin": d["temperature_2m_min"],
    }).dropna()
    df["doy"] = pd.to_datetime(df["date"]).dt.day_of_year
    return df


def _fetch_era5_hourly_year(lat, lon, year, start_mmdd, end_mmdd, crosses_year):
    rows = []
    try:
        segments = (
            [(f"{year}-{start_mmdd}", f"{year}-{end_mmdd}")]
            if not crosses_year else
            [(f"{year}-{start_mmdd}", f"{year}-12-31"),
             (f"{year+1}-01-01",      f"{year+1}-{end_mmdd}")]
        )
        for start, end in segments:
            if int(start[:4]) > ERA5_END_YEAR:
                continue
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
                "latitude": lat, "longitude": lon,
                "start_date": start, "end_date": end,
                "hourly": "temperature_2m",
                "timezone": TIMEZONE,
            }, timeout=30)
            r.raise_for_status()
            h = r.json()["hourly"]
            for t, v in zip(h["time"], h["temperature_2m"]):
                if v is not None:
                    rows.append((pd.Timestamp(t), float(v)))
    except Exception:
        pass
    return rows


def fetch_era5_hourly_all(lat, lon):
    today     = datetime.date.today()
    win_start = today - datetime.timedelta(days=CLIMATE_WINDOW)
    win_end   = today + datetime.timedelta(days=FORECAST_DAYS + CLIMATE_WINDOW)
    crosses   = win_start.month > win_end.month
    start_mmdd, end_mmdd = win_start.strftime("%m-%d"), win_end.strftime("%m-%d")

    all_rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(_fetch_era5_hourly_year,
                      lat, lon, yr, start_mmdd, end_mmdd, crosses): yr
            for yr in range(ERA5_START_YEAR, ERA5_END_YEAR + 1)
        }
        for fut in as_completed(futures):
            all_rows.extend(fut.result())

    if not all_rows:
        return pd.DataFrame(columns=["dt", "temp", "hour", "doy"])
    df = pd.DataFrame(all_rows, columns=["dt", "temp"])
    df["hour"] = df["dt"].dt.hour
    df["doy"]  = df["dt"].dt.day_of_year
    return df.sort_values("dt").reset_index(drop=True)


# ── Stats ─────────────────────────────────────────────────────────────────────

def pstats(vals):
    if len(vals) < 5:
        return None
    return dict(p10=np.percentile(vals, 10), p25=np.percentile(vals, 25),
                median=np.median(vals),       p75=np.percentile(vals, 75),
                p90=np.percentile(vals, 90))


def doy_mask(df, target_date):
    doy = pd.Timestamp(target_date).day_of_year
    lo, hi = doy - CLIMATE_WINDOW, doy + CLIMATE_WINDOW
    if lo < 1:
        return (df["doy"] <= hi) | (df["doy"] >= 365 + lo)
    if hi > 365:
        return (df["doy"] >= lo) | (df["doy"] <= hi - 365)
    return (df["doy"] >= lo) & (df["doy"] <= hi)


def slot_stats(era5h, date, hour):
    mask = doy_mask(era5h, date) & (era5h["hour"] == hour)
    return pstats(era5h.loc[mask, "temp"].values)


def min_stats(era5m, date):
    return pstats(era5m.loc[doy_mask(era5m, date), "tmin"].values)


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(loc_name, forecasts, era5h_df, era5_min_df, all_dates):
    """
    Wide-format CSV: datetime index, one column per forecast model,
    plus ERA5 slot median and ERA5 daily-min median for context.
    """
    # Combine all forecast series into one DataFrame
    fc_df = pd.DataFrame(forecasts)
    fc_df.index.name = "datetime"

    # ERA5 slot median at each forecast hour
    era5_slot_col = []
    for dt in fc_df.index:
        st = slot_stats(era5h_df, dt.date(), dt.hour)
        era5_slot_col.append(st["median"] if st else np.nan)
    fc_df["ERA5_slot_median"] = era5_slot_col

    # ERA5 daily min median (repeated across each day's rows)
    era5_min_col = []
    for dt in fc_df.index:
        st = min_stats(era5_min_df, dt.date())
        era5_min_col.append(st["median"] if st else np.nan)
    fc_df["ERA5_daily_min_median"] = era5_min_col

    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", loc_name).strip("_")
    fname = f"climate_context_{safe_name}.csv"
    fc_df.round(2).to_csv(fname)
    print(f"  CSV → {fname}  ({len(fc_df)} rows × {len(fc_df.columns)} cols)")


# ── Panel drawing ─────────────────────────────────────────────────────────────

def draw_panel(ax, dates, forecasts, sunrise_df, era5_min_df, era5h_df,
               title, is_near, show_legend, show_ylabel):

    x_min = pd.Timestamp(dates[0])
    x_max = pd.Timestamp(dates[-1]) + pd.Timedelta(hours=23)
    ax.set_xlim(x_min, x_max)
    ax.set_facecolor("#FFFDF5" if is_near else "#F5F8FF")

    # ── Sunrise shading ──
    for _, row in sunrise_df.iterrows():
        if row["date"] in dates:
            ax.axvspan(row["sunrise"] - pd.Timedelta(minutes=45),
                       row["sunrise"] + pd.Timedelta(minutes=45),
                       color="#FFE566", alpha=0.30, zorder=1)

    # ── ERA5 slot climatology band ──
    slot_x, slot_p25, slot_med, slot_p75 = [], [], [], []
    for d in dates:
        for h in TIME_SLOTS:
            st = slot_stats(era5h_df, d, h)
            if st:
                slot_x.append(pd.Timestamp(d) + pd.Timedelta(hours=h))
                slot_p25.append(st["p25"])
                slot_med.append(st["median"])
                slot_p75.append(st["p75"])

    if slot_x:
        ax.fill_between(slot_x, slot_p25, slot_p75,
                        color="#B8D9F5", alpha=0.50, zorder=2,
                        label=f"ERA5 p25–p75 at 0/6/12/18h ({ERA5_START_YEAR}–{ERA5_END_YEAR})")
        ax.plot(slot_x, slot_med, color="#4A90C4", lw=1.5,
                ls="--", zorder=3, label="ERA5 slot median")

    # ── ERA5 daily minimum band ──
    era5_p90_by_day = {}
    first = True
    for d in dates:
        st = min_stats(era5_min_df, d)
        if not st:
            continue
        era5_p90_by_day[d] = st["p90"]
        d0 = pd.Timestamp(d)
        d1 = d0 + pd.Timedelta(hours=23, minutes=59)
        ax.fill_between([d0, d1], [st["p25"]] * 2, [st["p75"]] * 2,
                        color="#7EC8C8", alpha=0.40, zorder=2,
                        label="ERA5 daily min p25–p75" if first else "_nolegend_")
        ax.hlines(st["median"], d0, d1, colors="#1A8C8C", lw=1.2, ls=":",
                  zorder=3,
                  label="ERA5 daily min median" if first else "_nolegend_")
        first = False

    # ── Outlier detection: ONE entry per model (peak exceedance only) ────────
    clip_top = None
    if not is_near and era5_p90_by_day:
        model_peaks = {}   # model_name → (date, peak_val, excess)
        for model_name, series in forecasts.items():
            for d in dates:
                p90 = era5_p90_by_day.get(d)
                if p90 is None:
                    continue
                day = series[series.index.date == d]
                if day.empty:
                    continue
                excess = day.max() - p90
                if excess > OUTLIER_THRESHOLD:
                    if (model_name not in model_peaks
                            or excess > model_peaks[model_name][2]):
                        model_peaks[model_name] = (d, day.max(), excess)

        if model_peaks:
            max_p90  = max(era5_p90_by_day.values())
            clip_top = max_p90 + OUTLIER_THRESHOLD + 3.0

            # Build compact annotation — one line per model
            lines = [f"⚠  Clipped (>{OUTLIER_THRESHOLD:.0f}° above ERA5 p90):"]
            for model_name, (d, peak, excess) in sorted(
                    model_peaks.items(), key=lambda x: -x[1][2]):
                day_str = pd.Timestamp(d).strftime("%a %b %-d")
                colour  = MODEL_COLOURS[model_name]
                lines.append(
                    f"  {model_name}: {peak:.1f}°C on {day_str} (+{excess:.1f}°)")

            ax.text(0.02, 0.98, "\n".join(lines),
                    transform=ax.transAxes,
                    fontsize=7.5, color="#7B0000",
                    va="top", ha="left", zorder=10,
                    bbox=dict(facecolor="#FFF0F0", alpha=0.90,
                              edgecolor="#CC4444", linewidth=0.8))

    # ── Forecast curves + daily minimum markers ──────────────────────────────
    lw   = 2.0 if is_near else 1.5
    alph = 0.90 if is_near else 0.70
    for model_name, series in forecasts.items():
        colour   = MODEL_COLOURS[model_name]
        day_mask = [d in dates for d in series.index.date]
        seg      = series[day_mask]
        if seg.empty:
            continue

        plot_vals = (np.clip(seg.values, None, clip_top + 2)
                     if clip_top is not None else seg.values)
        ax.plot(seg.index, plot_vals,
                color=colour, lw=lw, alpha=alph, zorder=5, label=model_name)

        for d in dates:
            day = series[series.index.date == d]
            if day.empty:
                continue
            min_t  = day.min()
            min_dt = day.idxmin()
            ax.plot(min_dt, min_t, marker="v", ms=6 if is_near else 5,
                    color=colour, zorder=6, label="_nolegend_")
            ax.annotate(f"{min_t:.1f}°",
                        xy=(min_dt, min_t), xytext=(0, -13),
                        textcoords="offset points",
                        ha="center", fontsize=7, color=colour, zorder=7)

    # ── Grid lines ──
    for d in dates:
        for h in range(24):
            dt = pd.Timestamp(d) + pd.Timedelta(hours=h)
            if h == 0 and d != dates[0]:
                ax.axvline(dt, color="#333", lw=0.6, alpha=0.35, zorder=1)
            elif h in (6, 12, 18):
                ax.axvline(dt, color="#888", lw=0.3, alpha=0.20, zorder=1)

    # ── Apply y-axis clip ──
    if clip_top is not None:
        ax.set_ylim(top=clip_top)

    # ── Axes ──
    ax.set_title(title, fontsize=11, fontweight="bold", pad=18)
    if show_ylabel:
        ax.set_ylabel("Temperature (°C)", fontsize=10)

    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=TIME_SLOTS))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", labelsize=7.5)

    # Top x: date labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([mdates.date2num(pd.Timestamp(d)) for d in dates])
    ax2.set_xticklabels(
        [pd.Timestamp(d).strftime("%a\n%b %-d") for d in dates], fontsize=8)
    ax2.tick_params(length=0)

    ax.grid(axis="y", alpha=0.20)

    if show_legend:
        # Build a clean, deduplicated legend
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l != "_nolegend_" and l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(),
                  loc="upper right", fontsize=7.5, framealpha=0.92)

    ax.text(0.01, 0.02,
            "▼ = daily min   │   Yellow = sunrise ±45 min",
            transform=ax.transAxes, fontsize=7, color="#666", va="bottom",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today    = datetime.date.today()
    n        = len(LOCATIONS)
    far_days = FORECAST_DAYS - NEAR_DAYS

    fig = plt.figure(figsize=(18, 5.2 * n))
    gs  = fig.add_gridspec(n, 2,
                           width_ratios=[NEAR_DAYS, far_days],
                           hspace=0.45, wspace=0.10)
    fig.suptitle(
        f"Hourly Forecast vs. ERA5 Climatology  ·  {today}\n"
        f"LEFT: detail ({NEAR_DAYS} days, auto-scale)   "
        f"RIGHT: medium range ({far_days} days, outlier-clipped)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    for row, (loc_name, coords) in enumerate(LOCATIONS.items()):
        lat, lon = coords["lat"], coords["lon"]
        print(f"\n{'─'*60}\n  {loc_name}\n{'─'*60}")

        forecasts = {}
        for model_name, model_id in FORECAST_MODELS.items():
            print(f"  {model_name} …", end=" ", flush=True)
            try:
                forecasts[model_name] = fetch_forecast_hourly(lat, lon, model_id)
                print("✓")
            except Exception as e:
                print(f"✗  {e}")

        if not forecasts:
            continue

        print("  Sunrise/sunset …", end=" ", flush=True)
        try:
            sunrise_df = fetch_sunrise_sunset(lat, lon)
            print("✓")
        except Exception as e:
            print(f"✗  {e}")
            sunrise_df = pd.DataFrame(columns=["date", "sunrise", "sunset"])

        print(f"  ERA5 daily min …", end=" ", flush=True)
        era5_min = fetch_era5_daily_min(lat, lon)
        print(f"✓  ({len(era5_min):,} rows)")

        print(f"  ERA5 hourly [{MAX_WORKERS} parallel] …", end=" ", flush=True)
        era5h = fetch_era5_hourly_all(lat, lon)
        print(f"✓  ({len(era5h):,} rows)")

        all_dates  = sorted({dt.date() for s in forecasts.values() for dt in s.index})
        near_dates = all_dates[:NEAR_DAYS]
        far_dates  = all_dates[NEAR_DAYS:]

        # ── CSV ──
        save_csv(loc_name, forecasts, era5h, era5_min, all_dates)

        # ── Panels ──
        ax_near = fig.add_subplot(gs[row, 0])
        ax_far  = fig.add_subplot(gs[row, 1])

        draw_panel(ax_near, near_dates, forecasts, sunrise_df, era5_min, era5h,
                   title=f"{loc_name}  —  Detail",
                   is_near=True, show_legend=False, show_ylabel=True)

        draw_panel(ax_far, far_dates, forecasts, sunrise_df, era5_min, era5h,
                   title="Medium range",
                   is_near=False, show_legend=True, show_ylabel=False)

    plt.tight_layout()
    out = "climate_context.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n✓  Chart → {out}")
    plt.show()


if __name__ == "__main__":
    main()
