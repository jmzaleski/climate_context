"""
climate_context.py
------------------
Plots weather model forecasts against ERA5 historical climatology
for Golden BC and Vancouver BC.

Purpose: show where the forecast sits relative to the historical
         distribution for the same time of year — calm, factual context.

Data sources (both free, no API key needed):
  Forecasts  → Open-Meteo Forecast API  (GFS, ECMWF, GEM, ICON)
  Climatology→ Open-Meteo Archive API   (ERA5, 1981–2023)

Requirements:
    pip install requests numpy pandas matplotlib
    -- or --
    uv run --with requests,numpy,pandas,matplotlib climate_context.py
"""

import datetime
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

# All fetched from Open-Meteo generic forecast endpoint via the 'models' param.
# These are the best available for western Canada.
FORECAST_MODELS = {
    "GFS (NOAA)":   "gfs_seamless",
    "ECMWF IFS":    "ecmwf_ifs025",
    "GEM (Canada)": "gem_seamless",
    "ICON (DWD)":   "icon_seamless",
}

MODEL_COLOURS = {
    "GFS (NOAA)":   "#E05C2A",  # orange
    "ECMWF IFS":    "#8B2BE0",  # purple
    "GEM (Canada)": "#2AB05C",  # green
    "ICON (DWD)":   "#2A7BE0",  # blue
}

TIMEZONE        = "America/Vancouver"
FORECAST_DAYS   = 10
ERA5_START_YEAR = 1981
ERA5_END_YEAR   = 2023   # last complete year in ERA5 archive
CLIMATE_WINDOW  = 10     # ± days around each date to pool for percentiles

# ── API helpers ───────────────────────────────────────────────────────────────

def fetch_forecast(lat, lon, model_id):
    """
    Return a Series of daily max temperature (°C) indexed by date,
    from the Open-Meteo forecast API for a single model.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         "temperature_2m_max",
        "models":        model_id,
        "timezone":      TIMEZONE,
        "forecast_days": FORECAST_DAYS,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    daily = data["daily"]
    s = pd.Series(
        data=daily["temperature_2m_max"],
        index=pd.to_datetime(daily["time"]).date,
        name=model_id,
        dtype=float,
    )
    return s.dropna()


def fetch_era5(lat, lon):
    """
    Return a DataFrame with columns [date, tmax, doy] covering
    ERA5_START_YEAR through ERA5_END_YEAR from the Open-Meteo archive.
    This is one request for the full period (~16 000 daily rows).
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": f"{ERA5_START_YEAR}-01-01",
        "end_date":   f"{ERA5_END_YEAR}-12-31",
        "daily":      "temperature_2m_max",
        "timezone":   TIMEZONE,
    }
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]).date,
        "tmax": data["daily"]["temperature_2m_max"],
    }).dropna(subset=["tmax"])
    df["doy"] = pd.to_datetime(df["date"]).dt.day_of_year
    return df


def climate_stats(era5_df, target_date):
    """
    Pool ERA5 values within ±CLIMATE_WINDOW days-of-year around
    target_date and return percentile statistics.
    """
    doy  = pd.Timestamp(target_date).day_of_year
    low  = doy - CLIMATE_WINDOW
    high = doy + CLIMATE_WINDOW

    if low < 1:
        mask = (era5_df["doy"] <= high) | (era5_df["doy"] >= 365 + low)
    elif high > 365:
        mask = (era5_df["doy"] >= low) | (era5_df["doy"] <= high - 365)
    else:
        mask = (era5_df["doy"] >= low) & (era5_df["doy"] <= high)

    vals = era5_df.loc[mask, "tmax"].values
    if len(vals) < 10:
        return None
    return {
        "p10":    np.percentile(vals, 10),
        "p25":    np.percentile(vals, 25),
        "median": np.median(vals),
        "mean":   np.mean(vals),
        "p75":    np.percentile(vals, 75),
        "p90":    np.percentile(vals, 90),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_location(ax, loc_name, coords, forecasts, era5_df):
    today = datetime.date.today()

    # Collect all dates that appear in any model forecast
    all_dates = sorted(set(d for s in forecasts.values() for d in s.index))
    x = [pd.Timestamp(d) for d in all_dates]

    # Historical stats for each date
    p10, p25, med, mean, p75, p90 = [], [], [], [], [], []
    for d in all_dates:
        st = climate_stats(era5_df, d)
        if st:
            p10.append(st["p10"]);  p25.append(st["p25"])
            med.append(st["median"]); mean.append(st["mean"])
            p75.append(st["p75"]);  p90.append(st["p90"])
        else:
            for lst in (p10, p25, med, mean, p75, p90):
                lst.append(np.nan)

    # ── Shaded climatology bands ──
    ax.fill_between(x, p10, p90,
                    color="#B8D9F5", alpha=0.55,
                    label=f"ERA5 10th–90th %ile  ({ERA5_START_YEAR}–{ERA5_END_YEAR})")
    ax.fill_between(x, p25, p75,
                    color="#60A8E0", alpha=0.65,
                    label="ERA5 25th–75th %ile (IQR)")
    ax.plot(x, med,  color="#1A5C99", lw=2,   ls="--", label="ERA5 median")
    ax.plot(x, mean, color="#555555", lw=1.5, ls=":",  label="ERA5 mean")

    # ── Model forecast lines ──
    for model_name, series in forecasts.items():
        fc_x = [pd.Timestamp(d) for d in series.index]
        fc_y = list(series.values)
        colour = MODEL_COLOURS[model_name]
        ax.plot(fc_x, fc_y,
                color=colour, lw=2.5, marker="o", ms=6,
                label=model_name, zorder=5)
        for xi, yi in zip(fc_x, fc_y):
            ax.annotate(f"{yi:.1f}°",
                        xy=(xi, yi), xytext=(0, 9),
                        textcoords="offset points",
                        ha="center", fontsize=7.5,
                        color=colour, zorder=6)

    # ── Today marker ──
    ax.axvline(pd.Timestamp(today), color="#333", lw=1, alpha=0.35)

    # ── Formatting ──
    ax.set_title(loc_name, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Daily Max Temperature (°C)", fontsize=10)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%b %-d"))
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", alpha=0.12)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    note = (
        f"Bands show ERA5 daily max temps within ±{CLIMATE_WINDOW} days "
        f"of each calendar date  ({ERA5_START_YEAR}–{ERA5_END_YEAR}, "
        f"{ERA5_END_YEAR - ERA5_START_YEAR + 1} years)"
    )
    ax.text(0.01, 0.02, note, transform=ax.transAxes,
            fontsize=7.5, color="#555", va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today = datetime.date.today()
    n     = len(LOCATIONS)
    fig, axes = plt.subplots(n, 1, figsize=(13, 6.5 * n), squeeze=False)

    fig.suptitle(
        f"Forecast vs. ERA5 Historical Climatology  ·  Generated {today}",
        fontsize=13, fontweight="bold", y=1.005,
    )

    for ax, (loc_name, coords) in zip(axes[:, 0], LOCATIONS.items()):
        lat, lon = coords["lat"], coords["lon"]
        print(f"\n{'─'*60}\n  {loc_name}\n{'─'*60}")

        # Forecasts
        forecasts = {}
        for model_name, model_id in FORECAST_MODELS.items():
            print(f"  Fetching {model_name} …", end=" ", flush=True)
            try:
                s = fetch_forecast(lat, lon, model_id)
                forecasts[model_name] = s
                print(f"✓  ({len(s)} days, "
                      f"{s.iloc[0]:.1f}°C today → {s.iloc[-1]:.1f}°C day {len(s)})")
            except Exception as e:
                print(f"✗  {e}")

        if not forecasts:
            print("  No forecast data — skipping.")
            ax.set_visible(False)
            continue

        # ERA5 climatology
        print(f"  Fetching ERA5 archive ({ERA5_START_YEAR}–{ERA5_END_YEAR}) …",
              end=" ", flush=True)
        era5_df = fetch_era5(lat, lon)
        print(f"✓  ({len(era5_df):,} daily rows)")

        plot_location(ax, loc_name, coords, forecasts, era5_df)

    plt.tight_layout()
    out = "climate_context.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n✓  Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
