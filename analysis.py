"""London cycle-hire demand: the full analysis, reproducibly.

Runs every step behind the findings in the README and the dashboard:
load → validate → engineer features → quantify the drivers → forecast.

    python analysis.py            # prints the findings, writes results.json

Data: 17,414 hourly observations of Transport for London cycle-hire counts
(Jan 2015 – Jan 2017) joined to weather. One row = one hour.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA = Path(__file__).parent / "data" / "london_bikes.csv"
OUT = Path(__file__).parent / "results.json"

# weather_code: 1 clear, 2 scattered cloud, 3 broken cloud, 4 cloudy,
# 7 rain, 10 rain+thunder, 26 snowfall
RAIN_CODES = [7, 10, 26]
WEATHER_LABELS = {1: "Clear", 2: "Scattered cloud", 3: "Broken cloud",
                  4: "Cloudy", 7: "Rain", 10: "Rain + thunder", 26: "Snow"}
SEASONS = {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}

COMMUTE_HOURS = [7, 8, 9, 17, 18]
LEISURE_HOURS = list(range(11, 17))
FEATURES = ["hour", "dow", "month", "doy", "t1", "t2", "hum", "wind_speed",
            "weather_code", "is_holiday", "is_weekend", "season", "is_rain"]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp")
    df["hour"] = df.timestamp.dt.hour
    df["dow"] = df.timestamp.dt.dayofweek
    df["month"] = df.timestamp.dt.month
    df["doy"] = df.timestamp.dt.dayofyear
    df["is_rain"] = df.weather_code.isin(RAIN_CODES).astype(int)
    # Demand relative to the typical level for that hour and day type. This is
    # the key move: it lets weather be compared without the daily commute
    # rhythm drowning the signal.
    df["rel"] = df.cnt / df.groupby(["hour", "is_weekend"]).cnt.transform("mean")
    return df


def validate(df: pd.DataFrame) -> dict:
    """Check the things that would quietly invalidate the analysis."""
    expected = pd.date_range(df.timestamp.min(), df.timestamp.max(), freq="h")
    missing = expected.difference(df.timestamp)
    report = {
        "rows": int(len(df)),
        "expected_hours": int(len(expected)),
        "missing_hours": int(len(missing)),
        "missing_pct": round(100 * len(missing) / len(expected), 2),
        "duplicate_timestamps": int(df.timestamp.duplicated().sum()),
        "null_values": int(df.isna().sum().sum()),
        "start": str(df.timestamp.min().date()),
        "end": str(df.timestamp.max().date()),
    }
    assert report["duplicate_timestamps"] == 0, "duplicate timestamps present"
    return report


def _hourly_profile(seg: pd.DataFrame) -> dict:
    """Mean hires per hour, split by day type, plus the sample size behind it."""
    out = {}
    for key, mask in (("weekday", seg.is_weekend == 0), ("weekend", seg.is_weekend == 1)):
        g = seg[mask].groupby("hour").cnt
        out[key] = [round(v) for v in g.mean().reindex(range(24)).fillna(0)]
        out[key + "_n"] = [int(v) for v in g.size().reindex(range(24)).fillna(0)]
    return out


def demand_shape(df: pd.DataFrame) -> dict:
    """The daily rhythm — and how sharply it differs by day type."""
    wk = df[df.is_weekend == 0].groupby("hour").cnt.mean()
    we = df[df.is_weekend == 1].groupby("hour").cnt.mean()
    return {
        "weekday_by_hour": [round(v) for v in wk.reindex(range(24)).fillna(0)],
        "weekend_by_hour": [round(v) for v in we.reindex(range(24)).fillna(0)],
        "weekday_peak_hour": int(wk.idxmax()),
        "weekday_peak": round(wk.max()),
        "weekend_peak_hour": int(we.idxmax()),
        "weekend_peak": round(we.max()),
        # The commute signature: the 8am gap between a workday and a weekend.
        "commute_ratio_8am": round(wk[8] / we[8], 1),
        # Same profile recomputed on dry and wet hours only, so the dashboard can
        # show *where in the day* rain takes its bite rather than only stating it.
        "profiles": {
            "all": _hourly_profile(df),
            "clear": _hourly_profile(df[df.is_rain == 0]),
            "rain": _hourly_profile(df[df.is_rain == 1]),
        },
    }


def weather_effect(df: pd.DataFrame) -> dict:
    """How much demand rain removes — and from whom.

    Compared on `rel` (demand relative to the same hour and day type), so the
    effect is not contaminated by rain being more common at some hours.
    """
    def drop(seg: pd.DataFrame) -> dict:
        e = seg.groupby("is_rain").rel.mean()
        if 1 not in e or 0 not in e:
            return {"drop_pct": None, "rain_hours": int(seg.is_rain.sum())}
        return {"clear": round(e[0], 3), "rain": round(e[1], 3),
                "drop_pct": round((1 - e[1] / e[0]) * 100),
                "rain_hours": int(seg.is_rain.sum())}

    commute = df[(df.is_weekend == 0) & (df.hour.isin(COMMUTE_HOURS))]
    leisure = df[(df.is_weekend == 1) & (df.hour.isin(LEISURE_HOURS))]

    by_code = (df.groupby("weather_code")
                 .agg(rel=("rel", "mean"), hours=("rel", "size"))
                 .reset_index())
    by_code["label"] = by_code.weather_code.map(WEATHER_LABELS)

    # Temperature response, clear weather only so rain doesn't confound it.
    clear = df[df.is_rain == 0].copy()
    bins = [-10, 0, 5, 10, 15, 20, 25, 40]
    labels = ["<0", "0-5", "5-10", "10-15", "15-20", "20-25", "25+"]
    clear["tbin"] = pd.cut(clear.t2, bins, labels=labels)
    temp = (clear.groupby("tbin", observed=True)
                 .agg(rel=("rel", "mean"), hours=("rel", "size")).reset_index())

    return {
        "overall": drop(df),
        "commute": drop(commute),
        "leisure": drop(leisure),
        "by_weather": [{"label": r.label, "rel": round(r.rel, 3), "hours": int(r.hours)}
                       for r in by_code.itertuples() if pd.notna(r.label)],
        "by_temperature": [{"bin": str(r.tbin), "rel": round(r.rel, 3), "hours": int(r.hours)}
                           for r in temp.itertuples()],
    }


def seasonality(df: pd.DataFrame) -> dict:
    """Monthly means, with partial months flagged rather than silently plotted.

    The series starts on 4 January 2015 and stops on 3 January 2017, so the last
    month holds three days. Charting it as a month would invent a cliff.
    """
    g = df.set_index("timestamp").resample("MS").cnt
    m, n = g.mean(), g.size()
    s = df.groupby("season").cnt.mean()
    return {
        "monthly": [{"month": str(i.date())[:7], "mean_hourly": round(v),
                     "hours": int(n[i]), "complete": bool(n[i] >= 336)}
                    for i, v in m.items()],
        "by_season": [{"season": SEASONS.get(int(k), str(k)), "mean_hourly": round(v)}
                      for k, v in s.items()],
    }


def forecast(df: pd.DataFrame, cutoff: str = "2016-10-01") -> dict:
    """Predict hourly demand, judged against baselines on unseen future data.

    The split is by time, never random: a random split would leak future hours
    into training and flatter the model. Both baselines are learned on the
    training period only.
    """
    cut = pd.Timestamp(cutoff)
    tr, te = df[df.timestamp < cut], df[df.timestamp >= cut]

    mean_pred = np.full(len(te), tr.cnt.mean())
    profile = tr.groupby(["hour", "is_weekend"]).cnt.mean()
    naive_pred = te.set_index(["hour", "is_weekend"]).index.map(profile).to_numpy(float)

    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=0)
    model.fit(tr[FEATURES], tr.cnt)
    pred = model.predict(te[FEATURES]).clip(0)

    def score(p):
        return {"mae": round(mean_absolute_error(te.cnt, p), 1),
                "r2": round(r2_score(te.cnt, p), 3)}

    res = {
        "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "test_start": str(te.timestamp.min().date()), "test_end": str(te.timestamp.max().date()),
        "mean_baseline": score(mean_pred),
        "seasonal_naive": score(naive_pred),
        "model": score(pred),
    }
    res["improvement_vs_naive_pct"] = round(
        (1 - res["model"]["mae"] / res["seasonal_naive"]["mae"]) * 100)

    # A week of actual-vs-predicted for the dashboard.
    sample = te.head(24 * 7)
    res["sample_week"] = {
        "timestamps": [str(t) for t in sample.timestamp],
        "actual": [int(v) for v in sample.cnt],
        "predicted": [round(float(v)) for v in pred[:len(sample)]],
    }
    return res


def main() -> None:
    df = load()
    results = {
        "data_quality": validate(df),
        "demand_shape": demand_shape(df),
        "weather": weather_effect(df),
        "seasonality": seasonality(df),
        "forecast": forecast(df),
        "headline": {
            "total_hires": int(df.cnt.sum()),
            "mean_hourly": round(df.cnt.mean()),
            "busiest_hour_count": int(df.cnt.max()),
            "busiest_hour_when": str(df.loc[df.cnt.idxmax(), "timestamp"]),
        },
    }
    OUT.write_text(json.dumps(results, indent=2))

    q, s, w, f = (results["data_quality"], results["demand_shape"],
                  results["weather"], results["forecast"])
    print(f"Data      {q['rows']:,} hours, {q['start']} to {q['end']}; "
          f"{q['missing_hours']} hours missing ({q['missing_pct']}%)")
    print(f"Commute   weekday 8am is {s['commute_ratio_8am']}x the weekend 8am level")
    print(f"Rain      overall -{w['overall']['drop_pct']}% | "
          f"commuters -{w['commute']['drop_pct']}% | leisure -{w['leisure']['drop_pct']}%")
    print(f"Forecast  MAE {f['model']['mae']} vs naive {f['seasonal_naive']['mae']} "
          f"({f['improvement_vs_naive_pct']}% better), R2 {f['model']['r2']}")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
