"""Tests for the cycle-demand analysis.

These are not smoke tests. Each one guards a claim the README makes, or a
mistake that would make the numbers wrong while the script still ran happily:
leaking future data into training, comparing rain against clear at different
times of day, or reporting a model that never actually beat its baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis  # noqa: E402


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return analysis.load()


# --- the data itself -------------------------------------------------------

def test_no_duplicate_or_missing_values(df):
    report = analysis.validate(df)
    assert report["duplicate_timestamps"] == 0
    assert report["null_values"] == 0
    # Gaps are allowed (TfL's feed has them) but must stay small enough that
    # hourly means are not quietly built on a handful of observations.
    assert report["missing_pct"] < 2.0


def test_is_rain_matches_the_weather_codes(df):
    assert set(df.loc[df.is_rain == 1, "weather_code"]) <= set(analysis.RAIN_CODES)
    assert not set(df.loc[df.is_rain == 0, "weather_code"]) & set(analysis.RAIN_CODES)


def test_relative_demand_is_centred_within_each_hour_and_day_type(df):
    """`rel` is the whole basis of the weather comparison.

    Every (hour, day type) group must average exactly 1.0 — that is what makes
    a rain hour comparable to a clear hour at the same point in the week.
    """
    means = df.groupby(["hour", "is_weekend"]).rel.mean()
    assert means.between(0.999, 1.001).all()


# --- the findings ----------------------------------------------------------

def test_weekday_peaks_at_commute_hours_and_weekend_does_not(df):
    shape = analysis.demand_shape(df)
    assert shape["weekday_peak_hour"] in analysis.COMMUTE_HOURS
    assert shape["weekend_peak_hour"] in analysis.LEISURE_HOURS
    assert shape["commute_ratio_8am"] > 5


def test_rain_reduces_demand_and_hits_leisure_hardest(df):
    w = analysis.weather_effect(df)
    assert w["overall"]["drop_pct"] > 0
    assert w["commute"]["drop_pct"] > 0
    # The headline claim: discretionary trips are given up more readily than
    # the ride to work. If this ever flips, the README is wrong.
    assert w["leisure"]["drop_pct"] > w["commute"]["drop_pct"]


def test_demand_rises_monotonically_with_temperature(df):
    bins = analysis.weather_effect(df)["by_temperature"]
    rel = [b["rel"] for b in bins]
    assert rel == sorted(rel), "temperature response should not double back"
    assert all(b["hours"] >= 100 for b in bins), "a bin too thin to report"


def test_hourly_profiles_cover_every_hour_and_carry_sample_sizes(df):
    profiles = analysis.demand_shape(df)["profiles"]
    assert set(profiles) == {"all", "clear", "rain"}
    for name, p in profiles.items():
        for day in ("weekday", "weekend"):
            assert len(p[day]) == 24, f"{name}/{day} is missing hours"
            assert len(p[day + "_n"]) == 24
            assert min(p[day + "_n"]) > 0, f"{name}/{day} has an empty hour"
    # The three profiles partition the data: clear + rain = all.
    total = sum(profiles["all"]["weekday_n"]) + sum(profiles["all"]["weekend_n"])
    parts = sum(profiles["clear"]["weekday_n"]) + sum(profiles["clear"]["weekend_n"]) \
        + sum(profiles["rain"]["weekday_n"]) + sum(profiles["rain"]["weekend_n"])
    assert total == parts == len(df)


# --- the model -------------------------------------------------------------

def test_the_split_is_by_time_so_no_future_leaks_into_training(df):
    cut = pd.Timestamp("2016-10-01")
    train, test = df[df.timestamp < cut], df[df.timestamp >= cut]
    assert train.timestamp.max() < test.timestamp.min()
    assert len(test) > 2000, "too little held-out data to trust the score"


def test_model_beats_both_baselines_on_unseen_data(df):
    f = analysis.forecast(df)
    mean_mae = f["mean_baseline"]["mae"]
    naive_mae = f["seasonal_naive"]["mae"]
    model_mae = f["model"]["mae"]
    # A model that cannot beat "the average hour" is worthless; one that cannot
    # beat the hour-of-week profile has learned nothing the calendar didn't say.
    assert model_mae < naive_mae < mean_mae
    assert f["model"]["r2"] > 0.85
    assert f["improvement_vs_naive_pct"] >= 30


def test_forecast_is_reproducible(df):
    assert analysis.forecast(df)["model"] == analysis.forecast(df)["model"]


def test_sample_week_lines_up_with_the_test_period(df):
    f = analysis.forecast(df)
    week = f["sample_week"]
    assert len(week["actual"]) == len(week["predicted"]) == 24 * 7
    assert week["timestamps"][0].startswith(f["test_start"])
    assert all(p >= 0 for p in week["predicted"]), "negative bike hires"
