from datetime import timedelta
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import KBinsDiscretizer
import polars as pl
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import skrub

from neuralk_foundry.models import RemoteClassifier

data_dir = Path(__file__).parent


def load_usage(line_name):
    usage = (
        pl.scan_parquet(data_dir / f"{line_name}.parquet")
        .select(
            pl.col("JOUR").alias("DATE"),
            pl.col("NB_VALD")
            .str.strip_chars()
            .replace("Moins de 5", "4")
            .cast(pl.Int32)
            .alias("N"),
        )
        .group_by("DATE")
        .agg(pl.col("N").sum())
        .sort("DATE")
    )
    return usage


def regular_time_grid(usage, max_offset=0):
    all_dates = usage.collect()["DATE"]
    date_range = pl.date_range(
        all_dates.min(),
        all_dates.max() + timedelta(days=max_offset),
        "1d",
        eager=True,
    )
    dates = pl.LazyFrame({"DATE": date_range})
    return dates.join(usage, on=["DATE"], how="left")


def add_lagged_features(usage):
    lags = {
        f"N_lag_{lag}": pl.col("N").shift(lag)
        for lag in [3, 4, 5, 6, 7, 14, 21, 28, 35]
    }
    averages = {}
    avg_lag = 3
    for width in [3, 7, 30, 90]:
        avg = pl.mean("N").rolling(
            index_column="DATE", period=f"{width}d", offset=f"{- width - avg_lag}d"
        )
        averages[f"N_lag_{avg_lag}_avg_{width}"] = avg
    usage = usage.with_columns(**lags, **averages)
    return usage


def add_datetime_features(usage):
    return usage.with_columns(
        pl.col("DATE").dt.year().alias("year"),
        pl.col("DATE").dt.month().alias("month"),
        pl.col("DATE").dt.day().alias("day"),
        pl.col("DATE").dt.weekday().alias("weekday"),
        pl.col("DATE").dt.ordinal_day().alias("day_of_year"),
    )


def add_school_holidays(usage):
    holidays = pl.scan_parquet(data_dir / "school_holidays.parquet").filter(
        pl.col("location") == "Paris", pl.col("population").is_in(["-", "Élèves"])
    )
    start = holidays.select(
        DATE=pl.col("start_date").cast(pl.Date),
        is_school_holiday=1,
    )
    end = holidays.select(
        DATE=pl.col("end_date").cast(pl.Date),
        is_school_holiday=0,
    )
    events = pl.concat([start, end]).sort("DATE")
    return usage.join_asof(events, on="DATE").with_columns(
        pl.col("is_school_holiday").fill_null(0)
    )


def add_holidays(usage):
    holidays = pl.read_parquet(data_dir / "holidays.parquet")
    return usage.with_columns(is_holiday=pl.col("DATE").is_in(holidays["date"]))


def add_features(dates, line_name, *, lagged, school_holidays, holidays):
    usage = load_usage(line_name)
    usage = regular_time_grid(usage, 10)
    usage = add_datetime_features(usage)
    if lagged:
        usage = add_lagged_features(usage)
    if school_holidays:
        usage = add_school_holidays(usage)
    if holidays:
        usage = add_holidays(usage)
    usage = usage.drop("N").collect()
    return dates.join(usage, on="DATE", how="left").drop("DATE")


class NiclRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, url, transfer_mode):
        self.url = url
        self.transfer_mode = transfer_mode

    def fit(self, X, y):
        self.discretizer_ = KBinsDiscretizer(
            n_bins=min(200, max(1, len(set(y)) // 5)), encode="ordinal"
        )
        y = self.discretizer_.fit_transform(np.asarray([y]).T).squeeze()
        self.classifier_ = RemoteClassifier(self.url, self.transfer_mode).fit(X, y)
        return self

    def predict(self, X):
        y = self.classifier_.predict(X)
        return self.discretizer_.inverse_transform(np.asarray([y]).T).squeeze()


def get_predictor(line_name):
    data = skrub.var("data")
    dates = data.select("DATE").skb.mark_as_X()
    counts = data["N"].skb.mark_as_y()
    X = skrub.deferred(add_features)(
        dates,
        line_name,
        lagged=skrub.choose_bool(name="use_lagged_features"),
        school_holidays=skrub.choose_bool(name="use_school_holidays"),
        holidays=skrub.choose_bool(name="use_holidays"),
    ).skb.apply(SimpleImputer())
    hgb = HistGradientBoostingRegressor(
        learning_rate=skrub.choose_float(0.001, 0.8, log=True, name="lr"),
        max_leaf_nodes=skrub.choose_int(2, 65, log=True, name="max leaf nodes"),
        max_bins=skrub.choose_int(3, 256, log=True, name="max bins"),
        min_samples_leaf=skrub.choose_int(1, 100, log=True, name="min samples leaf"),
        early_stopping=True,
        n_iter_no_change=10,
        max_iter=1000,
    )
    nicl = NiclRegressor(
        "https://nicl-723294174640.europe-west1.run.app",
        transfer_mode=("storage", "foundry-datasets"),
    )
    model = skrub.choose_from({"nicl": nicl, "hgb": hgb}, name="model")
    pred = X.skb.apply(model, y=counts)
    return pred


def cv_split(data, gap=3, test_length=90, min_train_size=90):
    split_dates = pl.date_range(
        data["DATE"].min() + timedelta(days=min_train_size),
        data["DATE"].max() - timedelta(days=gap),
        interval=timedelta(days=test_length),
        closed="left",
        eager=True,
    )
    for split_d in split_dates:
        train = (
            data.with_row_index().filter(pl.col("DATE") < split_d)["index"].to_numpy()
        )
        test = (
            data.with_row_index()
            .filter(
                (pl.col("DATE") >= split_d + timedelta(days=gap))
                & (pl.col("DATE") < split_d + timedelta(days=test_length + gap))
            )["index"]
            .to_numpy()
        )
        if len(train) and len(test):
            yield train, test


class Splitter:
    def __init__(self, max_splits=None, gap=3, test_length=90, min_train_size=90):
        self.max_splits = max_splits
        self.gap = gap
        self.test_length = test_length
        self.min_train_size = min_train_size

    def split(self, X, y=None, groups=None):
        splits = cv_split(
            X,
            gap=self.gap,
            test_length=self.test_length,
            min_train_size=self.min_train_size,
        )
        splits = list(splits)
        if self.max_splits is None:
            return splits
        return splits[max(0, len(splits) - self.max_splits) :]

    def get_n_splits(self, X, y=None, groups=None):
        return len(self.split(X))
