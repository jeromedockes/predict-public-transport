from datetime import timedelta
from pathlib import Path

import polars as pl
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error

data_dir = Path(__file__).parent
line_name = "T2"


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
    all_dates = usage.collect()['DATE']
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
    return usage.join_asof(events, on="DATE")


def add_holidays(usage):
    holidays = pl.read_parquet(data_dir / "holidays.parquet")
    return usage.with_columns(is_holiday=pl.col("DATE").is_in(holidays["date"]))


def load_features():
    usage = load_usage(line_name)
    usage = regular_time_grid(usage, 10)
    usage = add_datetime_features(usage)
    usage = add_lagged_features(usage)
    usage = add_school_holidays(usage)
    usage = add_holidays(usage)
    return usage


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


def get_cv_predictions(usage, model):
    all_test_data = []
    for i, (train, test) in enumerate(Splitter().split(usage)):
        print(i)
        train_data = usage.select(pl.all().gather(train))
        test_data = usage.select(pl.all().gather(test))
        print(f"  train: {train_data['DATE'].min()} - {train_data['DATE'].max()}")
        print(f"  test:  {test_data['DATE'].min()} - {test_data['DATE'].max()}")
        est = clone(model).fit(train_data.drop("N"), train_data["N"])
        pred = est.predict(test_data.drop("N"))
        err = mean_absolute_percentage_error(test_data["N"], pred)
        print(f"  MAPE: {err:.1%}")
        all_test_data.append(test_data.with_columns(predicted=pred))

    results = pl.concat(all_test_data)
    return results


if __name__ == "__main__":
    usage = load_features().drop_nulls("N").collect()
    results = get_cv_predictions(usage, HistGradientBoostingRegressor())
    results.write_parquet(f"{line_name}_cv_predictions.parquet")
