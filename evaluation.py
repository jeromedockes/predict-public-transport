from datetime import date, timedelta

import ibis
from ibis import _
import polars as pl


def cv_split(data, gap=3, test_length=90, min_train_size=90):
    split_dates = pl.date_range(
        data["DAY"].min() + timedelta(days=min_train_size),
        data["DAY"].max() - timedelta(days=gap),
        interval=timedelta(days=test_length),
        closed="left",
        eager=True,
    )
    for split_d in split_dates:
        train = (
            data.with_row_index().filter(pl.col("DAY") < split_d)["index"].to_numpy()
        )
        test = (
            data.with_row_index()
            .filter(
                (pl.col("DAY") >= split_d + timedelta(days=gap))
                & (pl.col("DAY") < split_d + timedelta(days=test_length + gap))
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
        return splits[max(0, len(splits) - self.max_splits):]

    def get_n_splits(self, X, y=None, groups=None):
        return len(self.split(X))
