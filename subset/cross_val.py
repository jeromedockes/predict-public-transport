import pickle
from pathlib import Path

import polars as pl
from sklearn.base import clone
from sklearn.metrics import mean_absolute_percentage_error

import utils

LINE_NAME = "T2"


def get_cv_predictions(usage, estimator):
    all_test_data = []
    for i, (train, test) in enumerate(utils.Splitter().split(usage)):
        print(i)
        train_data = usage.select(pl.all().gather(train))
        test_data = usage.select(pl.all().gather(test))
        print(f"  train: {train_data['DATE'].min()} - {train_data['DATE'].max()}")
        print(f"  test:  {test_data['DATE'].min()} - {test_data['DATE'].max()}")
        est = clone(estimator).fit({"data": train_data})
        pred = est.predict({"data": test_data})
        err = mean_absolute_percentage_error(test_data["N"], pred)
        print(f"  MAPE: {err:.1%}")
        all_test_data.append(test_data.with_columns(predicted=pred))

    results = pl.concat(all_test_data)
    return results


usage = utils.load_usage(LINE_NAME).collect()
pred = utils.get_predictor(LINE_NAME)

if (pickle_path := Path("best-model.pickle")).is_file():
    with open("best-model.pickle", "rb") as stream:
        est = pickle.load(stream)
else:
    pred = utils.get_predictor(LINE_NAME)
    est = pred.skb.make_learner()

results = get_cv_predictions(usage, est)
results.write_parquet(f"{LINE_NAME}_cv_predictions.parquet")
