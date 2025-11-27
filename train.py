import argparse
import pickle

import polars as pl
import ibis
from ibis import _
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
import skrub

from data_access import load_surface, load_surface_features
from evaluation import Splitter

T2 = "100__112__12"
USE_T2_ONLY = True
parser = argparse.ArgumentParser()
parser.add_argument("--cross_validate", action="store_true")
parser.add_argument("--report", action="store_true")
cl_args = parser.parse_args()


connection = ibis.duckdb.connect("data/ratp.duckdb", read_only=True)
all_days_df = load_surface(connection)

if USE_T2_ONLY:
    all_days_df = all_days_df.filter(_.LINE == T2)

all_days_df = all_days_df.to_polars()

# %%
all_days = skrub.var("all_days", all_days_df)
X = all_days.select(["DAY", "LINE"]).skb.mark_as_X()
y = all_days["N"].skb.mark_as_y()

con = skrub.var("connection", connection)
X = skrub.deferred(ibis.memtable)(X)
features = skrub.deferred(load_surface_features)(con).skb.set_name("features")
X = (
    X.join(features, [X.DAY == features.DAY, X.LINE == features.LINE])
    .order_by(["DAY", "LINE"])
    .drop(["DAY", "LINE_NAME"])
    .to_polars()
)
X = X.skb.apply(
    OrdinalEncoder(unknown_value=float("nan"), handle_unknown="use_encoded_value"),
    cols="LINE",
)

# %%
hgb = HistGradientBoostingRegressor(
    learning_rate=skrub.choose_float(0.001, 0.8, log=True, name="lr"),
    max_leaf_nodes=skrub.choose_int(2, 65, log=True, name="max leaf nodes"),
    max_bins=skrub.choose_int(3, 256, log=True, name="max bins"),
    min_samples_leaf=skrub.choose_int(1, 100, log=True, name="min samples leaf"),
    early_stopping=True,
    n_iter_no_change=10,
    max_iter=1000,
)
prediction = X.skb.apply(hgb, y=y)

# %%
if cl_args.report:
    prediction.skb.full_report()

# %%

# Note: ibis tables cannot be passed to subprocesses; to use n_jobs we would
# have to do the join in polars
features_df = load_surface_features(connection).cache()

# %%


def get_cv_predictions():
    all_test_data = []
    for i, (train, test) in enumerate(Splitter().split(all_days_df)):
        print(f"=================== fold {i} ========================")
        est = prediction.skb.get_randomized_search(
            cv=Splitter(min_train_size=60 if i == 0 else 90, max_splits=4),
            verbose=1,
            n_iter=8,
            scoring="neg_mean_absolute_percentage_error",
        )
        est.fit(
            {
                "all_days": all_days_df.select(pl.all().gather(train)),
                "features": features_df,
            }
        )
        test_data = all_days_df.select(pl.all().gather(test))
        pred = est.predict({"all_days": test_data, "features": features_df})
        all_test_data.append(test_data.with_columns(predicted=pred))
        print(est.get_cv_results_table())
        print()

    results = pl.concat(all_test_data)
    return results


if cl_args.cross_validate:

    results = get_cv_predictions()
    results.write_parquet("cv_predictions.parquet")

else:
    estimator = prediction.skb.get_randomized_search(
        cv=Splitter(max_splits=10),
        verbose=1,
        n_iter=16,
        scoring="neg_mean_absolute_percentage_error",
    )
    estimator.fit({"all_days": all_days_df, "features": features_df})
    print(estimator.get_cv_results_table())

    # %%
    with open("search-model.pickle", "wb") as stream:
        pickle.dump(estimator, stream)

    with open("best-model.pickle", "wb") as stream:
        pickle.dump(estimator.best_estimator_, stream)
