# 🚧 WIP

setup

```
pip install git+https://github.com/jeromedockes/skrub.git@expr-v1
pip install duckdb plotly requests polars
pip install "ibis-framework[duckdb]"
```

run the scripts

```
# get data
python download.py
python download_holidays.py

# train a model & save in best-model.pickle; --report is to also open a report of the model
python train.py --report

# once that is done we can ask for a prediction for a given day
python predict.py 2023-01-01

# cross-validate & save out-of-sample predictions in cv_predictions.parquet
python train.py --cross_validate

# once that is done we can plot the predictions
python plot.py
```
