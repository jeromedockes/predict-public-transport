import polars as pl
import matplotlib
from matplotlib import pyplot as plt

from train import regular_time_grid


matplotlib.use("tkagg")

results = pl.read_parquet("T2_cv_predictions.parquet")
results = regular_time_grid(results.lazy()).collect()
fig, ax = plt.subplots()
ax.plot(results["DATE"], results["N"])
ax.plot(results["DATE"], results["predicted"])
plt.show()

import plotly.graph_objects as go

fig = go.Figure()
fig.update_layout(title="daily travellers on tramway T2")
fig.add_trace(
    go.Scatter(x=results["DATE"], y=results["N"], mode="lines", name="y_true")
)
fig.add_trace(
    go.Scatter(x=results["DATE"], y=results["predicted"], mode="lines", name="y_pred")
)
fig.write_html("predictions.html")
fig.show()
