import polars as pl
import matplotlib
from matplotlib import pyplot as plt

from data_access import (
    load_data_points,
    load_surface_features,
    polars_regular_time_grid,
)


matplotlib.use("tkagg")
T2 = "100__112__12"

results = pl.read_parquet("cv_predictions.parquet")
results = results.filter(pl.col("LINE") == T2)
results = polars_regular_time_grid(results)
fig, ax = plt.subplots()
ax.plot(results["DAY"], results["N"])
ax.plot(results["DAY"], results["predicted"])
plt.show()

import plotly.graph_objects as go

fig = go.Figure()
fig.update_layout(title="daily travellers on tramway T2")
fig.add_trace(
    go.Scatter(x=results["DAY"], y=results["N"], mode="lines", name="y_true")
)
fig.add_trace(
    go.Scatter(x=results["DAY"], y=results["predicted"], mode="lines", name="y_pred")
)
fig.write_html("predictions.html")
fig.show()
