import datetime
import pickle

import polars as pl


with open("model.pickle", "rb") as stream:
    model = pickle.load(stream)

date = datetime.date(2023, 1, 2)
data = pl.DataFrame({"DATE": [date]})
pred = model.predict({"data": data})[0]
print(f"prediction for {date}: {pred:,.0f} travellers")
