import argparse
import pickle
import datetime

import polars as pl
import ibis

parser = argparse.ArgumentParser()
parser.add_argument("date")
args = parser.parse_args()
date = datetime.date.fromisoformat(args.date)

assert date < datetime.date(2023, 1, 11), "Too far in the future"

connection = ibis.duckdb.connect("data/ratp.duckdb", read_only=True)

query = pl.DataFrame({"DAY": [date], "LINE": ["100__112__12"]})

with open("best-model.pickle", "rb") as stream:
    model = pickle.load(stream)

prediction = model.predict({"all_days": query, "connection": connection})

print(f"prediction: {int(prediction[0]):,} travellers on {date:%a %d %b %Y}")
