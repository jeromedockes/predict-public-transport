import pickle
from pathlib import Path

import utils

LINE_NAME = "T2"

usage = utils.load_usage(LINE_NAME).collect()

if (pickle_path := Path("best-model.pickle")).is_file():
    with open("best-model.pickle", "rb") as stream:
        est = pickle.load(stream)
else:
    pred = utils.get_predictor(LINE_NAME)
    est = pred.skb.get_estimator()

est.fit({"data": usage})

with open("model.pickle", "wb") as stream:
    pickle.dump(est, stream)
