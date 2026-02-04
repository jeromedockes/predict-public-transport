import pickle

import utils

LINE_NAME = "T2"

usage = utils.load_usage(LINE_NAME).collect()
pred = utils.get_predictor(LINE_NAME)
db = 'sqlite:///optuna.sqlite3'
search = pred.skb.make_randomized_search(
    backend="optuna",
    scoring="neg_median_absolute_error",
    cv=utils.Splitter(),
    n_iter=32,
    n_jobs=8,
    verbose=0,
    storage=db,
)
search.fit({"data": usage})

with open("search-model.pickle", "wb") as stream:
    pickle.dump(search, stream)

with open("best-model.pickle", "wb") as stream:
    pickle.dump(search.best_learner_, stream)

print(search.results_)
search.plot_results().show()
