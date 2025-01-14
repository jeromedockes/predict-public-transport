import pickle

import utils

LINE_NAME = "T2"

usage = utils.load_usage(LINE_NAME).collect()
pred = utils.get_predictor(LINE_NAME)
search = pred.skb.get_randomized_search(
    scoring="neg_median_absolute_error",
    cv=utils.Splitter(),
    n_iter=32,
    verbose=1,
    n_jobs=8,
)
search.fit({"data": usage})
print(search.get_cv_results_table())
search.plot_parallel_coord().show()

with open("search-model.pickle", "wb") as stream:
    pickle.dump(search, stream)

with open("best-model.pickle", "wb") as stream:
    pickle.dump(search.best_estimator_, stream)
