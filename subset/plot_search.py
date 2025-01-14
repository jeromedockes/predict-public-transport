import pickle

with open("search-model.pickle", "rb") as stream:
    search = pickle.load(stream)


fig = search.plot_parallel_coord()
fig.write_html("search_results.html")
fig.show()
