# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(style="whitegrid", font_scale=1.5)
# %config InlineBackend.figure_format='retina'

# %%
data_dir = "../../../benchmark/data/"
docs_dir = "../../../docs/_static/"


def data_path(name):  # noqa
    return data_dir + name + ".csv"


def docs_path(name):  # noqa
    return docs_dir + name + ".png"


def make_figure(bench_name, matrix_name="tabmat", title=None):  # noqa
    df = pd.read_csv(data_path(bench_name))
    df2 = (
        df.set_index(["operation", "storage"])
        .drop("design", axis=1)
        .stack()
        .reset_index()
        .rename(columns={0: "val", "level_2": "metric"})
        .set_index(["operation", "metric"])
        .sort_values("storage")
        .sort_index()
    )

    df2["norm"] = df2["val"] / df2[df2["storage"] == "tabmat"]["val"]
    df2 = df2.reset_index()
    df2["storage"] = [{"tabmat": matrix_name}.get(elt, elt) for elt in df2["storage"]]

    hue_order = [matrix_name] + [elt for elt in df["storage"] if elt != matrix_name]
    g = sns.FacetGrid(
        data=df2,
        row="operation",
        hue="storage",
        sharex=False,
        col="metric",
        hue_order=hue_order,
        palette=["k", "k", "k"],
        aspect=2,
    )
    g.map(plt.barh, "storage", "norm")

    for j, name in enumerate(["Matvec", "Sandwich", "Mat-T-Vec"]):
        g.axes[j, 0].set_ylabel(name)
        for k in range(2):
            g.axes[j, k].set_title("")
            g.axes[j, k].set_xlabel("")

    g.axes[0, 0].set_title("Memory (fraction of tabmat)")
    g.axes[0, 1].set_title("Time (fraction of tabmat)")

    if title is not None:
        plt.suptitle(title, y=1.05)

    plt.tight_layout()
    plt.savefig(docs_path(bench_name), dpi=300)


# %%
make_figure("sparse_bench", "SparseMatrix", "Sparse Matrix Benchmark")

# %%
make_figure("dense_bench", "DenseMatrix", "Dense Matrix Benchmark")

# %%
make_figure("one_cat_bench", "CategoricalMatrix", "Categorical Matrix Benchmark")

# %%
make_figure("two_cat_bench", "SplitMatrix", "Two-Categorical Matrix Benchmark")

# %%
make_figure("dense_cat_bench", "SplitMatrix", "Dense + Two Categoricals Benchmark")

# %%
bench_names = [
    ("sparse_bench", "sparse"),
    ("one_cat_bench", "one categorical"),
    ("two_cat_bench", "two categoricals"),
    ("dense_cat_bench", "dense plus categoricals"),
]
dfs = []
for n, rename in bench_names:
    df = pd.read_csv(data_path(n))
    df = df[df["operation"] == "sandwich"].drop(["operation"], axis=1)
    df["design"] = rename
    dfs.append(df)

# %%
df = pd.concat(dfs).set_index(["design"])  # , 'storage'])


def get_ratio_time(x):  # noqa
    x["time_ratio"] = x["time"] / x[x["storage"] == "tabmat"]["time"]
    return x


df = df.groupby(["design"]).apply(get_ratio_time)


def get_ratio_mem(x):  # noqa
    x["memory_ratio"] = x["memory"] / x[x["storage"] == "tabmat"]["memory"]
    return x


df = df.groupby(["design"]).apply(get_ratio_mem)

# %%
SMALL_SIZE = 13
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %%
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.0, 10))


plot_df = df[["storage", "time_ratio"]].pivot(columns="storage")
plot_df.index.name = ""
plot_df.columns = plot_df.columns.get_level_values(1)

plot_df.plot.bar(
    ax=axes[0],
    ylim=[0, 80],
    title="Runtime of sandwich products",
    legend=False,
    width=0.8,
    ylabel="time (fraction of best)",
    yticks=[0, 20, 40, 60, 80],
    cmap="Paired",
)
plt.sca(axes[0])
plt.legend(bbox_to_anchor=(1, 1.05), loc="upper right", ncol=1)
plt.xticks(rotation=45, ha="right")

ax = plt.gca()

# Hide the right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

plt.tight_layout()

plot_df = df[["storage", "memory_ratio"]].pivot(columns="storage")
plot_df.index.name = ""
plot_df.columns = plot_df.columns.get_level_values(1)

plot_df.plot.bar(
    ax=axes[1],
    ylim=[0, 12],
    title="Memory usage of sandwich products",
    legend="upper right",
    width=0.8,
    ylabel="memory usage (fraction of best)",
    yticks=[0, 3, 6, 9, 12],
    cmap="Paired",
)
plt.sca(axes[1])
plt.legend(bbox_to_anchor=(1.05, 1.05), loc="upper right", ncol=1)
plt.xticks(rotation=45, ha="right")

ax = plt.gca()

# Hide the right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

plt.tight_layout()

plt.savefig(docs_path("headline"), dpi=300)
plt.show()
