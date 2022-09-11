import matplotlib.pyplot as plt
from evaluate import *


def grid_search(X_train, y_train, model_cls, grid={}, **kwargs):
    if len(kwargs) == len(grid):
        model_fn = create_model_fn(model_cls, grid)
        metrics = cross_validate(X_train, y_train, model_fn)
        metrics.update(grid)
        return [metrics]

    for arg in kwargs:
        if arg in grid:
            continue
        break

    out = []
    for value in kwargs[arg]:
        new = dict(grid)
        new[arg] = value
        out.extend(grid_search(X_train, y_train, model_cls, new, **kwargs))
    return out


def plot(df, x, xlabel, grouping, log=True):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(nrows=3, figsize=(10, 10), sharex=True)
    ax[-1].set_xlabel(xlabel)

    for g in df[grouping].unique():
        r1 = df[df[grouping] == g]
        r1 = r1.groupby(x, as_index=False).max()
        ax[0].plot(r1[x], r1.Accuracy, label=g)
        ax[0].set_ylabel("Accuracy")
        ax[1].plot(r1[x], r1.AUROC, label=g)
        ax[1].set_ylabel("AUROC")
        ax[2].plot(r1[x], r1.AUPRC, label=g)
        ax[2].set_ylabel("AUPRC")
        if log:
            plt.xscale("log", base=2)

    for a in ax:
        a.legend()
    fig.tight_layout()
    return fig
