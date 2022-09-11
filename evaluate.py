import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, make_scorer, precision_recall_curve
from sklearn.model_selection import GridSearchCV
import datasets


def pr_auc_score(y_true, y_score, average="macro"):
    aucs = []
    counts = []
    for class_ in range(y_score.shape[1]):
        precision, recall, _ = precision_recall_curve(
            y_true[:, class_] == 1,
            y_score[:, class_],
        )
        aucs.append(auc(recall, precision))
        counts.append((y_true[:, class_] == 1).sum())

    if average == "macro":
        return sum(aucs) / len(aucs)
    else:
        return sum(np.array(aucs) * np.array(counts) / sum(counts))


def onehot(true):
    true_onehot = np.zeros((true.shape[0], true.max() + 1))
    for c in true.unique():
        true_onehot[true == c, c] = 1
    return true_onehot


def grid_search(X_train, y_train, model, **kwargs):
    gs = GridSearchCV(
        model,
        kwargs,
        n_jobs=-1,
        scoring={
            "accuracy": "accuracy",
            "auroc": "roc_auc",
            "auprc": make_scorer(pr_auc_score, needs_threshold=True),
        },
        cv=5,
        refit="auprc",
    )

    gs.fit(X_train, onehot(y_train))
    return pd.DataFrame(gs.cv_results_)


def plot(df, x, xlabel, grouping, dataset, log=True, fn="plot"):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(nrows=3, figsize=(10, 10), sharex=True)
    ax[-1].set_xlabel(xlabel)
    ax[0].set_title(dataset)

    def plotter(axis, *args, **kwargs):
        getattr(axis, fn)(*args, **kwargs)

    for g in df[grouping].unique():
        r1 = df[df[grouping] == g]
        r1 = r1.groupby(x, as_index=False).max()
        plotter(ax[0], r1[x], r1.mean_test_accuracy, label=f"{grouping}={g}")
        ax[0].set_ylabel("Accuracy")
        plotter(ax[1], r1[x], r1.mean_test_auroc, label=f"{grouping}={g}")
        ax[1].set_ylabel("AUROC")
        plotter(ax[2], r1[x], r1.mean_test_auprc, label=f"{grouping}={g}")
        ax[2].set_ylabel("AUPRC")

        if log:
            plt.xscale("log", base=2)

    if len(df[grouping].unique()) > 1:
        for a in ax:
            a.legend()

    fig.tight_layout()
    return fig


def run_trials(model, experiment, dataset, model_args, plot_args):
    model_name = model.__module__.split(".")[1]
    output = f"readings/{model_name}_{experiment}_{dataset}.csv"

    if not os.path.exists(output):
        (X_train, y_train), _ = datasets.load_dataset(dataset)
        readings = grid_search(X_train, y_train, model, **model_args)
        readings.to_csv(output, index=False)
    else:
        readings = pd.read_csv(output)

    fig = plot(readings, dataset=dataset, **plot_args)
    fig.savefig(output.replace(".csv", ".png"))
