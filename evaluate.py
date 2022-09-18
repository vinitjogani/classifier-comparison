import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, make_scorer, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
import datasets


def onehot(true):
    true_onehot = np.zeros((true.shape[0], true.max() + 1))
    for c in true.unique():
        true_onehot[true == c, c] = 1
    return true_onehot


def make_2d(fn):
    def out(y_true, y_score, *args, **kwargs):
        if len(y_score.shape) == 1:
            y_score = np.stack([1 - y_score, y_score], axis=-1)
        return fn(y_true, y_score, *args, **kwargs)

    return out


def friendly_auroc(y_true, y_score):
    return roc_auc_score(onehot(y_true), y_score)


def pr_auc_score(y_true, y_score, average="macro"):
    aucs = []
    counts = []
    for class_ in range(y_score.shape[1]):
        precision, recall, _ = precision_recall_curve(
            y_true == class_,
            y_score[:, class_],
        )
        aucs.append(auc(recall, precision))
        counts.append((y_true == class_).sum())

    if average == "macro":
        return sum(aucs) / len(aucs)
    else:
        return sum(np.array(aucs) * np.array(counts) / sum(counts))


def grid_search(X_train, y_train, model, **kwargs):
    gs = GridSearchCV(
        model,
        kwargs,
        n_jobs=1,
        scoring={
            "accuracy": "accuracy",
            "auroc": make_scorer(make_2d(friendly_auroc), needs_proba=True),
            "auprc": make_scorer(make_2d(pr_auc_score), needs_proba=True),
        },
        cv=5,
        refit="auprc",
    )

    gs.fit(X_train, y_train)
    return pd.DataFrame(gs.cv_results_)


def plot(df, x, xlabel, grouping, dataset, log=True):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(nrows=2, figsize=(6, 6), sharex=True)
    ax[-1].set_xlabel(xlabel)
    ax[0].set_title(dataset)

    if grouping not in df.columns:
        df = df.copy()
        df[grouping] = 1

    for g in df[grouping].unique():
        r1 = df[df[grouping] == g]
        r1 = r1.groupby(x, as_index=False).max()
        ax[0].plot(r1[x], r1.mean_test_accuracy, label=f"{grouping}={g}")
        ax[0].set_ylabel("Accuracy")
        # ax[1].plot(r1[x], r1.mean_test_auroc, label=f"{grouping}={g}")
        # ax[1].set_ylabel("AUROC")
        ax[1].plot(r1[x], r1.mean_test_auprc, label=f"{grouping}={g}")
        ax[1].set_ylabel("AUPRC")

        if log:
            plt.xscale("log", base=2)

    if len(df[grouping].unique()) > 1:
        for a in ax:
            a.legend()

    fig.tight_layout()
    return fig


def bar(df, x, xlabel, grouping, dataset):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(nrows=2, figsize=(6, 6), sharex=True)
    for a in ax:
        a.set_xlabel(xlabel)
    ax[0].set_title(dataset)

    if grouping not in df.columns:
        df = df.copy()
        df[grouping] = 1

    for g in df[grouping].unique():
        r1 = df[df[grouping] == g]
        r1 = r1.groupby(x, as_index=False).max()
        ax[0].bar(r1[x], r1.mean_test_accuracy, label=f"{grouping}={g}")
        ax[0].set_ylabel("Accuracy")
        ax[0].set_ylim(r1.mean_test_accuracy.min() * 0.9)
        # ax[1].bar(r1[x], r1.mean_test_auroc, label=f"{grouping}={g}")
        # ax[1].set_ylabel("AUROC")
        # ax[1].set_ylim(r1.mean_test_auroc.min() * 0.9)
        ax[1].bar(r1[x], r1.mean_test_auprc, label=f"{grouping}={g}")
        ax[1].set_ylabel("AUPRC")
        ax[1].set_ylim(r1.mean_test_auprc.min() * 0.9)

    if len(df[grouping].unique()) > 1:
        for a in ax:
            a.legend()

    fig.tight_layout()
    return fig


def run_trials(
    model,
    experiment,
    dataset,
    model_args,
    plot_args,
    clean_readings=None,
    plotter="plot",
):
    model_name = model.__module__.split(".")[1]
    print(f"Running {experiment} experiment on {dataset} with {model_name}...")
    output = f"readings/{model_name}_{experiment}_{dataset}.csv"

    if not os.path.exists(output):
        (X_train, y_train), _ = datasets.load_dataset(dataset)
        readings = grid_search(X_train, y_train, model, **model_args)
        readings.to_csv(output, index=False)
    else:
        readings = pd.read_csv(output)

    if clean_readings:
        clean_readings(readings)

    if plotter == "plot":
        fig = plot(readings, dataset=dataset, **plot_args)
    else:
        fig = bar(readings, dataset=dataset, **plot_args)
    fig.savefig(output.replace(".csv", ".png"))
    return readings
