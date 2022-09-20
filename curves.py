import pickle
import matplotlib.pyplot as plt
from knn import best as best_knn
from tree import best as best_tree
from mlp import best as best_mlp
from boosting import best as best_boosting
from svm import best as best_svm
from evaluate import learning_curves, iter_learning_curves, pr_auc_score
from datasets import load_dataset
import numpy as np
import os
import time
import pandas as pd


def summary():
    models = {
        "KNN": best_knn,
        "Decision Tree": best_tree,
        "Neural Network": best_mlp,
        "SVM": best_svm,
        "Boosting": best_boosting,
    }
    model_names = list(models)

    datasets = ["credit_score", "term_deposits"]
    out = []
    for di, dataset in enumerate(datasets):
        (X_train, y_train), (X_test, y_test) = load_dataset(dataset)

        for mi, model_name in enumerate(model_names):
            print(dataset, model_name)
            model = models[model_name](dataset)
            start = time.time()
            model.fit(X_train, y_train)
            fitting_time = time.time() - start
            y_pred = model.predict_proba(X_train)
            train_auc = pr_auc_score(y_train, y_pred)
            start = time.time()
            y_pred = model.predict_proba(X_test)
            testing_time = time.time() - start
            test_auc = pr_auc_score(y_test, y_pred)
            out.append(
                {
                    "Model": model_name,
                    "Train AUC": train_auc,
                    "Test AUC": test_auc,
                    "Fitting Time": fitting_time,
                    "Testing Time": testing_time,
                }
            )

    pd.DataFrame(out).to_csv("readings/summary.csv", index=False)


def by_training_size():

    plt.style.use("ggplot")

    models = {
        "KNN": best_knn,
        "Decision Tree": best_tree,
        "Neural Network": best_mlp,
        "SVM": best_svm,
        "Boosting": best_boosting,
    }
    model_names = list(models)

    datasets = ["credit_score", "term_deposits"]
    fig, ax = plt.subplots(
        ncols=len(models),
        nrows=len(datasets),
        figsize=(24, 8),
        sharex=True,
        sharey=True,
    )

    for di, dataset in enumerate(datasets):
        ax[di, 0].set_ylabel(f"AUPRC on {dataset}")

        (X_train, y_train), (X_test, y_test) = load_dataset(dataset)

        for mi, model_name in enumerate(model_names):
            print(dataset, model_name)
            ax[-1, mi].set_xlabel("Training Set Size (%)")
            model = models[model_name](dataset)
            cache = f"readings/{dataset}_{model_name}.pkl"
            if os.path.exists(cache):
                curves = pickle.load(open(cache, "rb"))
            else:
                curves = learning_curves(model, X_train, y_train, X_test, y_test)
                pickle.dump(curves, open(cache, "wb"))
            x, y1, y2 = curves
            ax[0, mi].set_title(model_name)
            ax[di, mi].plot(x, y1, label="train")
            ax[di, mi].plot(x, y2, label="test")
            ax[di, mi].legend()

    plt.tight_layout()
    fig.savefig("readings/by_training_size.png")


def by_iterations():
    plt.style.use("ggplot")

    datasets = ["credit_score", "term_deposits"]
    fig, ax = plt.subplots(
        ncols=2,
        nrows=len(datasets),
        figsize=(24, 8),
        sharex=True,
        sharey=True,
    )

    for di, dataset in enumerate(datasets):
        ax[di, 0].set_ylabel(f"AUPRC on {dataset}")

        (X_train, y_train), (X_test, y_test) = load_dataset(dataset)
        models = {
            "Neural Network": best_mlp(dataset).set_params(
                max_iter=100,
                warm_start=True,
            ),
            "Boosting": best_boosting(dataset).set_params(
                max_iter=100,
                warm_start=True,
            ),
        }
        model_names = list(models)
        model_names.sort()
        for mi, model_name in enumerate(model_names):
            print(dataset, model_name)
            ax[-1, mi].set_xlabel("Iterations")
            model = models[model_name]
            cache = f"readings/iter_{dataset}_{model_name}.pkl"
            if os.path.exists(cache):
                curves = pickle.load(open(cache, "rb"))
            else:
                curves = iter_learning_curves(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    end_fn=lambda m: m.set_params(max_iter=m.max_iter + 100),
                )
                pickle.dump(curves, open(cache, "wb"))
            x, y1, y2 = curves
            ax[0, mi].set_title(model_name)
            ax[di, mi].plot(x * 100, y1, label="train")
            ax[di, mi].plot(x * 100, y2, label="test")
            ax[di, mi].legend()

    plt.tight_layout()
    fig.savefig("readings/by_iterations.png")


if __name__ == "__main__":
    summary()
    by_training_size()
    by_iterations()
