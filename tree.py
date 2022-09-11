import os

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import datasets
from evaluate import grid_search, plot


def pruning_trials(dataset):
    if not os.path.exists(f"readings/tree_pruning_{dataset}.csv"):
        (X_train, y_train), _ = datasets.load_dataset(dataset)

        readings = grid_search(
            X_train,
            y_train,
            DecisionTreeClassifier(),
            min_samples_leaf=[1, 8, 16, 32, 64, 128, 256],
            max_depth=[None, 8, 16, 32, 64],
            criterion=["gini", "entropy"],
        )
        readings.to_csv(f"readings/tree_pruning_{dataset}.csv", index=False)
    else:
        readings = pd.read_csv(f"readings/tree_pruning_{dataset}.csv")

    fig = plot(
        readings,
        "param_min_samples_leaf",
        "Min Samples Leaf",
        "param_criterion",
        dataset,
    )
    fig.savefig(f"readings/tree_pruning_{dataset}.png")


if __name__ == "__main__":
    # neighbor_trials("credit_score")
    # neighbor_trials("term_deposits")

    pruning_trials("credit_score")
    pruning_trials("term_deposits")
