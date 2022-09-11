import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import datasets
from experiments import grid_search, plot


def pruning_trials(dataset):
    (X_train, y_train), _ = datasets.load_dataset(dataset)

    readings = pd.DataFrame(
        grid_search(
            X_train,
            y_train,
            HistGradientBoostingClassifier,
            max_leaf_nodes=[8, 16, 32, 64, 128],
            l2_regularization=[0.35],
            learning_rate=[0.035],
            validation_fraction=[None],
            max_iter=[25, 50, 100, 150],
        )
    )
    readings.to_csv(f"readings/boosting_pruning_{dataset}.csv", index=False)

    fig = plot(readings, "max_leaf_nodes", "Max Leaf Nodes", "max_iter", dataset)
    fig.savefig(f"readings/boosting_pruning_{dataset}.png")


if __name__ == "__main__":
    # neighbor_trials("credit_score")
    # neighbor_trials("term_deposits")

    pruning_trials("credit_score")
    pruning_trials("term_deposits")
