import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import datasets
from experiments import grid_search, plot


def neighbor_trials(dataset):
    (X_train, y_train), _ = datasets.load_dataset(dataset)

    readings = pd.DataFrame(
        grid_search(
            X_train,
            y_train,
            KNeighborsClassifier,
            n_neighbors=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            weights=["uniform", "distance"],
        ),
    )
    fig = plot(readings, "n_neighbors", "Neighbors (K)", "weights", dataset)

    readings.to_csv(f"readings/knn_neighbors_{dataset}.csv", index=False)
    fig.savefig(f"readings/knn_neighbors_{dataset}.png")


def metric_trials(dataset):
    (X_train, y_train), _ = datasets.load_dataset(dataset)

    NEIGHBORS = {
        "credit_score": 16,
        "term_deposits": 256,
    }

    readings = pd.DataFrame(
        grid_search(
            X_train,
            y_train,
            KNeighborsClassifier,
            n_neighbors=[NEIGHBORS[dataset]],
            weights=["distance"],
            metric=["manhattan", "cosine", "l2"],
        ),
    )
    readings.to_csv(f"readings/knn_metrics_{dataset}.csv", index=False)

    fig = plot(readings, "metric", "Metric", "weights", dataset, log=False, fn="bar")
    fig.savefig(f"readings/knn_metrics_{dataset}.png")


if __name__ == "__main__":
    # neighbor_trials("credit_score")
    # neighbor_trials("term_deposits")

    metric_trials("credit_score")
    metric_trials("term_deposits")
