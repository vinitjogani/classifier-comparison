import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import datasets
from experiments import grid_search, plot


def trials(dataset):
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
    fig = plot(readings, "n_neighbors", "Neighbors (K)", "weights")

    readings.to_csv(f"readings/knn_{dataset}.csv", index=False)
    fig.savefig(f"readings/knn_{dataset}.png")


if __name__ == "__main__":
    trials("credit_score")
    trials("term_deposits")
