import pandas as pd
from sklearn.neural_network import MLPClassifier

import datasets
from experiments import grid_search, plot


def size_trials(dataset):
    (X_train, y_train), _ = datasets.load_dataset(dataset)

    readings = pd.DataFrame(
        grid_search(
            X_train,
            y_train,
            MLPClassifier,
            hidden_layer_sizes=[
                (64,),
                (128,),
                (256,),
                (512,),
                (64, 32),
                (128, 64),
                (256, 128),
                (512, 256),
                (64, 32, 16),
                (128, 64, 32),
                (256, 128, 64),
                (512, 256, 128),
            ],
            early_stopping=[True],
        )
    )
    readings["n_layers"] = readings["hidden_layer_sizes"].map(len)
    readings["n_units"] = readings["hidden_layer_sizes"].map(lambda x: x[0])
    readings.to_csv(f"readings/mlp_size_{dataset}.csv", index=False)

    fig = plot(readings, "n_units", "Hidden Units", "n_layers", dataset)
    fig.savefig(f"readings/mlp_size_{dataset}.png")


if __name__ == "__main__":
    # neighbor_trials("credit_score")
    # neighbor_trials("term_deposits")

    size_trials("credit_score")
    size_trials("term_deposits")
