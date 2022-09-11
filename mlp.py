import pandas as pd
from sklearn.neural_network import MLPClassifier

import datasets
from evaluate import run_trials


def size_trials(dataset):
    def clean_readings(readings):
        readings["param_n_layers"] = readings["hidden_layer_sizes"].map(len)
        readings["param_n_units"] = readings["hidden_layer_sizes"].map(lambda x: x[0])

    run_trials(
        MLPClassifier(early_stopping=True),
        "size",
        dataset,
        model_args=dict(
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
        ),
        plot_args=dict(
            x="param_n_units",
            xlabel="Hidden Units",
            grouping="param_n_layers",
        ),
        clean_readings=clean_readings,
    )


if __name__ == "__main__":
    size_trials("credit_score")
    size_trials("term_deposits")
