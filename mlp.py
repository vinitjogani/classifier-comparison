from sklearn.neural_network import MLPClassifier
from evaluate import run_trials


def clean_readings(readings):
    def tup(x):
        if isinstance(x, tuple):
            return x
        else:
            x = x.replace("(", "").replace(",)", "").replace(")", "")
            x = tuple(map(int, x.split(",")))
            return x

    readings["param_hidden_layer_sizes"] = readings["param_hidden_layer_sizes"].map(tup)
    readings["param_n_layers"] = readings["param_hidden_layer_sizes"].map(len)
    readings["param_n_units"] = readings["param_hidden_layer_sizes"].map(lambda x: x[0])


def size_trials(dataset):

    SIZES = {
        "credit_score": [
            (256,),
            (512,),
            (1024,),
            (2048,),
            (128, 64),
            (256, 128),
            (512, 256),
            (1024, 512),
            (2048, 512),
            (128, 64, 32),
            (512, 256, 128),
            (1024, 256, 256),
            (2048, 256, 256),
            (128, 64, 64, 32),
            (512, 256, 256, 128),
            (1024, 512, 256, 128),
            (2048, 512, 256, 128),
        ],
        "term_deposits": [
            (128,),
            (256,),
            (512,),
            (1024,),
            (128, 64),
            (256, 128),
            (512, 256),
            (1024, 512),
            (128, 64, 32),
            (512, 256, 128),
            (1024, 256, 256),
        ],
    }

    run_trials(
        MLPClassifier(early_stopping=True),
        "size",
        dataset,
        model_args=dict(
            hidden_layer_sizes=SIZES[dataset],
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
