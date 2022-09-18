from sklearn.neighbors import KNeighborsClassifier

from evaluate import run_trials


def neighbor_trials(dataset):
    run_trials(
        KNeighborsClassifier(),
        "k",
        dataset,
        model_args=dict(
            n_neighbors=[128],
            weights=["uniform", "distance"],
        ),
        plot_args=dict(
            x="param_n_neighbors",
            xlabel="Neighbors (K)",
            grouping="param_weights",
        ),
    )


def metric_trials(dataset):
    NEIGHBORS = {
        "credit_score": 16,
        "term_deposits": 256,
    }

    run_trials(
        KNeighborsClassifier(weights="distance", n_neighbors=NEIGHBORS[dataset]),
        "metrics",
        dataset,
        model_args=dict(
            metric=["manhattan", "cosine", "l2"],
        ),
        plot_args=dict(
            x="param_metric",
            xlabel="Metric",
            grouping="param_weights",
        ),
        plotter="bar",
    )


if __name__ == "__main__":
    for dataset in ["credit_score", "term_deposits"]:
        neighbor_trials(dataset)
        metric_trials(dataset)
