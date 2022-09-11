from sklearn.tree import DecisionTreeClassifier
from evaluate import run_trials


def pruning_trials(dataset):
    run_trials(
        DecisionTreeClassifier(),
        "pruning",
        dataset,
        model_args=dict(
            min_samples_leaf=[1, 8, 16, 32, 64, 128, 256],
            max_depth=[None, 8, 16, 32, 64],
            criterion=["gini", "entropy"],
        ),
        plot_args=dict(
            x="param_min_samples_leaf",
            xlabel="Min Samples Leaf",
            grouping="param_criterion",
        ),
    )


if __name__ == "__main__":
    # neighbor_trials("credit_score")
    # neighbor_trials("term_deposits")

    pruning_trials("credit_score")
    pruning_trials("term_deposits")
