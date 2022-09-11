from sklearn.ensemble import HistGradientBoostingClassifier
from evaluate import run_trials


def pruning_trials(dataset):
    MAX_ITER = {
        "credit_score": [100, 200, 300, 400],
        "term_deposits": [25, 50, 100, 150],
    }

    MAX_LEAF_NODES = {
        "credit_score": [64, 128, 256, 512, 1024],
        "term_deposits": [8, 16, 32, 64, 128],
    }

    run_trials(
        HistGradientBoostingClassifier(
            l2_regularization=0.35,
            learning_rate=0.035,
            validation_fraction=None,
        ),
        "pruning",
        dataset,
        model_args=dict(
            max_leaf_nodes=MAX_LEAF_NODES[dataset],
            max_iter=MAX_ITER[dataset],
        ),
        plot_args=dict(
            x="param_max_leaf_nodes",
            xlabel="Max Leaf Nodes",
            grouping="param_max_iter",
        ),
    )


if __name__ == "__main__":
    pruning_trials("credit_score")
    pruning_trials("term_deposits")
