from sklearn.ensemble import HistGradientBoostingClassifier
from evaluate import run_trials


def pruning_trials(dataset):
    MAX_ITER = {
        "credit_score": [100, 200, 300, 400],
        "term_deposits": [100, 200, 300, 400],
    }

    MAX_DEPTH = {
        "credit_score": [2, 4, 8, 12, 16, 20, 24],
        "term_deposits": [2, 4, 8, 12, 16],
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
            max_depth=MAX_DEPTH[dataset],
            max_iter=MAX_ITER[dataset],
        ),
        plot_args=dict(
            x="param_max_depth",
            xlabel="Max Depth",
            grouping="param_max_iter",
            log=False,
        ),
    )


if __name__ == "__main__":
    pruning_trials("credit_score")
    pruning_trials("term_deposits")
