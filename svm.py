import numpy as np
from evaluate import run_trials
from sklearn.svm import SVC


class SoftmaxSVC(SVC):
    def predict_proba(self, X):
        logits = self.decision_function(X)
        exp = np.exp(logits)
        p = exp / exp.sum(axis=1, keepdims=True)
        return p


def kernel_trials(dataset):
    run_trials(
        SoftmaxSVC(),
        "kernel",
        dataset,
        model_args=dict(
            kernel=["linear", "poly", "rbf"],
            C=[0.1, 0.3, 1, 3, 10],
        ),
        plot_args=dict(
            x="param_C",
            xlabel="Regularization",
            grouping="param_kernel",
        ),
    )


if __name__ == "__main__":
    kernel_trials("credit_score")
    kernel_trials("term_deposits")
