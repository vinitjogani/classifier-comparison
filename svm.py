import numpy as np
from evaluate import run_trials
from sklearn.svm import SVC


class SoftmaxSVC(SVC):
    __module__ = SVC.__module__

    def predict_proba(self, X):
        logits = super().decision_function(X)
        if len(logits.shape) == 1 or logits.shape[1] == 1:
            p = 1 / (1 + np.exp(-logits))
            p = np.stack([1 - p, p], axis=-1)
        else:
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            p = exp / (exp.sum(axis=1, keepdims=True) + 1e-5)
        return p

    def decision_function(self, X):
        return self.predict_proba(X)


def kernel_trials(dataset):
    KERNELS = {
        "credit_score": ["linear", "rbf"],
        "term_deposits": ["linear", "poly", "rbf"],
    }
    C = {
        "credit_score": [0.3, 1, 3],
        "term_deposits": [0.1, 0.3, 1, 3, 10],
    }
    run_trials(
        SoftmaxSVC(),
        "kernel",
        dataset,
        model_args=dict(
            kernel=KERNELS[dataset],
            C=C[dataset],
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
