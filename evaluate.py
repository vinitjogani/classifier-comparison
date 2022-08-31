import numpy as np
from tqdm import tqdm
from collections import defaultdict

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, auc, precision_recall_curve


def pr_auc_score(true, pred, average="macro"):
    aucs = []
    counts = []
    for class_ in range(pred.shape[1]):
        precision, recall, _ = precision_recall_curve(true == class_, pred[:, class_])
        aucs.append(auc(recall, precision))
        counts.append((true == class_).sum())

    if average == "macro":
        return sum(aucs) / len(aucs)
    else:
        return sum(np.array(aucs) * np.array(counts) / sum(counts))


def get_metrics(true, pred, threshold=0.5):
    label = pred.argmax(axis=1)

    accuracy = (true == label).mean()
    f1 = f1_score(true, pred.argmax(axis=1), average="macro")
    pr = pr_auc_score(true, pred, average="macro")

    return {
        "Accuracy": accuracy,
        "F1": f1,
        "AUPRC": pr,
    }


def create_model_fn(model_cls, kwargs):
    def model_fn(X_train, y_train, X_val, y_val):
        model = model_cls(**kwargs)
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)
        return get_metrics(y_val, pred)

    return model_fn


def cross_validate(X, y, model_fn, folds=5):
    summary = defaultdict(int)
    kfold = KFold(n_splits=folds, shuffle=True, random_state=0)
    for train, val in tqdm(kfold.split(X)):
        X_train, y_train = X[train], y.iloc[train]
        X_val, y_val = X[val], y.iloc[val]
        metrics = model_fn(X_train, y_train, X_val, y_val)
        for m in metrics:
            summary[m] += metrics[m]
    for m in summary:
        summary[m] /= folds
    return dict(summary)
