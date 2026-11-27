"""Binary metrics with class 1 positive; undefined precision/recall/F1 are zero."""
import numpy as np


def confusion_matrix(y_true, y_pred):
    a, b = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if a.size == 0 or a.shape != b.shape or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('Expected nonempty aligned binary labels')
    return np.bincount((2*a+b).astype(int), minlength=4).reshape(2, 2)


def classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return dict(accuracy=float((tp+tn)/(tp+tn+fp+fn)),
                precision=float(tp/(tp+fp)) if tp+fp else 0.,
                recall=float(tp/(tp+fn)) if tp+fn else 0.,
                f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)


def accuracy(y, pred): return classification_metrics(y, pred)['accuracy']
def precision(y, pred): return classification_metrics(y, pred)['precision']
def recall(y, pred): return classification_metrics(y, pred)['recall']
def f1(y, pred): return classification_metrics(y, pred)['f1']
