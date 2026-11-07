# title: SelfTrainingClassifier
# when: Baseline self-training. Label -1 untuk unlabeled.
# avoid: Estimator tanpa predict_proba.
# tags: self_training scikit
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

def make_self_training(threshold=.95, rounds=2, C=4.0):
    return SelfTrainingClassifier(LogisticRegression(C=C, max_iter=2000), threshold=threshold, max_iter=rounds)
