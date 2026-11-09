# title: LR, LinearSVC, SGD, Ridge, Dummy
# when: Baseline semua klasifikasi. Mulai tanpa class weight, lalu decision layer.
# avoid: multi_class=, penalty= lama, dan loss='log'.
# tags: lr svm sgd baseline dummy majority ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC

def make_lr(C=4.0, class_weight=None):
    return LogisticRegression(C=C, max_iter=2000, solver="lbfgs", class_weight=class_weight)

def make_svc(C=.5, class_weight=None, calibrated=False):
    m = LinearSVC(C=C, dual="auto", max_iter=5000, class_weight=class_weight)
    return CalibratedClassifierCV(m, method="sigmoid", cv=3) if calibrated else m

def make_sgd(alpha=1e-6, class_weight=None, seed=42):
    return SGDClassifier(loss="log_loss", alpha=alpha, class_weight=class_weight, random_state=seed)

def make_ridge_clf(alpha=1.0):
    return RidgeClassifier(alpha=alpha)

def make_dummy(strategy="prior"):
    return DummyClassifier(strategy=strategy)
