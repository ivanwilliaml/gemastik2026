assert cls_report([0, 0, 1, 1], pred=[0, 1, 1, 1])["macro_f1"] == round((2 / 3 + .8) / 2, 4)
r = cls_report([0, 1, 1, 0], np.array([[.9, .1], [.2, .8], [.6, .4], [.7, .3]])); assert r["roc_auc"] == 1 and r["acc"] == .75
assert "top2" in cls_report([0, 1, 2, 2], np.eye(3)[[0, 1, 2, 1]] * .7 + .1)
assert multilabel_report([[1, 0], [0, 1]], [[1, 0], [0, 0]])["hamming"] == .25
