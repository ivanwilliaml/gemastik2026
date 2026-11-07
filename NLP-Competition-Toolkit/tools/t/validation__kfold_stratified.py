assert len(folds_stratified(y, 3)) == 3 and len(folds_regression(np.arange(100.), 4)) == 4 and len(folds_random(20, 2)) == 2
