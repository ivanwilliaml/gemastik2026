assert overlap_rate(["Halo!", "b"], ["halo", "c"]) == .5
assert check_folds(["a", "a", "b", "c"], [(np.array([0, 2]), np.array([1, 3]))]) == [.5]
