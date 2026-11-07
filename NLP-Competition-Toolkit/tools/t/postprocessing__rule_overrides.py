p, n = lookup_train(["Halo!", "halo", "x", "x"], [1, 1, 0, 1], ["HALO", "x"], [0, 0]); assert n == 1 and p.tolist() == [1, 0]
assert default_empty(["", "a"], [1, 1], 9).tolist() == [9, 1] and apply_rule(["slot gacor"], [0], r"gacor", 1).tolist() == [1]
