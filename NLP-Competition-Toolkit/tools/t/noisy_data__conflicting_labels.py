tab, maj, mask = conflicts(["A b", "a b", "a  B!", "c"], [0, 1, 1, 0]); assert maj[0] == 1 and mask.tolist() == [True, True, True, False]
