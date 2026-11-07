r = reg_report([1, 2, 3], [1, 2, 4]); assert r["mae"] == round(1 / 3, 4) and r["spearman"] == 1 and "rmsle" in r and "mape" in r
assert qwk([1, 2, 3], [1, 2, 3]) == 1
