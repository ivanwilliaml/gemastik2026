assert rouge_report(["a b c"], ["a b c"]) == {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0}
assert abs(rouge_n("a b", "a c") - .5) < 1e-9 and abs(rouge_l("a x b", "a b") - .8) < 1e-9
