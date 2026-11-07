assert get_spans(["B-A", "I-A", "O", "B-B", "B-B"]) == [("A", 0, 2), ("B", 3, 4), ("B", 4, 5)]
r = span_f1([["B-A", "I-A", "O"]], [["B-A", "O", "O"]]); assert r["micro"]["f1"] == 0
assert span_f1([["B-A"]], [["B-A"]])["micro"]["f1"] == 1 and span_errors([["B-A", "I-A", "O"]], [["B-A", "O", "B-C"]]) == dict(tipe=0, batas=1, terlewat=0, berlebih=1)
