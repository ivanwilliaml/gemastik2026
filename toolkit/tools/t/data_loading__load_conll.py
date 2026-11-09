open("c.txt", "w", encoding="utf-8").write("Budi\tB-PER\npergi\tO\n\nJakarta\tB-LOC\n")
s, t = read_conll("c.txt", sep="\t")
assert s == [["Budi", "pergi"], ["Jakarta"]] and t[1] == ["B-LOC"]
assert spans_to_bio([(0, 4), (5, 10), (11, 14)], [(5, 14, "X")]) == ["O", "B-X", "I-X"]
