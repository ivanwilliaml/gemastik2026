assert most_frequent_tag(ner_s[:200], ner_t[:200])(ner_s[:1])[0][0] in {x for t in ner_t[:200] for x in t}
assert gazetteer_tag([["Di", "Kota", "Garut", "ada"]], ["kota garut"], "LOC") == [["O", "B-LOC", "I-LOC", "O"]]
assert len(fit_crf(ner_s[:60], ner_t[:60], iters=5)(ner_s[:2])) == 2
