d_ = ["nasi goreng enak", "nasi goreng murah", "bola kaki seru", "bola basket seru"]
m_, per = npmi_coherence([["nasi", "goreng"], ["nasi", "bola"]], d_); assert per[0] > per[1]
assert topic_diversity([["a", "b"], ["a", "c"]]) == .75 and -1 <= cv_coherence([["nasi", "goreng"], ["bola", "seru"]], d_) <= 1
