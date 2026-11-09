assert marker_lang("aku ora iso") == "jv" and marker_lang("teu kunanaon atuh") == "su" and marker_lang("saya makan") == "lain"
m = make_lang_clf().fit(lang_t, lang_y); assert m.score(lang_t, lang_y) > .8
assert callable(fast_detect)
