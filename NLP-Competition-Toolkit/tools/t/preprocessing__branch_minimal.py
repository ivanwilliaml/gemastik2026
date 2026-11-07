assert clean_minimal(" Halo\n  DUNIA ") == "halo dunia"
assert clean_minimal("ＡＢ", lower=False) == "AB"
