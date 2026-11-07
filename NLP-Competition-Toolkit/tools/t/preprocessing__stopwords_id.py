assert remove_stopwords("makanan yang tidak enak") == "makanan tidak enak"
assert not (STOP_ID & NEGATION_ID)
