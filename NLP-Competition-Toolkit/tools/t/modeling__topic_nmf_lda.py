for f in (fit_nmf, fit_lda):
    l, tw, W = f(texts, k=4, top=5, min_df=2); assert len(l) == len(texts) and len(tw) == 4 and len(tw[0]) == 5
