class _N(nn.Module):
    def __init__(s, v, n): super().__init__(); s.e = nn.EmbeddingBag(v, 16, padding_idx=0); s.f = nn.Linear(16, n)
    def forward(s, x): return s.f(s.e(x))
assert NNTextClassifier(_N, epochs=1, device="cpu").fit(texts, y).predict_proba(texts_test).shape == (len(texts_test), 3)
