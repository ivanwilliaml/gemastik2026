# title: BiGRU dengan attention pooling
# when: Bobot token bisa dibaca saat error analysis.
# avoid: Menyamakan attention dengan penjelasan.
# tags: attention bigru
# needs: modeling/nn_train
class AttnGRU(nn.Module):
    def __init__(self, vocab, n_out, dim=200, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.rnn = nn.GRU(dim, hidden, batch_first=True, bidirectional=True)
        self.att, self.out = nn.Linear(hidden * 2, 1), nn.Linear(hidden * 2, n_out)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        w = self.att(h).squeeze(-1).masked_fill(x == 0, -1e4).softmax(1)
        return self.out((h * w.unsqueeze(-1)).sum(1))

def make_attn_gru(**kw):
    return NNTextClassifier(AttnGRU, **kw)
