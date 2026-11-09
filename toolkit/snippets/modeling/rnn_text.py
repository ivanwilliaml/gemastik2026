# title: LSTM, BiLSTM, GRU, BiGRU
# when: Konteks urutan tanpa pretrained.
# avoid: Berharap menang atas transformer.
# tags: lstm gru bilstm bigru rnn
# needs: modeling/nn_train
from functools import partial

class TextRNN(nn.Module):
    def __init__(self, vocab, n_out, kind="gru", bidirectional=True, dim=200, hidden=128, drop=.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(dim, hidden, batch_first=True, bidirectional=bidirectional)
        self.drop, self.out = nn.Dropout(drop), nn.Linear(hidden * (2 if bidirectional else 1), n_out)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        m = (x != 0).unsqueeze(-1)
        return self.out(self.drop((h * m).sum(1) / m.sum(1).clamp(min=1)))

def make_rnn(kind="gru", bidirectional=True, **kw):
    return NNTextClassifier(partial(TextRNN, kind=kind, bidirectional=bidirectional), **kw)
