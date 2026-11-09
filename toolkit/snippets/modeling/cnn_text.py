# title: TextCNN
# when: Pola lokal pada data besar, atau anggota ensemble yang beragam.
# tags: cnn neural
# needs: modeling/nn_train
class TextCNN(nn.Module):
    def __init__(self, vocab, n_out, dim=200, filters=128, sizes=(3, 4, 5), drop=.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.conv = nn.ModuleList(nn.Conv1d(dim, filters, k, padding=k // 2) for k in sizes)
        self.drop, self.out = nn.Dropout(drop), nn.Linear(filters * len(sizes), n_out)
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        return self.out(self.drop(torch.cat([c(e).relu().amax(2) for c in self.conv], 1)))

def make_textcnn(**kw):
    return NNTextClassifier(TextCNN, **kw)
