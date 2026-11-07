class _M(torch.nn.Module):
    def __init__(s): super().__init__(); s.layer = torch.nn.Linear(2, 2)
    def forward(s, x, labels):
        class O: pass
        o = O(); o.loss = torch.nn.functional.cross_entropy(s.layer(x), labels); return o
m_ = _M(); opt = torch.optim.AdamW(m_.parameters(), lr=.1)
L = [{"x": torch.eye(2), "labels": torch.tensor([0, 1])}] * 6
a = train_loop(m_, L, opt, accum=2, device="cpu"); b = train_loop(m_, L, opt, accum=2, device="cpu"); assert b < a
assert len(llrd_groups(m_)) == 2
