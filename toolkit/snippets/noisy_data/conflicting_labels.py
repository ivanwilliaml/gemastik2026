# title: Label bertentangan
# when: Teks sama dengan label berbeda. Voting mayoritas atau buang.
# avoid: Mengubah di fold validasi.
# tags: konflik label
import re
import pandas as pd

def conflicts(texts, y):
    """Kembalikan (tabel konflik, label mayoritas per baris, mask konflik)."""
    d = pd.DataFrame({"k": [re.sub(r"\W+", " ", str(t).lower()).strip() for t in texts], "y": list(y)})
    n = d.groupby("k").y.transform("nunique")
    maj = d.groupby("k").y.transform(lambda s: s.mode().iloc[0])
    print("baris konflik", round((n > 1).mean(), 4))
    return d[n > 1].sort_values("k"), maj.to_numpy(), (n > 1).to_numpy()
