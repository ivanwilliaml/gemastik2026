# title: Fitur graf pasangan
# when: Duplikat pertanyaan, bila cara sampling sama di test dan aturan mengizinkan. Laporkan bila dipakai.
# avoid: Mengeksploitasi sampling tanpa dilaporkan.
# tags: graf duplikat frekuensi tetangga
import re
import numpy as np
import pandas as pd
import networkx as nx

def graph_features(a, b, fit_a=None, fit_b=None):
    """Graf dibangun dari pasangan fit_* (default pasangan itu sendiri). Tidak memakai label."""
    key = lambda t: re.sub(r"\W+", " ", str(t).lower()).strip()
    G = nx.Graph()
    G.add_edges_from((key(x), key(y)) for x, y in zip(fit_a if fit_a is not None else a, fit_b if fit_b is not None else b))
    rows = []
    for x, y in zip(a, b):
        x, y = key(x), key(y)
        nx_, ny = set(G[x]) if x in G else set(), set(G[y]) if y in G else set()
        rows.append(dict(deg_a=len(nx_), deg_b=len(ny), common=len(nx_ & ny), union=len(nx_ | ny)))
    return pd.DataFrame(rows).astype("float32")
