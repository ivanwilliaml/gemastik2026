# title: Inferensi dan embedding ke memmap
# when: 500 ribu baris. Tulis per batch, urut panjang, dan aman bila kernel mati.
# avoid: Lupa mengembalikan urutan asli.
# tags: memmap cache inference urut_panjang
import numpy as np

def batched_to_memmap(fn, texts, dim, path="emb.npy", batch=256, sort=True):
    """fn(list teks) -> array (b, dim). Baris terurut sesuai teks asli."""
    texts = [str(t) for t in texts]
    M = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(texts), dim))
    order = np.argsort([len(t) for t in texts]) if sort else np.arange(len(texts))
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        M[idx] = fn([texts[j] for j in idx])
    M.flush()
    return M
