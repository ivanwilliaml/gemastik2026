M = batched_to_memmap(lambda t: np.array([[len(x), 1] for x in t]), ["aaa", "b", "cc"], 2, batch=2); assert M[:, 0].tolist() == [3, 1, 2] and np.load("emb.npy", mmap_mode="r").shape == (3, 2)
