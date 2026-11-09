from sklearn.model_selection import GroupShuffleSplit

def connected_groups(frame, text_cols, group_col=None, normalized_cols=(), shared_endpoints=False):
    """Gabung pengguna, duplikat native, dan collision normalisasi sebelum split."""
    parent = np.arange(len(frame))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i,(_,row) in enumerate(frame.iterrows()):
        keys = [(('endpoint' if shared_endpoints else col), safe_text(row[col]).casefold()) for col in text_cols]
        keys += [('endpoint' if shared_endpoints else 'normalized_'+col, str(row[col]).casefold()) for col in normalized_cols]
        if group_col and pd.notna(row[group_col]):
            keys.append(('user',str(row[group_col])))
        for key in keys:
            if not key[1]:
                continue
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    return np.array([root(i) for i in range(len(frame))])

