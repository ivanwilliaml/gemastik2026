if MODELS.get("nli"):
    S = zero_shot(MODELS["nli"], ["Anak keracunan makanan basi di sekolah"], ["keamanan pangan", "anggaran negara"], batch=2); assert S.shape == (1, 2) and abs(S.sum() - 1) < 1e-3
