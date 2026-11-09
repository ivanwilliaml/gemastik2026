# title: Head + tail truncation
# when: Teks panjang terpotong. Rata-rata prediksi dua potongan.
# avoid: Teks pendek: tak ada informasi baru.
# tags: tta head tail panjang chunk
def head_tail(text, head=128, tail=128):
    """Potong berdasarkan kata: head kata pertama dan tail kata terakhir."""
    w = str(text).split()
    return " ".join(w) if len(w) <= head + tail else " ".join(w[:head] + w[-tail:])

def chunks(text, size=200, stride=150):
    """Potongan bertumpang tindih untuk teks sangat panjang. Rata-ratakan prediksi per dokumen."""
    w = str(text).split()
    return [" ".join(w[i:i + size]) for i in range(0, max(len(w) - size + stride, 1), stride)]

def tta_views(texts, head=128):
    """Dua pandangan: awal dan akhir teks. Prediksi keduanya lalu rata-rata."""
    a = [" ".join(str(t).split()[:head]) for t in texts]
    b = [" ".join(str(t).split()[-head:]) for t in texts]
    return a, b
