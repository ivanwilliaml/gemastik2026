# title: Stopword Indonesia tanpa kata negasi
# when: Hanya untuk kata topik, word cloud, dan kata kunci.
# avoid: Sebelum classifier.
# tags: stopword indonesia negasi
STOP_ID = set("""ada adalah agar akan aku anda antara apa apakah atau bagi bahwa banyak beberapa begitu belum bila
bisa dalam dan dari dengan di dia dua hal hanya harus ia ini itu jadi jika juga kami kamu karena ke kepada ketika
kita lagi lain lebih maka masih mereka mungkin nya oleh pada para saat saja sama sampai sangat satu saya sebagai
sebuah sedang sehingga sekali selalu semua seperti setelah sudah supaya tapi telah tentang tersebut tetapi untuk
yaitu yang""".split())
NEGATION_ID = {"tidak", "tak", "bukan", "jangan", "belum", "kurang", "tanpa", "gak", "ga", "nggak"}
STOP_ID = STOP_ID - NEGATION_ID

def remove_stopwords(text, stop=STOP_ID):
    return " ".join(w for w in str(text).split() if w.lower() not in stop)
