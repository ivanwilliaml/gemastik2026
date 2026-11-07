# %% [markdown]
# **Stopword.** Daftar ini hanya dipakai untuk merangkum kata kunci topik, bukan untuk input model
# embedding. Tambahkan kata yang terlalu umum di kasus lu, misalnya `mbg` dan `makan`.

# %%
STOPWORDS = set(
    """
    ada adalah agar akan aku anda apa apakah atas atau bagaimana bagi bahkan bahwa banyak baru
    begitu belum benar berapa bersama bisa buat bukan cuma dalam dan dapat dari dengan di dia
    dong dulu gak ga gitu gue gua hanya harus hal ini itu iya jadi jangan juga jika kalau kalo
    kami kamu kan karena kata ke kita kok lagi lah lain lalu lebih lu lo masih mau maka mana
    memang mereka meski mungkin nah nanti ni nya oleh pada pak para pun punya saat saja sama
    sampai saya sebagai sebagainya sedang sudah sudah sih seperti siapa supaya tak tapi telah
    tentang terus tetapi tidak tuh udah untuk ya yaitu yang yg dgn dg krn utk tdk jg aja kalo
    aja nih deh kak bang bu ibu bapak sekarang semua setiap sangat tanpa ketika orang enggak
    quot amp mas mbak sih dah tau kayak kaya sampe gimana emang kenapa
    """.split()
)
