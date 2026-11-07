# title: DuckDB: SQL langsung ke CSV dan Parquet
# when: File besar. Hitung, saring, dan gabung tanpa memuat semua ke pandas.
# avoid: Mengambil seluruh kolom teks ke pandas lalu menganggap hemat RAM.
# tags: duckdb sql parquet csv cepat besar agregasi
import duckdb

def sql(query):
    """df = sql("SELECT label, COUNT(*) n FROM 'train.parquet' GROUP BY 1"). Sebut file langsung di FROM."""
    return duckdb.sql(query).df()

def profile_file(path, text="text"):
    """Baris, teks unik, kosong, dan rata-rata panjang tanpa memuat file."""
    p = str(path).replace("'", "''")
    return sql(f"""SELECT COUNT(*) AS baris, COUNT(DISTINCT "{text}") AS unik, SUM(CASE WHEN "{text}" IS NULL THEN 1 ELSE 0 END) AS kosong,
                   AVG(LENGTH("{text}")) AS panjang_rata FROM '{p}'""")
