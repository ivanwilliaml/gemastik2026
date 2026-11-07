o = clean_advanced("<b>Baguuuus</b> @andi cek http://a.id 😀 &amp; tidak enak", neg=True)
assert "HTTPURL" in o and "@USER" in o and "baguus" in o and "tidak_enak" in o and "<b>" not in o and "grinning" in o
assert "NUM" in clean_advanced("harga 1.500", number="NUM", lower=False)
