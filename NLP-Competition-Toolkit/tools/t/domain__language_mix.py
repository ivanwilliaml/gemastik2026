assert detect_local_language("aku ora ngerti piye kuwi") == "jav"
assert detect_local_language("teu acan aya atuh kumaha") == "sun"
assert detect_local_language("saya makan nasi di sekolah") == "ind"
s = language_report(["aku ora ngerti piye kuwi", "saya makan nasi", "teu acan aya atuh kumaha"], show=1)
assert (s != "ind").sum() == 2
