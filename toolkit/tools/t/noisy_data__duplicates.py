t_ = ["Halo!", "halo", "dunia"]; assert duplicate_report(t_).iloc[0] == 2 and dedup_index(t_).tolist() == [0, 2]
