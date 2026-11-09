save_run("a", np.zeros((2, 2)), np.ones(2), .5, {"x": 1}); assert load_run("a")["oof"].shape == (2, 2) and load_run("a")["score"] == .5 and leaderboard().iloc[0].run == "a"
