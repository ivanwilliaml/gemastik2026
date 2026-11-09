c = "Program dimulai di Jakarta. Presiden makan nasi di Bandung pada Senin."
assert "Bandung" in best_sentence("Di mana presiden makan nasi?", c) and answer_tail("presiden makan", "Presiden makan nasi") == "nasi"
