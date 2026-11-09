p_ = np.eye(3)[(y + 1) % 3] * .8 + .0667; assert len(worst_errors(texts, y, p_, 5)) == 5 and len(confused_pairs(y, p_.argmax(1), k=2)) == 2
assert len(random_errors(texts, y, p_.argmax(1), 3)) == 3 and "macro avg" in per_label(y, p_.argmax(1)).index
assert len(segment_scores(y, y, np.arange(len(y)) % 2)) == 2 and len(residual_table(np.arange(50.), np.arange(50.))) == 5
