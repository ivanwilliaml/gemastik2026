def _f(b):
    if b > 8: raise torch.cuda.OutOfMemoryError("x")
    return b * 2
assert with_oom_retry(_f, 32) == (16, 8)
