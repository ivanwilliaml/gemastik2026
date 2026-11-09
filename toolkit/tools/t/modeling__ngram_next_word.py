m = NGramLM().fit(["saya makan nasi goreng", "saya makan roti", "dia minum air"]); assert m.topk("saya makan", 2)[0] in {"nasi", "roti"} and "air" in m.topk("tidak ada", 9)
