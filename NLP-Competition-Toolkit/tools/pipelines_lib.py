"""Rakit pipelines/*.py menjadi sel per tahap. Dipakai build_library.py dan test_pipelines.py."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = re.compile(r"^#\s*(title|when|avoid|tags|needs|deps):.*\n", flags=re.M)


def snippet_body(slot):
    return META.sub("", (ROOT / "snippets" / (slot + ".py")).read_text(encoding="utf-8")).strip()


def needs_of(slot):
    m = re.search(r"^#\s*needs:\s*(.*)$", (ROOT / "snippets" / (slot + ".py")).read_text(encoding="utf-8"), flags=re.M)
    return [x.strip() for x in m.group(1).split(",") if x.strip()] if m else []


def add_chain(slot, seen):
    if slot in seen:
        return []
    out = []
    for n in needs_of(slot):
        out += add_chain(n, seen)
    seen.append(slot)
    return out + [slot]


def parse(path):
    """Kembalikan dict title, tasks, stages. Fungsi yang sudah muncul di sel sebelumnya tidak diulang."""
    text = path.read_text(encoding="utf-8")
    title = re.search(r"^#\s*title:\s*(.*)$", text, flags=re.M).group(1)
    tasks = re.search(r"^#\s*for:\s*(.*)$", text, flags=re.M).group(1).split()
    seen, stages = [], []
    for block in re.split(r"^# %% stage: ", text, flags=re.M)[1:]:
        head, _, rest = block.partition("\n")
        sid, _, name = head.partition(" | ")
        note = re.search(r"^# note:\s*(.*)$", rest, flags=re.M).group(1)
        uses = [x.strip() for x in re.search(r"^# uses:[ \t]*(.*)$", rest, flags=re.M).group(1).split(",") if x.strip()]
        glue = re.sub(r"^# (note|uses):.*\n?", "", rest, flags=re.M).strip()
        parts, used = [], []
        for u in uses:
            for s in add_chain(u, seen):
                parts.append(snippet_body(s))
                used.append(s)
        code = "\n\n".join(parts + ["# ===== jalankan =====\n" + glue if parts else glue])
        stages.append({"id": sid.strip(), "name": name.strip(), "note": note, "uses": used, "code": code + "\n"})
    return {"title": title, "tasks": tasks, "stages": stages}


def all_pipelines():
    return {p.stem: parse(p) for p in sorted((ROOT / "pipelines").glob("*.py"))}
