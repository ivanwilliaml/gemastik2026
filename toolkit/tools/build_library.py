"""Bangun ulang data aplikasi dari folder notebooks/, snippets/, dan datasets/catalog.json.

Pakai:  python tools/build_library.py
Hanya memakai pustaka standar Python.
"""
import ast
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipelines_lib import all_pipelines  # noqa: E402
from build_cheatsheet import build as build_cheatsheet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "assets" / "js" / "generated"
NB_OUT = GEN / "nb"
META_KEYS = ("title", "when", "avoid", "tags", "deps", "needs")


def read_tasks():
    text = (ROOT / "assets/js/content/tasks.js").read_text(encoding="utf-8")
    return re.findall(r'\{ id: "([a-z_]+)", group:', text)


def read_slots():
    text = (ROOT / "assets/js/content/slots.js").read_text(encoding="utf-8")
    return re.findall(r'^\s*"([a-z_]+/[a-z0-9_]+)":', text, flags=re.M)


def src(v):
    return "".join(v) if isinstance(v, list) else (v or "")


def parse_notebook(path, tasks, manifest):
    rel = path.relative_to(ROOT).as_posix()
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    code = [c for c in cells if c.get("cell_type") == "code" and src(c.get("source")).strip()]
    extra = {**nb.get("metadata", {}).get("toolkit", {}), **manifest.get(rel, {})}

    prefix = path.stem.split("__")[0]
    linked = extra.get("tasks") or ([prefix] if prefix in tasks else [])
    bad = [t for t in linked if t not in tasks]

    title = extra.get("title")
    if not title:
        for c in cells:
            if c.get("cell_type") == "markdown":
                m = re.search(r"^#\s+(.+)$", src(c.get("source")), flags=re.M)
                if m:
                    title = m.group(1).strip()
                    break
    title = title or path.stem.replace("__", " · ").replace("_", " ")

    errors = sum(any(o.get("output_type") == "error" for o in c.get("outputs", [])) for c in code)
    text = " ".join(src(c.get("source")) for c in cells if c.get("cell_type") in ("markdown", "code"))
    nb_id = re.sub(r"[^A-Za-z0-9_-]+", "-", path.relative_to(ROOT / "notebooks").with_suffix("").as_posix())
    # Nama pendek agar aman dari batas panjang path Windows.
    script = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12] + ".js"
    meta = {
        "id": nb_id, "path": rel, "file": path.name, "script": f"assets/js/generated/nb/{script}",
        "title": title, "tasks": linked, "tags": extra.get("tags", []), "when": extra.get("when", ""),
        "nCode": len(code), "nExecuted": sum(c.get("execution_count") is not None for c in code),
        "nErrors": errors, "bytes": path.stat().st_size, "searchText": text[:60000],
    }
    return meta, nb, bad


def parse_snippet(path):
    rel = path.relative_to(ROOT / "snippets").with_suffix("").as_posix()
    code = path.read_text(encoding="utf-8")
    info = {}
    body = []
    header = True
    for line in code.splitlines():
        m = re.match(r"#\s*(\w+):\s*(.*)", line) if header else None
        if m and m.group(1) in META_KEYS:
            info[m.group(1)] = m.group(2).strip()
        else:
            header = False
            body.append(line)
    try:
        ast.parse(code)
        valid = True
    except SyntaxError as e:
        valid = False
        print(f"  ! Sintaks tidak valid: {rel}: baris {e.lineno}: {e.msg}")
    return rel, {
        "path": f"snippets/{rel}.py", "title": info.get("title", ""), "when": info.get("when", ""),
        "avoid": info.get("avoid", ""), "tags": info.get("tags", ""), "needs": [x.strip() for x in info.get("needs", "").split(",") if x.strip()],
        "code": "\n".join(body).strip() + "\n", "valid": valid,
    }


def js_assign(name, value):
    return f"window.TK = window.TK || {{}};\nTK.{name} = {json.dumps(value, ensure_ascii=False, indent=0)};\n"


def build_catalog():
    data = json.loads((ROOT / "datasets/catalog.json").read_text(encoding="utf-8"))
    for c in data["datasets"]:
        c["local"] = any(p.is_file() for p in (ROOT / "datasets" / c["key"]).glob("*"))
    (GEN / "catalog.js").write_text(
        js_assign("catalog", data["datasets"]) + f"TK.catalogNote = {json.dumps(data.get('note', ''), ensure_ascii=False)};\n",
        encoding="utf-8",
    )
    lines = ["# Katalog Dataset", "", data.get("note", ""), "", "| Key | Dataset | Baris | Kelas | Lisensi | Sumber | Karakter |", "|---|---|---|---|---|---|---|"]
    for c in data["datasets"]:
        chars = c["chars"].replace("|", "/")
        lines.append(f"| `{c['key']}` | {c['name']} | {c['rows']} | {c.get('classes') or '–'} | {c['license']} | [{c['source']}]({c['url']}) | {chars} |")
    (ROOT / "datasets/CATALOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data["datasets"]


def build_bundle():
    """Versi paket bundle dari references/bundle_freeze.txt (hasil pip freeze)."""
    path = ROOT / "references/bundle_freeze.txt"
    pkgs = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "==" in line:
                name, ver = line.split("==", 1)
                pkgs[name.strip().lower().replace("_", "-")] = ver.strip()
    (GEN / "bundle.js").write_text(js_assign("bundle", pkgs), encoding="utf-8")
    return len(pkgs)


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    build_cheatsheet()
    assert NB_OUT.resolve().is_relative_to(GEN.resolve()) and NB_OUT.resolve() != GEN.resolve()
    shutil.rmtree(NB_OUT, ignore_errors=True)
    NB_OUT.mkdir()
    tasks, slots = read_tasks(), read_slots()

    manifest_path = ROOT / "notebooks/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    notebooks = []
    for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        if 'modular' in path.parts: continue  # arsip rakitan lama, bukan jalur cheatsheet aktif
        if ".ipynb_checkpoints" in path.parts:
            continue
        try:
            meta, nb, bad = parse_notebook(path, tasks, manifest)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  ! Lewati {path.name}: bukan notebook valid ({e})")
            continue
        if bad:
            print(f"  ! {path.name}: task_id tidak dikenal: {bad}")
        if not meta["tasks"]:
            print(f"  ! {path.name}: belum terhubung ke task. Pakai nama <task_id>__x.ipynb atau manifest.json")
        (NB_OUT / Path(meta["script"]).name).write_text(f"TK_NB({json.dumps(meta['id'])}, {json.dumps(nb, ensure_ascii=False)});\n", encoding="utf-8")
        notebooks.append(meta)

    snippets = {}
    for path in sorted((ROOT / "snippets").rglob("*.py")):
        rel, snip = parse_snippet(path)
        snippets[rel] = snip

    catalog = build_catalog()
    build_bundle()
    library = {"builtAt": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "notebooks": notebooks, "snippets": snippets, "pipelines": all_pipelines()}
    result = ROOT / "references/snippet_test_result.json"
    if result.exists():
        library["tests"] = json.loads(result.read_text(encoding="utf-8"))
    pt, pb = ROOT / "references/pipeline_test.json", ROOT / "references/pipeline_test_bert.json"
    if pt.exists():
        rep, rb = json.loads(pt.read_text(encoding="utf-8")), json.loads(pb.read_text(encoding="utf-8")) if pb.exists() else {}
        library.setdefault("tests", {})["pipelines"] = {k: ("tanpa transformer" + (" dan dengan transformer" if rb.get(k, {}).get("ok") else "")) for k, v in rep.items() if v["ok"]}
    (GEN / "library.js").write_text(js_assign("library", library), encoding="utf-8")

    linked = {t for n in notebooks for t in n["tasks"]}
    print(f"Notebook : {len(notebooks)} (task terhubung {len(linked)} dari {len(tasks)})")
    print(f"Snippet  : {sum(s in snippets for s in slots)} dari {len(slots)} slot, ditambah {sum(s not in slots for s in snippets)} snippet lain")
    print(f"Pipeline : {len(library['pipelines'])}, mencakup {len({t for p in library['pipelines'].values() for t in p['tasks']})} task")
    print(f"Dataset  : {len(catalog)} di katalog, {sum(c['local'] for c in catalog)} tersimpan lokal")


if __name__ == "__main__":
    main()
