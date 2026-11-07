// Pencarian lokal: indeks seluruh konten, notebook, dan snippet.
window.Search = (function () {
  const norm = s => String(s ?? "").toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
  let docs = null;

  function build() {
    const d = [], add = (kind, title, route, text, tags, forStr, pre) => d.push({ kind, title, route, text: text || "", tags: tags || "", forStr: forStr || "", pre: !!pre, nt: norm(title), nb: norm(text), ntag: norm(tags) });
    const L = TK.library || { notebooks: [], snippets: {} };
    TK.tasks.forEach(t => add("Task", t.name, "#/tasks/" + t.id,
      [t.goal, t.baseline, t.examples.join(". "), t.traits.join(". "), t.advanced.join(". "), t.pitfalls.join(". "), t.metrics.join(", ")].join(" "), t.id + " " + TK.taskGroups[t.group], t.id));
    TK.lomba.rules.forEach(r => add("Lomba", r.replace(/\[[^\]]+\]\s*|\*\*/g, "").slice(0, 70), "#/lomba", r));
    TK.lomba.questions.forEach(r => add("Lomba", r.slice(0, 70), "#/lomba", r, "panitia pertanyaan"));
    TK.lomba.dont.forEach(r => add("Jangan", r[0], "#/lomba", r[1]));
    TK.mbg.lexicon.forEach(l => add("Leksikon MBG", l[0], "#/mbg/" + R.slug(l[0]), l[1], "aspek mbg"));
    TK.mbg.events.forEach(e => add("Peristiwa MBG", e[1].slice(0, 70), "#/mbg", e[0] + " " + e[1]));
    TK.strategy.forEach(s => {
      add("Strategi", s.title, "#/strategy/" + s.id, [s.signal, s.first, s.avoid.join(" ")].join(" "), "", TK.strategyFor[s.id]);
      s.recs.forEach(r => add("Strategi", r.t, "#/strategy/" + s.id, r.why + " · " + s.title, r.slot, TK.slotFor[r.slot]));
    });
    TK.diagnostics.forEach(g => g.items.forEach(i => add("Diagnosis", i.n, "#/diagnostics/" + R.slug(i.n), [i.read, i.act, i.leak || ""].join(" "), g.title, TK.diagFor[g.id])));
    TK.methods.forEach(m => add("Metode", m.n.replace(/`/g, ""), "#/stage/" + m.stage + "/" + m.id.slice(m.stage.length + 1), [m.what, m.when, m.avoid, m.imp, m.api].join(" "), m.stage + " " + m.group + " " + (m.lib || ""), m.for, m.pre));
    TK.bundleModels.forEach(m => add("Pretrained", m[2], "#/models", m[3] + " " + m[4], m[0], m[5], true));
    TK.metrics.forEach(g => g.items.forEach(m => add("Metrik", m.n, "#/metrics/" + R.slug(m.n), [m.f, m.when, m.trap].join(" "), g.group, m.for || g.for)));
    TK.hardware.profiles.forEach(p => add("Hardware", p.n, "#/hardware", p.rec.join(". ") + " " + p.avoid));
    TK.hardware.transformer.forEach(r => add("Hardware", r[0], "#/hardware", r[1] + " " + r[2], "transformer"));
    TK.versionTraps.forEach(v => add("Versi", v[0] + ": " + v[1].replace(/`/g, ""), "#/readiness", v[2] + " " + v[3]));
    TK.readiness.forEach(g => g.items.forEach(i => add("Offline", i.replace(/`/g, "").slice(0, 70), "#/readiness", i, g.group)));
    TK.lessons.forEach(l => add("Pelajaran", l.t, "#/decision", l.d, l.tag));
    (TK.catalog || []).forEach(c => add("Dataset", c.name, "#/datasets/" + c.key, [c.source, c.chars].join(" "), (c.tasks || []).join(" ") + " " + c.language, (c.tasks || []).join(" ")));
    Object.keys(TK.slots).forEach(id => { if (!L.snippets[id]) add("Slot kosong", id, "#/snippets/" + id, TK.slots[id], "", TK.slotFor[id]); });
    Object.entries(L.snippets).forEach(([id, s]) => add("Snippet", s.title || id, "#/snippets/" + id, [s.when, s.code].join(" "), id + " " + (s.tags || ""), TK.slotFor[id]));
    L.notebooks.forEach(n => add("Notebook", n.title, "#/notebooks/" + n.id, [n.when, n.searchText].join(" "), (n.tasks || []).join(" ") + " " + (n.tags || []).join(" "), (n.tasks || []).join(" ")));
    return d;
  }

  function query(q) {
    if (!docs) docs = build();
    const toks = norm(q).split(/\s+/).filter(Boolean);
    if (!toks.length) return [];
    const res = [];
    for (const d of docs) {
      if (Filter.noPre && d.pre) continue;
      if (Filter.task && d.forStr && !TK.applies(d.forStr, Filter.task)) continue;
      let score = 0, ok = true;
      for (const t of toks) {
        let s = 0;
        if (d.nt === t) s += 30;
        if (d.nt.startsWith(t)) s += 16;
        else if (new RegExp("(^|[^a-z0-9])" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).test(d.nt)) s += 12;
        else if (d.nt.includes(t)) s += 6;
        if (d.ntag.includes(t)) s += 4;
        if (d.nb.includes(t)) s += 2;
        if (!s) { ok = false; break; }
        score += s;
      }
      if (ok) res.push({ d, score });
    }
    return res.sort((a, b) => b.score - a.score).slice(0, 30).map(r => r.d);
  }

  function context(d, q) {
    const t = norm(q).split(/\s+/).filter(Boolean)[0] || "";
    const i = d.nb.indexOf(t);
    const from = i > 40 ? i - 40 : 0;
    const txt = d.text.slice(from, from + 140).replace(/\s+/g, " ");
    return (from ? "…" : "") + txt;
  }

  function mark(html, q) {
    norm(q).split(/\s+/).filter(t => t.length > 1).forEach(t => {
      html = html.replace(new RegExp("(" + R.esc(t).replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "<mark>$1</mark>");
    });
    return html;
  }

  function init() {
    const input = document.getElementById("q"), box = document.getElementById("results");
    let sel = 0, items = [];
    const close = () => { box.hidden = true; };
    const paint = () => Array.from(box.querySelectorAll(".result")).forEach((e, i) => {
      e.classList.toggle("sel", i === sel);
      if (i === sel) e.scrollIntoView({ block: "nearest" });
    });
    input.addEventListener("input", () => {
      const q = input.value.trim();
      if (!q) return close();
      items = query(q); sel = 0;
      box.innerHTML = items.length
        ? items.map(d => `<a class="result" href="${d.route}"><span class="kind">${R.esc(d.kind)}</span><b>${mark(R.esc(d.title), q)}</b><small>${mark(R.esc(context(d, q)), q)}</small></a>`).join("")
        : `<div class="no-result">Tidak ada hasil untuk “${R.esc(q)}”${Filter.task ? " pada filter task ini" : ""}.</div>`;
      box.hidden = false; paint();
    });
    input.addEventListener("keydown", e => {
      if (e.key === "ArrowDown") { sel = Math.min(sel + 1, items.length - 1); paint(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { sel = Math.max(sel - 1, 0); paint(); e.preventDefault(); }
      else if (e.key === "Enter" && items[sel]) { location.hash = items[sel].route; close(); input.blur(); }
      else if (e.key === "Escape") { close(); input.blur(); }
    });
    input.addEventListener("focus", () => { if (input.value.trim()) box.hidden = false; });
    document.addEventListener("click", e => { if (!e.target.closest(".search")) close(); else if (e.target.closest(".result")) close(); });
    document.addEventListener("keydown", e => {
      const typing = /INPUT|TEXTAREA/.test(document.activeElement.tagName);
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); input.focus(); input.select(); }
      else if (e.key === "/" && !typing) { e.preventDefault(); input.focus(); input.select(); }
    });
  }
  return { init, query, reset() { docs = null; } };
})();
