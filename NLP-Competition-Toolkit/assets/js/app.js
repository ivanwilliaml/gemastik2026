// Router dan interaksi. Tidak ada jaringan: semua dimuat lewat tag script lokal.
(function () {
  const $ = id => document.getElementById(id);
  const store = {
    get(k, d) { try { const v = localStorage.getItem("nlptk:" + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem("nlptk:" + k, JSON.stringify(v)); } catch (e) { /* mode privat */ } },
    del(k) { try { localStorage.removeItem("nlptk:" + k); } catch (e) { /* mode privat */ } }
  };

  const NAV = [
    ["Mulai", [["home", "Beranda"], ["lomba", "Info Lomba"], ["decision", "Decision Guide"], ["pipeline", "Pipeline per Task"], ["cart", "Keranjang Kode"]]],
    ["Metode per tahap", TK.stages.map(s => ["stage/" + s.id, s.name])],
    ["Referensi", [["metrics", "Kamus Metrik"], ["models", "Model Pretrained"], ["tasks", "Task Explorer"], ["notebooks", "Notebook Library"], ["snippets", "Snippet Library"]]],
    ["Masalah", [["strategy", "Strategy & Troubleshooting"], ["diagnostics", "Diagnostics"]]],
    ["Domain", [["mbg", "Domain MBG"]]],
    ["Persiapan", [["hardware", "Hardware & Waktu"], ["readiness", "Versi & Offline"], ["library", "Library"], ["datasets", "Dataset"], ["coverage", "Progres"]]]
  ];

  function renderNav(active) {
    const count = id => {
      if (id === "cart") return (Filter.cart || []).length;
      if (id === "tasks") return TK.tasks.length;
      if (id === "notebooks") return (TK.library || {}).notebooks?.length ?? 0;
      if (id === "strategy") return TK.strategy.filter(s => TK.applies(TK.strategyFor[s.id] || "all", Filter.task)).length;
      if (id.startsWith("stage/")) return TK.methods.filter(m => m.stage === id.slice(6) && TK.applies(m.for, Filter.task) && !(Filter.noPre && m.pre)).length;
      return undefined;
    };
    $("nav").innerHTML = NAV.map(([g, items]) => `<h4>${g}</h4>` + items.map(([id, name]) => {
      const c = count(id);
      return `<a href="#/${id}" class="${id === active ? "active" : ""}"${id === active ? ' aria-current="page"' : ""}>${name}${c !== undefined ? `<span>${c}</span>` : ""}</a>`;
    }).join("")).join("");
  }

  // ---------- Filter global ----------
  function initFilter() {
    Filter.task = store.get("filterTask", "");
    if (Filter.task && !TK.tasks.some(t => t.id === Filter.task)) Filter.task = "";
    Filter.noPre = store.get("filterNoPre", false);
    Filter.prio1 = store.get("filterPrio1", false);
    Filter.cart = store.get("cart", []).filter(i => (TK.library.snippets || {})[i]);
    $("taskSel").innerHTML = '<option value="">Semua task</option>' + Object.keys(TK.taskGroups).map(g =>
      `<optgroup label="${R.esc(TK.taskGroups[g])}">${TK.tasks.filter(t => t.group === g).map(t => `<option value="${t.id}">${R.esc(t.name)}</option>`).join("")}</optgroup>`).join("");
    paintFilter();
    $("taskSel").addEventListener("change", () => setFilter($("taskSel").value));
    $("noPre").addEventListener("change", () => { Filter.noPre = $("noPre").checked; store.set("filterNoPre", Filter.noPre); paintFilter(); route(true); });
    $("clearF").addEventListener("click", clearFilter);
  }
  function paintFilter() {
    $("taskSel").value = Filter.task;
    $("noPre").checked = Filter.noPre;
    $("clearF").hidden = !Filter.task && !Filter.noPre;
    document.body.classList.toggle("filtered", !!(Filter.task || Filter.noPre));
  }
  function setFilter(id) {
    Filter.task = id; store.set("filterTask", id); paintFilter(); route(true);
  }
  function clearFilter() {
    Filter.task = ""; Filter.noPre = false; store.set("filterTask", ""); store.set("filterNoPre", false); paintFilter(); route(true);
  }

  const OLD = { preprocessing: "stage/prep", features: "stage/feat", postprocessing: "stage/post" };
  function route(keepScroll) {
    const y = window.scrollY;
    let parts = (location.hash.replace(/^#\/?/, "") || "home").split("/").map(decodeURIComponent);
    if (OLD[parts[0]]) parts = OLD[parts[0]].split("/");
    const page = Pages[parts[0]] ? parts[0] : "home";
    renderNav(page === "stage" ? "stage/" + (parts[1] || TK.stages[0].id) : page);
    $("content").innerHTML = Pages[page](parts[1], parts[2]);
    $("sidebar").classList.remove("open");
    $("menuBtn").setAttribute("aria-expanded", "false");
    document.title = (NAV.flatMap(g => g[1]).find(x => x[0] === (page === "stage" ? "stage/" + (parts[1] || TK.stages[0].id) : page)) || [, "Toolkit"])[1] + " · NLP Toolkit";
    afterRender(page, parts);
    const focus = parts.length > 2 ? parts[1] + "-" + parts[2] : parts[1];
    const el = focus && document.getElementById("a-" + focus);
    if (keepScroll === true) window.scrollTo(0, y);
    else if (el) { el.scrollIntoView({ block: "center" }); el.classList.add("flash"); }
    else window.scrollTo(0, 0);
  }

  function afterRender(page, parts) {
    document.querySelectorAll("[data-check]").forEach(c => {
      c.checked = store.get("check:" + c.dataset.check, false);
      c.addEventListener("change", () => store.set("check:" + c.dataset.check, c.checked));
    });
    if (page === "decision") initDecision();
    if (page === "tasks" && !parts[1]) initTaskFilters();
    if (page === "notebooks") initNotebook(parts[1]);
    if (page === "datasets") initDatasetFilters();
    const p1 = $("prio1");
    if (p1) p1.addEventListener("change", () => { Filter.prio1 = p1.checked; store.set("filterPrio1", p1.checked); route(true); });
    paintTimer();
  }

  // ---------- Filter ----------
  function initTaskFilters() {
    const box = $("taskFilters"); if (!box) return;
    box.addEventListener("click", e => {
      const b = e.target.closest("[data-f]"); if (!b) return;
      box.querySelectorAll(".chip").forEach(c => c.classList.toggle("on", c === b));
      const f = b.dataset.f;
      document.querySelectorAll("section[data-group]").forEach(s => {
        s.hidden = !(f === "all" || f === "prio" || s.dataset.group === f);
        s.querySelectorAll(".card").forEach(c => { c.hidden = f === "prio" && c.dataset.prio !== "S"; });
        if (f === "prio") s.hidden = !s.querySelector('.card[data-prio="S"]');
      });
    });
  }
  function initDatasetFilters() {
    const box = $("dsFilters"); if (!box) return;
    box.addEventListener("click", e => {
      const b = e.target.closest("[data-l]"); if (!b) return;
      box.querySelectorAll(".chip").forEach(c => c.classList.toggle("on", c === b));
      const l = b.dataset.l;
      document.querySelectorAll("#dsTable tr[data-l]").forEach(r => {
        r.hidden = l === "local" ? r.dataset.local !== "1" : !!l && r.dataset.l !== l;
      });
    });
  }

  // ---------- Decision wizard + pohon ----------
  function collect(node, depth, out, parent, leaves) {
    const me = { node, depth, parent, y: 0 };
    out.push(me);
    if (node.opts) { me.kids = node.opts.map(o => { const k = collect(o.to, depth + 1, out, me, leaves); k.answer = o.a; return k; }); me.y = (me.kids[0].y + me.kids[me.kids.length - 1].y) / 2; }
    else { me.y = leaves.n++; }
    return me;
  }
  function drawTree(path) {
    const list = [], leaves = { n: 0 }, ROW = 50, W = 196, GAP = 70, H = 42;
    collect(TK.decisionTree, 0, list, null, leaves);
    const X = d => d * (W + GAP) + 8, Y = y => y * ROW + 8;
    const on = n => path.includes(n.node);
    const wrap = (t, max) => { const w = String(t).split(" "), l = [""]; w.forEach(x => { if ((l[l.length - 1] + " " + x).trim().length > max) l.push(x); else l[l.length - 1] = (l[l.length - 1] + " " + x).trim(); }); return l.slice(0, 2); };
    let s = "";
    list.forEach(n => { if (n.parent) {
      const x1 = X(n.parent.depth) + W, y1 = Y(n.parent.y) + H / 2, x2 = X(n.depth), y2 = Y(n.y) + H / 2, mx = (x1 + x2) / 2;
      s += `<path class="t-link ${on(n) && on(n.parent) ? "on" : ""}" d="M${x1} ${y1}C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}"/>`;
    }});
    list.forEach(n => {
      const leaf = !!n.node.leaf;
      const main = leaf ? n.node.title : n.node.q;
      const lines = n.answer ? [["a", n.answer], ["b", main]] : wrap(main, 26).map(t => ["b", t]);
      const texts = lines.map((l, i) => `<text class="${l[0]}" x="${X(n.depth) + 9}" y="${Y(n.y) + (lines.length === 1 ? 26 : 17 + i * 16)}">${R.esc(l[1].length > 31 ? l[1].slice(0, 30) + "…" : l[1])}</text>`).join("");
      s += `<g class="t-node ${leaf ? "leaf" : ""} ${on(n) ? "on" : ""}" ${leaf ? `data-task="${n.node.task}" role="link" tabindex="0" aria-label="${R.esc(main)}"` : ""}><rect x="${X(n.depth)}" y="${Y(n.y)}" width="${W}" height="${H}" rx="7"/>${texts}</g>`;
    });
    const depth = Math.max(...list.map(n => n.depth)) + 1;
    $("tree").innerHTML = `<svg viewBox="0 0 ${depth * (W + GAP) - GAP + 16} ${leaves.n * ROW + 8}" role="img" aria-label="Pohon keputusan pemilihan task NLP">${s}</svg>`;
  }
  function initDecision() {
    let path = [TK.decisionTree];
    const paint = () => {
      const cur = path[path.length - 1];
      drawTree(path);
      $("wres").innerHTML = "";
      if (cur.leaf) {
        $("wq").textContent = "Hasil: " + cur.title;
        $("wopts").innerHTML = "";
        $("wres").innerHTML = `<dl class="kv" style="margin:.2rem 0 .7rem"><dt>Baseline</dt><dd>${R.inline(cur.base)}</dd><dt>Metrik umum</dt><dd>${R.esc(cur.metric)}</dd></dl><div class="row"><button class="btn primary" data-set-task="${cur.task}" data-go="#/pipeline">Jadikan filter & buka pipeline</button><a class="btn" href="#/tasks/${cur.task}">Detail task</a></div>`;
      } else {
        $("wq").textContent = cur.q;
        $("wopts").innerHTML = cur.opts.map((o, i) => `<button class="chip" data-i="${i}">${R.esc(o.a)}</button>`).join("");
      }
    };
    $("wopts").addEventListener("click", e => { const b = e.target.closest("[data-i]"); if (!b) return; path.push(path[path.length - 1].opts[+b.dataset.i].to); paint(); });
    $("wreset").addEventListener("click", () => { path = [TK.decisionTree]; paint(); });
    const open = e => { const g = e.target.closest(".t-node.leaf"); if (g) { setFilter(g.dataset.task); location.hash = "#/pipeline"; } };
    $("tree").addEventListener("click", open);
    $("tree").addEventListener("keydown", e => { if (e.key === "Enter") open(e); });
    paint();
  }

  // ---------- Notebook ----------
  const nbCache = {};
  window.TK_NB = (id, nb) => { nbCache[id] = nb; };
  function loadNb(id) {
    if (nbCache[id]) return Promise.resolve(nbCache[id]);
    const meta = TK.library.notebooks.find(n => n.id === id);
    return new Promise((ok, no) => {
      const s = document.createElement("script");
      s.src = meta.script;
      s.onload = () => ok(nbCache[id]);
      s.onerror = () => no(new Error("Gagal memuat " + meta.script));
      document.head.appendChild(s);
    });
  }
  function nbView(nb) {
    const { html, toc } = R.notebook(nb);
    const t = toc.length > 1 ? `<nav class="nb-toc" aria-label="Daftar isi notebook">${toc.map(x => `<a class="l${x.level}" href="javascript:void(0)" data-jump="${R.esc(x.id)}">${R.esc(x.text)}</a>`).join("")}</nav>` : "";
    return `<div class="nb-layout" style="${t ? "" : "grid-template-columns:1fr"}"><div class="nb-body" style="min-width:0">${html}</div>${t}</div>`;
  }
  function download(name, text) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
    a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 3000);
  }
  function initNotebook(id) {
    const file = $("nbFile");
    if (file) file.addEventListener("change", () => {
      const f = file.files[0]; if (!f) return;
      const r = new FileReader();
      r.onload = () => {
        try { $("nbFileView").innerHTML = `<h2>${R.esc(f.name)}</h2>` + nbView(JSON.parse(r.result)); }
        catch (e) { $("nbFileView").innerHTML = `<p class="badge bad">Bukan file notebook yang valid.</p>`; }
      };
      r.readAsText(f);
    });
    const tags = $("nbTags");
    if (tags) tags.addEventListener("click", e => {
      const b = e.target.closest("[data-t]"); if (!b) return;
      tags.querySelectorAll(".chip").forEach(c => c.classList.toggle("on", c === b));
      document.querySelectorAll("[data-tags]").forEach(c => { c.hidden = !!b.dataset.t && !c.dataset.tags.split("|").includes(b.dataset.t); });
    });
    const view = $("nbView");
    if (view) loadNb(id).then(nb => { view.innerHTML = nbView(nb); }).catch(e => { view.innerHTML = `<p class="badge bad">${R.esc(e.message)}. Jalankan ulang tools/build_library.py.</p>`; });
  }

  // ---------- Timer lomba ----------
  const TOTAL = 300 * 60 * 1000;
  function paintTimer() {
    const start = store.get("timerStart", null), box = $("timer");
    const elapsed = start ? Date.now() - start : null;
    let phase = -1;
    if (elapsed === null) box.innerHTML = `<button data-timer="start" title="Mulai hitung mundur 5 jam">Mulai 5 jam</button>`;
    else {
      const left = Math.max(0, TOTAL - elapsed), h = Math.floor(left / 3600000), m = Math.floor(left / 60000) % 60, s = Math.floor(left / 1000) % 60, min = elapsed / 60000;
      phase = TK.timePlan.findIndex(p => min >= p.from && min < p.to);
      const name = phase >= 0 ? TK.timePlan[phase].name : "Selesai";
      const txt = `<span class="phase">${R.esc(name)}</span> <b>${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}</b> <button data-timer="reset" title="Atur ulang timer">↺</button>`;
      if (box.dataset.txt !== txt) { box.innerHTML = txt; box.dataset.txt = txt; }
    }
    if (elapsed === null) box.dataset.txt = "";
    document.querySelectorAll("#timeline [data-phase]").forEach(d => d.classList.toggle("now", +d.dataset.phase === phase));
  }

  // ---------- Salin ----------
  function toast(msg) {
    const t = $("toast"); t.textContent = msg; t.classList.add("show");
    clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.remove("show"), 1400);
  }
  function copyText(text) {
    const legacy = () => {
      const ta = document.createElement("textarea");
      ta.value = text; ta.style.cssText = "position:fixed;opacity:0";
      document.body.appendChild(ta); ta.select();
      let ok = false; try { ok = document.execCommand("copy"); } catch (e) { /* ditangani di bawah */ }
      ta.remove();
      toast(ok ? "Tersalin" : "Gagal menyalin, pilih teks secara manual");
    };
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(() => toast("Tersalin"), legacy);
    else legacy();
  }

  document.addEventListener("click", e => {
    const st = e.target.closest("[data-set-task]");
    if (st) {
      e.preventDefault();
      setFilter(st.dataset.setTask);
      if (st.dataset.go) location.hash = st.dataset.go; else toast("Filter: " + st.textContent.trim());
      return;
    }
    if (e.target.closest("[data-clear-filter]")) { clearFilter(); return; }
    const ca = e.target.closest("[data-cart]");
    if (ca) {
      e.preventDefault();
      const id = ca.dataset.cart, i = Filter.cart.indexOf(id);
      if (i >= 0) Filter.cart.splice(i, 1); else Filter.cart.push(id);
      store.set("cart", Filter.cart);
      toast(i >= 0 ? "Dihapus dari keranjang" : "Masuk keranjang");
      const open = [...document.querySelectorAll("details[open]")].map(d => d.closest("[id]")?.id);
      route(true);
      open.forEach(x => { const el = x && document.getElementById(x); if (el) el.querySelectorAll("details").forEach(d => { d.open = true; }); });
      return;
    }
    if (e.target.closest("[data-cart-checked]")) {
      const ids = [...document.querySelectorAll("[data-slot]:checked")].map(x => x.dataset.slot).filter(x => x && TK.library.snippets[x]);
      ids.forEach(x => { if (!Filter.cart.includes(x)) Filter.cart.push(x); });
      store.set("cart", Filter.cart); toast(ids.length + " snippet masuk keranjang"); location.hash = "#/cart"; return;
    }
    const cc = e.target.closest("[data-copy-chain]");
    if (cc) { copyText(Code.assemble([cc.dataset.copyChain])); return; }
    const cs = e.target.closest("[data-copy-stage]");
    if (cs) { e.preventDefault(); e.stopPropagation(); copyText(Code.pipelineOf(Filter.task).stages[+cs.dataset.copyStage].code); return; }
    if (e.target.closest("[data-copy-all]")) { copyText(Code.pipelineCode(Code.pipelineOf(Filter.task))); return; }
    if (e.target.closest("[data-dl-nb]")) { const p = Code.pipelineOf(Filter.task); download(Filter.task + ".ipynb", Code.notebookJson(p)); return; }
    if (e.target.closest("[data-open-all]")) { document.querySelectorAll(".stagecell").forEach(d => { d.open = true; }); return; }
    if (e.target.closest("[data-close-all]")) { document.querySelectorAll(".stagecell").forEach(d => { d.open = false; }); return; }
    if (e.target.closest("[data-copy-cart]")) { copyText(Code.assemble(Filter.cart)); return; }
    if (e.target.closest("[data-dl-cart]")) { download("toolkit.py", Code.assemble(Filter.cart)); return; }
    if (e.target.closest("[data-clear-cart]")) { Filter.cart = []; store.set("cart", []); route(true); return; }
    const c = e.target.closest("[data-copy]");
    if (c) { copyText(c.parentElement.querySelector("[data-raw]").textContent); return; }
    const ct = e.target.closest("[data-copy-text]");
    if (ct) { copyText(ct.dataset.copyText); return; }
    const tm = e.target.closest("[data-timer]");
    if (tm) {
      if (tm.dataset.timer === "start") { store.set("timerStart", Date.now()); toast("Timer dimulai"); }
      else if (confirm("Atur ulang timer lomba?")) store.del("timerStart");
      paintTimer(); return;
    }
    const j = e.target.closest("[data-jump]");
    if (j) { e.preventDefault(); const el = document.getElementById(j.dataset.jump); if (el) el.scrollIntoView({ behavior: "auto", block: "start" }); return; }
    if (e.target.closest("[data-nb-copy]")) {
      const code = Array.from(document.querySelectorAll(".nb-body [data-raw]")).map(x => x.textContent).join("\n\n");
      copyText(code); return;
    }
    const dl = e.target.closest("[data-nb-dl]");
    if (dl) {
      const meta = TK.library.notebooks.find(n => n.id === dl.dataset.nbDl);
      loadNb(meta.id).then(nb => download(meta.file, JSON.stringify(nb, null, 1)));
    }
  });

  $("menuBtn").addEventListener("click", () => {
    const open = $("sidebar").classList.toggle("open");
    $("menuBtn").setAttribute("aria-expanded", open);
  });
  document.addEventListener("click", e => {
    if (!e.target.closest("#sidebar") && !e.target.closest("#menuBtn")) $("sidebar").classList.remove("open");
  });

  // ---------- Tema ----------
  const saved = store.get("theme", null);
  if (saved) document.documentElement.dataset.theme = saved;
  $("themeBtn").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    store.set("theme", next);
  });

  initFilter();
  Search.init();
  window.addEventListener("hashchange", () => route());
  setInterval(paintTimer, 1000);
  route();
})();
