// Tiap halaman: fungsi (sub) => string HTML. Pengaitan interaksi ada di app.js.
// Semua daftar dilewatkan ke Filter sehingga hanya item yang berlaku untuk task terpilih yang tampil.
window.Filter = { task: "", noPre: false, prio1: false };
// Rakit kode: snippet dengan dependensinya, tanpa duplikat, siap ditempel ke satu sel.
window.Code = (function () {
  const lib = () => (TK.library || {}).snippets || {};
  function chain(id, seen) {
    seen = seen || [];
    const s = lib()[id];
    if (!s || seen.includes(id)) return seen;
    (s.needs || []).forEach(n => chain(n, seen));
    seen.push(id);
    return seen;
  }
  function assemble(ids, tail) {
    const all = [];
    ids.forEach(i => chain(i, all));
    const head = ["import re", "import numpy as np", "import pandas as pd", ""].join("\n");
    const body = all.map(i => "# ===== " + i + " =====\n" + lib()[i].code.trim()).join("\n\n");
    return head + "\n" + body + "\n" + (tail ? "\n# ===== resep =====\n" + tail : "");
  }
  function pipelineOf(task) {
    const P = (TK.library || {}).pipelines || {};
    const k = Object.keys(P).find(n => P[n].tasks.includes(task));
    return k ? { key: k, ...P[k] } : null;
  }
  function pipelineCode(p) {
    return p.stages.map(s => "# ========== " + s.name + " ==========\n" + s.code).join("\n");
  }
  function notebookJson(p) {
    const lines = t => t.split("\n").map((l, i, a) => (i < a.length - 1 ? l + "\n" : l));
    const cells = [{ cell_type: "markdown", metadata: {}, source: ["# " + p.title] }];
    p.stages.forEach(s => {
      cells.push({ cell_type: "markdown", metadata: {}, source: ["## " + s.name + "\n\n" + s.note] });
      cells.push({ cell_type: "code", metadata: {}, execution_count: null, outputs: [], source: lines(s.code.trimEnd()) });
    });
    return JSON.stringify({ cells, metadata: { kernelspec: { display_name: "Python 3", language: "python", name: "python3" } }, nbformat: 4, nbformat_minor: 5 }, null, 1);
  }
  return { chain, assemble, pipelineOf, pipelineCode, notebookJson };
})();
window.Pages = (function () {
  const { esc, inline, slug, codebox } = R;
  const L = () => TK.library || { notebooks: [], snippets: {} };
  const list = (a, cls) => a && a.length ? `<ul${cls ? ` class="${cls}"` : ""}>${a.map(x => `<li>${inline(x)}</li>`).join("")}</ul>` : "";
  const prioLabel = { S: "Prioritas utama", A: "Prioritas tinggi", B: "Siapkan versi cepat", C: "Bila ada waktu" };
  const notebooksFor = id => L().notebooks.filter(n => (n.tasks || []).includes(id));
  const taskObj = id => TK.tasks.find(t => t.id === id);
  const taskName = id => (taskObj(id) || {}).name || id;
  const taskLink = id => `<a href="#/tasks/${id}">${esc(taskName(id))}</a>`;
  const fmtBytes = b => b == null ? "–" : b >= 1e9 ? (b / 1e9).toFixed(1).replace(".", ",") + " GB" : b >= 1e6 ? Math.round(b / 1e6) + " MB" : Math.max(1, Math.round(b / 1e3)) + " KB";
  const nbBadge = n => n.nErrors ? `<span class="badge bad">${n.nErrors} sel error</span>` : n.nCode && n.nExecuted === n.nCode ? `<span class="badge good">Output tersimpan, tanpa error</span>` : `<span class="badge warn">Belum dieksekusi penuh (${n.nExecuted}/${n.nCode})</span>`;

  // ---------- Filter ----------
  const ok = forStr => TK.applies(forStr, Filter.task);
  const slotOk = id => !id || ok(TK.slotFor[id] || "all");
  const methodOk = m => ok(m.for) && !(Filter.noPre && m.pre);
  const metricsVisible = g => ok(g.for);
  function banner(hidden, noun) {
    if (!Filter.task && !Filter.noPre) return "";
    const parts = [];
    if (Filter.task) parts.push(`Task: <b>${esc(taskName(Filter.task))}</b>`);
    if (Filter.noPre) parts.push("<b>Tanpa pretrained</b>");
    return `<div class="fbanner"><span>${parts.join(" · ")}${hidden ? ` <span class="muted">· ${hidden} ${noun || "item"} disembunyikan</span>` : ""}</span><button class="btn" data-clear-filter>Tampilkan semua</button></div>`;
  }

  const testBadge = id => {
    const r = ((L().tests || {}).result || {})[id];
    return r === "OK" ? `<span class="badge good" title="Lolos tools/test_snippets.py">teruji</span>` : r ? `<span class="badge bad" title="${esc(r)}">gagal uji</span>` : "";
  };
  const cartBtn = id => `<button class="btn mini" data-cart="${esc(id)}" aria-pressed="${(Filter.cart || []).includes(id)}">${(Filter.cart || []).includes(id) ? "✓ Keranjang" : "+ Keranjang"}</button>`;
  function slotActions(id) {
    const needs = (L().snippets[id].needs || []);
    return `<div class="row actions">${cartBtn(id)}${needs.length ? `<button class="btn mini" data-copy-chain="${esc(id)}">Salin + ${Code.chain(id).length - 1} dependensi</button><span class="small muted">butuh ${needs.map(n => `<a href="#/snippets/${n}">${esc(n)}</a>`).join(", ")}</span>` : ""}${testBadge(id)}</div>`;
  }
  function slot(id, opts) {
    if (!id) return "";
    const s = L().snippets[id];
    if (!TK.slots[id] && !s) return `<span class="slot">${esc(id)}</span>`;
    if (!s) return `<a class="slot" href="#/snippets/${id}" title="${esc(TK.slots[id])}">○ ${esc(id)}</a>`;
    if (opts && opts.link) return `<a class="slot filled" href="#/snippets/${id}">● ${esc(id)}</a>`;
    return `<div class="slotcode"><div class="small muted">Kode: <a href="#/snippets/${id}">${esc(s.title || id)}</a></div>${codebox(s.code, "python")}${slotActions(id)}</div>`;
  }

  const libStatus = p => {
    if (TK.stdlibOrExternal[p]) return "ext";
    const key = p.toLowerCase().replace(/_/g, "-");
    return TK.bundle && TK.bundle[key] ? "in" : (TK.extraPkgs && TK.extraPkgs[key] ? "add" : "out");
  };
  function libChips(lib) {
    if (!lib) return "";
    return lib.split(/\s+/).filter(Boolean).map(p => {
      const st = libStatus(p);
      const cls = st === "in" ? "" : st === "add" ? "warn" : st === "ext" ? "" : "bad";
      const tip = st === "in" ? "Ada di bundle" : st === "add" ? "Tambahan yang disarankan: lihat halaman Library" : st === "ext" ? "Di luar pip" : "Tidak ada di bundle";
      return `<a class="badge ${cls}" href="#/library" title="${tip}">${esc(p)}</a>`;
    }).join(" ");
  }

  function methodCard(m) {
    const a = "a-" + m.id;
    const code = m.imp || m.api ? (m.api && /[(\[.]/.test(m.api) && !/^[\w.]+$/.test(m.api) ? [m.imp, m.api].filter(Boolean).join("\n") : (m.imp || m.api)) : "";
    return `<div class="card method" id="${a}">
      <div class="between"><h3>${inline(m.n)}</h3><div class="row"><span class="badge p${m.prio}" title="1 = kerjakan dulu, 2 = alternatif kuat, 3 = situasional">P${m.prio}</span>${m.pre ? '<span class="badge warn" title="Butuh model pretrained">pretrained</span>' : ""}${m.cost ? `<span class="badge">${esc(m.cost)}</span>` : ""}</div></div>
      <p class="small" style="margin:.1rem 0 .3rem">${inline(m.what)}</p>
      <p class="small" style="margin:0 0 .2rem"><b>Kapan:</b> ${inline(m.when)}</p>
      ${m.avoid ? `<p class="small" style="margin:0 0 .3rem"><b>Hindari:</b> ${inline(m.avoid)}</p>` : ""}
      ${code ? `<div class="small muted" style="margin-bottom:.15rem">Panggilan inti:</div>` + codebox(code, "python") : ""}
      <div class="row">${libChips(m.lib)}</div>
      ${slot(m.slot)}
    </div>`;
  }

  // ---------- Beranda ----------
  function home() {
    const snipTotal = Object.keys(TK.slots).length, snipDone = Object.keys(TK.slots).filter(k => L().snippets[k]).length;
    return `
    <h1>Mulai di sini</h1>
    <p class="lead">Rabu 11 November, 13.00–18.00. Ikuti 25 menit pertama, pilih task di bilah atas, lalu kerjakan <a href="#/pipeline">Pipeline</a>. Semua halaman akan menyaring metode yang tidak berlaku. Tekan <kbd>Ctrl K</kbd> untuk mencari apa pun.</p>
    <div class="two">
      <div class="card"><h3>25 menit pertama</h3>
        ${TK.firstChecklist.map((c, i) => `<label class="check"><input type="checkbox" data-check="first-${i}"><span>${inline(c.t)} <span class="muted small">(${c.m} menit)</span>${c.slot ? " " + slot(c.slot, { link: true }) : ""}</span></label>`).join("")}
      </div>
      <div class="card"><h3>Pintasan saat panik</h3>
        <div class="row" style="margin-bottom:.6rem">
          <a class="chip" href="#/decision">Tentukan task</a>
          <a class="chip" href="#/strategy/imbalance">Imbalance</a>
          <a class="chip" href="#/strategy/shift">CV ≠ skor test</a>
          <a class="chip" href="#/strategy/vram">CUDA OOM</a>
          <a class="chip" href="#/strategy/no-pretrained">Pretrained dilarang</a>
          <a class="chip" href="#/strategy/time">Waktu mepet</a>
          <a class="chip" href="#/readiness">Argumen tak dikenal</a>
          <a class="chip" href="#/lomba">Aturan lomba</a>
        </div>
        <h3>Status isi toolkit</h3>
        <p class="small muted" style="margin-bottom:.3rem">${TK.methods.length} metode · ${Object.keys(L().pipelines || {}).length} pipeline · Notebook: <b>${L().notebooks.length}</b> · Snippet: <b>${snipDone}</b> dari ${snipTotal} slot</p>
        <div class="progress"><i style="width:${snipTotal ? (100 * snipDone / snipTotal) : 0}%"></i></div>
        <p class="small" style="margin-top:.5rem"><a href="#/coverage">Lihat apa yang belum terisi</a></p>
      </div>
    </div>
    <h2>Rencana 5 jam</h2>
    <div class="timeline" id="timeline">${TK.timePlan.map((p, i) => `<div data-phase="${i}" style="flex:${p.to - p.from}" title="${esc(p.name)}">${esc(p.name)}</div>`).join("")}</div>
    <div class="table-wrap"><table>
      <tr><th>T (menit)</th><th>WIB</th><th>Kerja</th><th>Gerbang</th></tr>
      ${TK.timePlan.map(p => `<tr><td style="white-space:nowrap">${p.from}–${p.to}</td><td style="white-space:nowrap">${TK.wib(p.from)}</td><td>${inline(p.name)}<br><span class="small muted">${inline(p.out)}</span></td><td>${inline(p.stop)}</td></tr>`).join("")}
    </table></div>
    <p class="small muted">Timeline ini [Penilaian] dari ringkasan persiapanmu, T0 = 13.00 WIB. Sesuaikan setelah latihan. Mulai timer dari tombol di bilah atas.</p>`;
  }

  // ---------- Info Lomba ----------
  function lomba() {
    const I = TK.lomba;
    return `<h1>Info Lomba</h1>
    <p class="lead">GEMASTIK XIX, Kompetisi III (Penambangan Data). Tema MBG dan 500.000 baris berasal dari informasimu, bukan dokumen publik.</p>
    <div class="two">
      <div class="card"><h3>Aturan</h3>${list(I.rules)}</div>
      <div class="card"><h3>Bobot nilai</h3>${I.score.map(s => `<div class="between small" style="margin-bottom:.15rem"><span>${esc(s[0])}</span><b>${s[1]}%</b></div><div class="progress" style="margin-bottom:.4rem"><i style="width:${s[1] * 2}%"></i></div>`).join("")}<p class="small muted" style="margin:.4rem 0 0">50% nilai ada di metode dan presentasi.</p></div>
    </div>
    <h2>Jadwal</h2><div class="table-wrap"><table><tr><th>Tanggal</th><th>WIB</th><th>Kegiatan</th><th>Tempat</th></tr>${I.schedule.map(r => `<tr>${r.map(c => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</table></div>
    <p class="small muted">${inline(I.scheduleNote)}</p>
    <div class="first">${inline(I.precedent)}</div>
    <h2>Task yang mungkin</h2><p class="small muted">Probabilitas adalah [Penilaian], bukan data.</p>
    <div class="table-wrap"><table><tr><th>Task</th><th style="width:40%">Peluang</th><th></th></tr>${I.taskProb.map(r => `<tr><td>${inline(r[1])}</td><td><div class="progress"><i style="width:${r[2] * 2.5}%"></i></div> <span class="small">${r[2]}%</span></td><td><button class="btn" data-set-task="${r[0]}">Pilih</button></td></tr>`).join("")}</table></div>
    <p class="small">${inline(I.modifiers)}</p>
    <h2>Pohon gerbang T40</h2><div class="grid">${I.gates.map(g => `<div class="card"><h3>${esc(g.t)}</h3><p class="small" style="margin:0">${inline(g.d)}</p></div>`).join("")}</div>
    <h2>Pertanyaan ke panitia</h2><div class="card">${I.questions.map((q, i) => `<label class="check"><input type="checkbox" data-check="q-${i}"><span>${inline(q)}</span></label>`).join("")}</div>
    <p class="small muted"><b>Belum diketahui:</b> ${inline(I.unknown)}</p>
    <h2>Jangan</h2><div class="table-wrap"><table><tr><th>Jangan</th><th>Alasan</th></tr>${I.dont.map(r => `<tr><td>${inline(r[0])}</td><td>${inline(r[1])}</td></tr>`).join("")}</table></div>
    <h2>Kerangka slide</h2><div class="two"><div class="card"><ol>${I.slides.map(s => `<li>${esc(s)}</li>`).join("")}</ol></div><div class="card"><h3>Kerja tim</h3><p class="small" style="margin:0">${inline(I.team)}</p></div></div>`;
  }
  function mbg() {
    const M = TK.mbg;
    return `<h1>Domain MBG</h1><p class="lead">Bahan label lemah, fitur lonjakan, dan slide. Semuanya hipotesis sampai cocok dengan data panitia.</p>
    <h2>Empat rezim wacana</h2><div class="card"><ol>${M.regimes.map(r => `<li>${esc(r)}</li>`).join("")}</ol><p class="small muted" style="margin:0">Model pada satu jendela waktu akan drift.</p></div>
    <h2>Peristiwa untuk fitur lonjakan</h2><div class="table-wrap"><table><tr><th>Tanggal</th><th>Peristiwa</th></tr>${M.events.map(e => `<tr><td style="white-space:nowrap">${esc(e[0])}</td><td>${esc(e[1])}</td></tr>`).join("")}</table></div>
    <div class="first">${inline(M.disputed)}</div>
    <h2>Jebakan data yang perlu diduga</h2><div class="card">${list(M.traps)}</div>
    <h2>Leksikon awal</h2><p class="small muted">${inline(M.lexNote)}</p>
    <div class="table-wrap"><table><tr><th>Aspek</th><th>Kata kunci</th><th></th></tr>${M.lexicon.map(l => `<tr><td id="a-${slug(l[0])}"><b>${esc(l[0])}</b></td><td class="small">${esc(l[1])}</td><td><button class="btn" data-copy-text="${esc(l[1].replace(/\[[DPI]\]\s*/g, ""))}">Salin</button></td></tr>`).join("")}</table></div>
    <p class="small">${inline(M.data)}</p>`;
  }

  // ---------- Decision Guide ----------
  function decision() {
    const rows = TK.quickRef.filter(r => ok(r[1]));
    return `
    <h1>Decision Guide</h1>
    <p class="lead">Dari dataset baru ke baseline pertama. Jawab pertanyaan, atau klik daun pada pohon. Hasilnya langsung menjadi filter task.</p>
    <div class="card"><div class="between"><h3 id="wq"></h3><button class="btn" id="wreset">Ulangi</button></div><div class="row" id="wopts"></div><div id="wres"></div></div>
    <div class="tree-wrap" id="tree"></div>
    <h2>Lima langkah</h2>
    <div class="grid">${TK.decisionSteps.map(s => `<div class="card"><h3>Langkah ${s.n}: ${esc(s.t)}</h3>${list(s.items)}${slot(s.slot, { link: true })}</div>`).join("")}</div>
    <h2>Tabel cepat</h2>
    ${banner(TK.quickRef.length - rows.length, "baris")}
    <div class="table-wrap"><table><tr><th>Yang kamu lihat</th><th>Task</th><th>Baseline</th><th>Metrik umum</th><th>Jangan lupa</th></tr>
      ${rows.map(r => `<tr><td>${esc(r[0])}</td><td>${taskLink(r[1])}</td><td>${esc(r[2])}</td><td>${esc(r[3])}</td><td>${esc(r[4])}</td></tr>`).join("")}
    </table></div>
    <h2>Prioritas perbaikan</h2>
    <p class="small">Decision layer ≥ perbaikan validasi dan struktur > adaptasi domain saat shift > anggota ensemble dari representasi baru > pembersihan label terkonfirmasi > fitur tertarget > seed/fold bagging > hyperparameter tuning > class weighting.</p>
    <h2>Pelajaran dari portofolio kamu</h2>
    <p class="small muted">Periksa dulu apakah kondisinya sama sebelum menerapkan.</p>
    <div class="grid">${TK.lessons.map(l => `<div class="card"><div class="between"><h3>${esc(l.t)}</h3><span class="badge">${esc(l.tag)}</span></div><p class="small" style="margin:0">${inline(l.d)}</p></div>`).join("")}</div>`;
  }

  // ---------- Pipeline ----------
  function pipeline() {
    if (!Filter.task) {
      return `<h1>Pipeline per Task</h1><p class="lead">Pilih task. Kamu mendapat satu sel per tahap: buka, salin, jalankan, lanjut ke tahap berikutnya. Semua variabel nyambung.</p>
      ${Object.keys(TK.taskGroups).map(g => `<h2>${esc(TK.taskGroups[g])}</h2><div class="row">${TK.tasks.filter(t => t.group === g).map(t => `<button class="chip" data-set-task="${t.id}">${esc(t.name)}</button>`).join("")}</div>`).join("")}`;
    }
    const t = taskObj(Filter.task), p = Code.pipelineOf(t.id);
    if (!p) return `<h1>Pipeline: ${esc(t.name)}</h1><div class="card">Belum ada pipeline untuk task ini.</div>`;
    const tested = ((L().tests || {}).pipelines || {})[p.key];
    const methodsIn = sid => TK.methods.filter(m => m.stage === (TK.stageMap[sid] || sid) && methodOk(m));
    return `<div class="between"><h1>${esc(t.name)}</h1><div class="row"><button class="btn primary" data-copy-all>Salin semua sel</button><button class="btn" data-dl-nb>Unduh .ipynb</button><a class="btn" href="#/tasks/${t.id}">Detail task</a></div></div>
    <p class="lead">Pipeline <b>${esc(p.title)}</b>. ${p.stages.length} sel berurutan. Ubah hanya sel Setup, lalu jalankan dari atas ke bawah.${tested ? ` <span class="badge good">Lolos uji berurutan: ${esc(tested)}</span>` : ""}</p>
    <div class="row" style="margin-bottom:.8rem"><button class="btn mini" data-open-all>Buka semua</button><button class="btn mini" data-close-all>Tutup semua</button></div>
    ${p.stages.map((s, i) => {
      const ms = methodsIn(s.id);
      return `<details class="stagecell" ${i === 0 ? "open" : ""} id="s-${s.id}"><summary><span class="num">${i + 1}</span><span class="sname">${esc(s.name)}</span><span class="small muted">${s.code.split("\n").length} baris</span><button class="btn mini" data-copy-stage="${i}">Salin sel</button></summary>
        <p class="small muted" style="margin:.4rem 0">${inline(s.note)}</p>
        ${codebox(s.code, "python")}
        ${ms.length ? `<details class="alt"><summary class="small">${ms.length} metode untuk tahap ini</summary><div class="small">${ms.sort((a, b) => a.prio - b.prio).map(m => `<a class="chip" style="display:inline-block;margin:.1rem" href="#/stage/${m.stage}/${m.id.slice(m.stage.length + 1)}">${inline(m.n)}</a>`).join("")}</div></details>` : ""}
      </details>`;
    }).join("")}`;
  }

  // ---------- Keranjang ----------
  function cart() {
    const ids = Filter.cart || [];
    const lib = L().snippets;
    const chain = [];
    ids.forEach(i => Code.chain(i, chain));
    const extra = chain.filter(i => !ids.includes(i));
    const stages = Object.keys(TK.snippetCategories);
    return `<h1>Keranjang Kode</h1><p class="lead">Pilih snippet dari kartu mana pun dengan <b>+ Keranjang</b>. Dependensi ikut otomatis, tanpa duplikat, dan urutannya mengikuti kebutuhan.</p>
    ${ids.length ? `<div class="row" style="margin-bottom:.8rem"><button class="btn primary" data-copy-cart>Salin semua (${chain.length} snippet)</button><button class="btn" data-dl-cart>Unduh toolkit.py</button><button class="btn" data-clear-cart>Kosongkan</button></div>
    ${ids.map(i => `<div class="card citem"><div class="between"><span><code>${esc(i)}</code> <b>${esc((lib[i] || {}).title || "")}</b></span><button class="btn mini" data-cart="${esc(i)}">Hapus</button></div></div>`).join("")}
    ${extra.length ? `<p class="small muted">Ikut sebagai dependensi: ${extra.map(esc).join(", ")}</p>` : ""}
    <h2>Pratinjau</h2>${codebox(Code.assemble(ids), "python")}` : `<div class="card"><p>Keranjang kosong.</p><div class="row">${stages.map(c => `<a class="chip" href="#/snippets/${c}">${esc(TK.snippetCategories[c])}</a>`).join("")}</div></div>`}`;
  }

  // ---------- Tahap ----------
  function stage(sub) {
    const s = TK.stages.find(x => x.id === sub) || TK.stages[0];
    const all = TK.methods.filter(m => m.stage === s.id);
    const vis = all.filter(m => methodOk(m) && !(Filter.prio1 && m.prio !== 1));
    const groups = [...new Set(vis.map(m => m.group))];
    return `<h1>${esc(s.name)}</h1><p class="lead">${inline(s.desc)}</p>
    <div class="filters">${TK.stages.map(x => `<a class="chip ${x.id === s.id ? "on" : ""}" href="#/stage/${x.id}">${esc(x.short)}</a>`).join("")}</div>
    ${banner(all.length - vis.length, "metode")}
    <label class="check small" style="margin-bottom:.5rem"><input type="checkbox" id="prio1" ${Filter.prio1 ? "checked" : ""}><span>Hanya prioritas 1</span></label>
    ${TK.stageMain[s.id] && L().snippets[TK.stageMain[s.id]] ? `<details class="stagecell" open><summary><span class="sname">Satu fungsi utuh tahap ini</span><span class="small muted">${esc(L().snippets[TK.stageMain[s.id]].title)}</span></summary>${L().snippets[TK.stageMain[s.id]].when ? `<p class="small muted" style="margin:.4rem 0">${inline(L().snippets[TK.stageMain[s.id]].when)}</p>` : ""}${codebox(L().snippets[TK.stageMain[s.id]].code, "python")}</details>` : ""}
    ${s.id === "prep" ? prepTable() : ""}
    ${vis.length ? groups.map(g => `<h2>${esc(g)}</h2><div class="grid wide">${vis.filter(m => m.group === g).sort((a, b) => a.prio - b.prio).map(methodCard).join("")}</div>`).join("") : '<div class="card muted">Tidak ada metode yang berlaku untuk filter ini.</div>'}`;
  }
  function prepTable() {
    return `<div class="card small"><b>Aturan umum:</b>${list(TK.preprocessingRules)}
    <div class="table-wrap" style="margin:0"><table><tr><th>Representasi</th><th>Cabang awal</th><th>Catatan</th></tr>${TK.prepChoice.filter(r => ok(r[3])).map(r => `<tr><td>${esc(r[0])}</td><td>${esc(r[1])}</td><td>${inline(r[2])}</td></tr>`).join("")}</table></div></div>`;
  }

  // ---------- Task ----------
  function tasks(sub) {
    if (sub && taskObj(sub)) return taskDetail(taskObj(sub));
    const g = Object.keys(TK.taskGroups);
    return `
    <h1>Task Explorer</h1>
    <p class="lead">${TK.tasks.length} task. Label prioritas adalah estimasi persiapan, bukan informasi panitia. Klik task untuk detail, lalu jadikan filter.</p>
    <div class="filters" id="taskFilters"><button class="chip on" data-f="all">Semua</button>${g.map(k => `<button class="chip" data-f="${k}">${esc(TK.taskGroups[k])}</button>`).join("")}<button class="chip" data-f="prio">Prioritas utama</button></div>
    ${g.map(k => `<section data-group="${k}"><h2>${esc(TK.taskGroups[k])}</h2><div class="grid">${TK.tasks.filter(t => t.group === k).map(t => {
      const nb = notebooksFor(t.id).length;
      return `<a class="card ${Filter.task === t.id ? "selected" : ""}" href="#/tasks/${t.id}" data-prio="${t.prio}"><div class="between"><h3>${esc(t.name)}</h3><span class="badge ${t.prio}" title="${prioLabel[t.prio]}">${t.prio}</span></div><p class="small muted" style="margin:.1rem 0 .4rem">${inline(t.baseline)}</p><div class="row"><span class="badge">Label: ${esc(t.label)}</span><span class="badge ${nb ? "good" : ""}">${nb} notebook</span></div></a>`;
    }).join("")}</div></section>`).join("")}`;
  }
  const METRIC_ALIAS = { "f1": "f1-score", "f1-kelas-spam": "f1-score", "f1-pasangan": "f1-score", "macro-f1-pasangan": "macro-f1", "silhouette": "silhouette-score", "pearson": "pearson-spearman", "spearman": "pearson-spearman", "interpretability": "topic-interpretability", "entity-level-p-r-f1": "entity-level-f1", "rouge-1-2-l": "rouge", "ari-nmi-bila-ada-ground-truth": "adjusted-rand-index", "precision-k-bila-ada-ground-truth": "precision-k", "p-k-r-k-f1-k-bila-ada-ground-truth": "precision-k", "roc-auc-pr-auc-bila-ada-label": "roc-auc", "rmsle": "rmse" };
  function metricChip(m) {
    const known = new Set(TK.metrics.flatMap(g => g.items.map(i => slug(i.n))));
    const s = slug(m.split(" (")[0]), target = METRIC_ALIAS[s] || s;
    const style = 'style="margin:0 .2rem .2rem 0;display:inline-block"';
    return known.has(target) ? `<a class="chip" ${style} href="#/metrics/${target}">${esc(m)}</a>` : `<span class="badge" ${style}>${esc(m)}</span>`;
  }
  function taskDetail(t) {
    const nbs = notebooksFor(t.id), isOn = Filter.task === t.id;
    return `
    <p class="small"><a href="#/tasks">← Task Explorer</a></p>
    <div class="between"><h1>${esc(t.name)}</h1><div class="row"><span class="badge ${t.prio}">${prioLabel[t.prio]}</span><button class="btn primary" data-set-task="${t.id}" data-go="#/pipeline">${isOn ? "Buka pipeline" : "Jadikan filter & buka pipeline"}</button></div></div>
    <p class="lead">${inline(t.goal)}</p>
    <div class="two">
      <div class="card"><dl class="kv">
        <dt>Butuh label</dt><dd>${esc(t.label)}</dd>
        <dt>Kelompok</dt><dd>${esc(TK.taskGroups[t.group])}</dd>
        <dt>Metrik</dt><dd>${t.metrics.map(metricChip).join("")}</dd>
      </dl></div>
      <div class="card"><h3>Baseline yang direkomendasikan</h3><p>${inline(t.baseline)}</p></div>
    </div>
    <div class="two">
      <div class="card"><h3>Contoh permasalahan</h3>${list(t.examples)}<h3>Karakteristik dataset</h3>${list(t.traits)}</div>
      <div class="card"><h3>Pendekatan lebih lanjut</h3>${list(t.advanced)}<h3>Jebakan</h3>${list(t.pitfalls, "avoid")}</div>
    </div>
    ${Code.pipelineOf(t.id) ? `<h2>Kode end-to-end</h2><div class="card recipe"><div class="between"><span>${Code.pipelineOf(t.id).stages.length} sel berurutan: <b>${esc(Code.pipelineOf(t.id).title)}</b></span><button class="btn primary" data-set-task="${t.id}" data-go="#/pipeline">Buka per tahap</button></div></div>` : ""}
    <h2>Notebook end-to-end</h2>
    ${nbs.length ? `<div class="grid">${nbs.map(nbCard).join("")}</div>` : `<div class="card small muted">Belum ada notebook. Simpan sebagai <code>notebooks/&lt;kategori&gt;/${t.id}__nama.ipynb</code>, lalu jalankan <code>python tools/build_library.py</code>.</div>`}
    <h2>Snippet modular</h2>
    <div class="card">${t.slots.map(s => `<div style="margin-bottom:.35rem">${slot(s)} <span class="small muted">${esc(TK.slots[s] || "")}</span></div>`).join("")}</div>
    <h2>Dataset demonstrasi</h2>
    <div class="row">${t.datasets.map(k => { const c = (TK.catalog || []).find(x => x.key === k); return c ? `<a class="chip" href="#/datasets/${k}">${esc(c.name)}</a>` : ""; }).join("")}</div>`;
  }

  // ---------- Notebook ----------
  function nbCard(n) {
    return `<a class="card" href="#/notebooks/${n.id}"><h3>${esc(n.title)}</h3><p class="small muted" style="margin:.1rem 0 .4rem">${inline(n.when || n.path)}</p><div class="row">${nbBadge(n)}${(n.tags || []).map(t => `<span class="badge">${esc(t)}</span>`).join("")}</div></a>`;
  }
  function notebooks(sub) {
    const all = L().notebooks;
    if (sub) {
      const n = all.find(x => x.id === sub);
      if (n) return `<p class="small"><a href="#/notebooks">← Notebook Library</a></p><div class="between"><h1>${esc(n.title)}</h1><div class="row"><button class="btn" data-nb-copy>Salin semua kode</button><button class="btn primary" data-nb-dl="${esc(n.id)}">Unduh .ipynb</button></div></div>
        <p class="small muted">${esc(n.path)} · ${n.nCode} sel kode · ${fmtBytes(n.bytes)}</p><div class="row" style="margin-bottom:1rem">${nbBadge(n)}${(n.tasks || []).map(t => `<a class="chip" href="#/tasks/${t}">${esc(taskName(t))}</a>`).join("")}</div>${n.when ? `<div class="first">${inline(n.when)}</div>` : ""}
        <div id="nbView" data-nb="${esc(n.id)}"><p class="muted">Memuat notebook…</p></div>`;
    }
    const vis = Filter.task ? all.filter(n => (n.tasks || []).includes(Filter.task)) : all;
    const tags = [...new Set(vis.flatMap(n => n.tags || []))].sort();
    return `
    <h1>Notebook Library</h1>
    <p class="lead">Pilih berdasarkan task dan kondisi dataset. Status eksekusi dibaca dari output yang tersimpan di file notebook.</p>
    ${banner(all.length - vis.length, "notebook")}
    <div class="drop"><p style="margin-bottom:.4rem">Lihat notebook apa pun tanpa build:</p><input type="file" id="nbFile" accept=".ipynb" aria-label="Pilih file notebook"></div>
    <div id="nbFileView"></div>
    ${vis.length ? `<h2>${vis.length} notebook</h2>
      ${tags.length ? `<div class="filters" id="nbTags"><button class="chip on" data-t="">Semua</button>${tags.map(t => `<button class="chip" data-t="${esc(t)}">${esc(t)}</button>`).join("")}</div>` : ""}
      ${Object.keys(TK.taskGroups).map(g => {
        const items = vis.filter(n => (n.tasks || []).some(t => (taskObj(t) || {}).group === g));
        return items.length ? `<h2>${esc(TK.taskGroups[g])}</h2><div class="grid">${items.map(n => nbCard(n).replace('<a class="card"', `<a class="card" data-tags="${esc((n.tags || []).join("|"))}"`)).join("")}</div>` : "";
      }).join("")}
      ${vis.some(n => !(n.tasks || []).length) ? `<h2>Belum terhubung ke task</h2><div class="grid">${vis.filter(n => !(n.tasks || []).length).map(nbCard).join("")}</div>` : ""}`
      : `<h2>Cara menambahkan</h2><div class="card"><ol>
          <li>Simpan di <code>notebooks/&lt;kategori&gt;/&lt;task_id&gt;__&lt;varian&gt;.ipynb</code>${Filter.task ? `, misalnya <code>notebooks/${Filter.task}__tfidf.ipynb</code>` : ", misalnya <code>notebooks/classification/binary__tfidf_lr.ipynb</code>"}.</li>
          <li>Jalankan <code>python tools/build_library.py</code>.</li>
          <li>Muat ulang halaman ini. Notebook muncul di sini, di task terkait, dan di hasil pencarian.</li>
        </ol><p class="small muted" style="margin:0">Daftar <code>task_id</code> ada di halaman <a href="#/coverage">Progres</a>. Nama bebas juga bisa, dengan mengisi <code>notebooks/manifest.json</code> (lihat README).</p></div>`}`;
  }

  // ---------- Snippet ----------
  function snippets(sub, sub2) {
    const lib = L().snippets, cats = TK.snippetCategories;
    const id = sub2 ? sub + "/" + sub2 : null;
    const filter = id ? sub : (sub && cats[sub] ? sub : "");
    const ids = [...new Set([...Object.keys(TK.slots), ...Object.keys(lib)])];
    const applicable = ids.filter(slotOk);
    const shown = filter ? applicable.filter(i => i.startsWith(filter + "/")) : applicable;
    const filled = applicable.filter(i => lib[i]).length;
    const usedCats = Object.keys(cats).filter(c => applicable.some(i => i.startsWith(c + "/")));
    return `
    <h1>Snippet Library</h1>
    <p class="lead">${filled} dari ${applicable.length} slot terisi. Vectorizer dan model mengikuti gaya scikit-learn, jadi bisa ditukar langsung di <code>oof_predict</code>. Kumpulkan yang kamu perlu di <a href="#/cart">Keranjang</a>.</p>
    ${banner(ids.length - applicable.length, "slot")}
    <div class="filters"><a class="chip ${filter ? "" : "on"}" href="#/snippets">Semua</a>${usedCats.map(c => `<a class="chip ${filter === c ? "on" : ""}" href="#/snippets/${c}">${esc(cats[c])}</a>`).join("")}</div>
    ${(filter ? [filter] : usedCats).map(c => {
      const items = shown.filter(i => i.startsWith(c + "/"));
      if (!items.length) return "";
      return `<h2>${esc(cats[c] || c)}</h2>${items.map(i => {
        const s = lib[i], a = "a-" + i.replace("/", "-");
        return s ? `<div class="card" id="${a}"><div class="between"><h3>${esc(s.title || i)}</h3>${s.valid === false ? `<span class="badge bad">Sintaks tidak valid</span>` : ""}</div>${s.when ? `<p class="small muted" style="margin-bottom:.3rem">${inline(s.when)}</p>` : ""}${s.avoid ? `<p class="small" style="margin-bottom:.3rem"><b>Hindari:</b> ${inline(s.avoid)}</p>` : ""}${codebox(s.code, "python")}${slotActions(i)}</div>`
          : `<div class="card" id="${a}" style="padding:.55rem .85rem"><div class="between"><span>○ <code>${esc(i)}</code></span><span class="badge">Slot kosong</span></div><p class="small muted" style="margin:.2rem 0 0">${esc(TK.slots[i])}</p></div>`;
      }).join("")}`;
    }).join("")}`;
  }

  // ---------- Strategy ----------
  function strategy(sub) {
    const s = TK.strategy.find(x => x.id === sub);
    if (s) {
      const recs = s.recs.filter(r => slotOk(r.slot));
      return `
      <p class="small"><a href="#/strategy">← Strategy & Troubleshooting</a></p>
      <h1>${esc(s.icon)} ${esc(s.title)}</h1>
      <p class="lead"><b>Tanda:</b> ${inline(s.signal)}</p>
      ${banner(s.recs.length - recs.length, "teknik")}
      <div class="first"><b>Langkah pertama:</b> ${inline(s.first)}</div>
      <h2>Teknik, urut dari yang paling layak</h2>
      <div class="card">${recs.map(r => `<div class="rec"><div><b>${inline(r.t)}</b><p class="small" style="margin:.1rem 0 .25rem">${inline(r.why)}</p>${slot(r.slot)}</div><span class="badge">${esc(r.cost)}</span></div>`).join("") || '<p class="small muted" style="margin:0">Tidak ada teknik untuk filter ini.</p>'}</div>
      <h2>Hindari</h2><div class="card">${list(s.avoid, "avoid")}</div>`;
    }
    const vis = TK.strategy.filter(x => ok(TK.strategyFor[x.id] || "all"));
    return `
    <h1>Strategy & Troubleshooting</h1>
    <p class="lead">Temukan kondisi dataset atau masalah yang kamu hadapi. Ambang angkanya aturan praktis, jadi tetap uji pada datamu.</p>
    ${banner(TK.strategy.length - vis.length, "kondisi")}
    <div class="grid">${vis.map(x => `<a class="card" href="#/strategy/${x.id}"><h3>${esc(x.icon)} ${esc(x.title)}</h3><p class="small muted" style="margin:0">${inline(x.signal)}</p></a>`).join("")}</div>`;
  }

  // ---------- Diagnostics ----------
  function diagnostics() {
    const groups = TK.diagnostics.map(g => ({ g, items: ok(TK.diagFor[g.id] || "all") ? g.items.filter(i => slotOk(i.slot)) : [] }));
    const total = TK.diagnostics.reduce((s, g) => s + g.items.length, 0), vis = groups.reduce((s, x) => s + x.items.length, 0);
    return `
    <h1>Dataset Diagnostics</h1>
    <p class="lead">Pemeriksaan yang bisa dijalankan cepat. Setiap temuan harus berakhir dengan keputusan.</p>
    ${banner(total - vis, "pemeriksaan")}
    ${groups.filter(x => x.items.length).map(({ g, items }) => `<h2>${esc(g.title)}</h2><div class="table-wrap"><table>
      <tr><th>Pemeriksaan</th><th>${g.id === "valid" ? "Kapan dipakai" : "Cara membaca"}</th><th>${g.id === "valid" ? "Catatan" : "Tindakan"}</th>${g.id === "valid" ? "<th>Risiko leakage</th>" : ""}<th>Kode</th></tr>
      ${items.map(i => `<tr id="a-${slug(i.n)}"><td><b>${esc(i.n)}</b></td><td>${inline(i.read)}</td><td>${inline(i.act)}</td>${g.id === "valid" ? `<td>${inline(i.leak)}</td>` : ""}<td>${slot(i.slot, { link: true })}</td></tr>`).join("")}
    </table></div>`).join("")}`;
  }

  // ---------- Metrik ----------
  function metrics() {
    const vis = TK.metrics.filter(metricsVisible);
    return `<h1>Evaluation Metrics</h1><p class="lead">Metrik panitia menentukan objective dan decision layer. Tulis ulang rumusnya dan uji sebelum mengoptimalkan.</p>
    ${banner(TK.metrics.length - vis.length, "kelompok metrik")}
    ${vis.map(g => `<div class="between"><h2>${esc(g.group)}</h2>${slot(g.slot, { link: true })}</div><div class="table-wrap"><table><tr><th>Metrik</th><th>Rumus</th><th>Kapan relevan</th><th>Jebakan</th></tr>
    ${g.items.filter(m => ok(m.for || "all")).map(m => `<tr id="a-${slug(m.n)}"><td><b>${esc(m.n)}</b></td><td>${inline(m.f)}</td><td>${inline(m.when)}</td><td>${inline(m.trap)}</td></tr>`).join("")}</table></div>`).join("")}`;
  }

  // ---------- Model ----------
  function models() {
    const vis = TK.bundleModels.filter(m => ok(m[5]));
    return `<h1>Model Pretrained</h1>
    <p class="lead">Item model di <code>01_download_all.py</code>. Skor terbit berasal dari domain lain, bukan data MBG. Metode pemakaiannya ada di <a href="#/stage/model">Modeling</a>.</p>
    ${banner(TK.bundleModels.length - vis.length, "model")}
    <div class="table-wrap"><table><tr><th>Kode</th><th>Profil</th><th>Model</th><th>Fungsi</th><th>Bobot</th></tr>${vis.map(m => `<tr><td>${esc(m[0])}</td><td><span class="badge ${m[1] === "core" ? "good" : m[1] === "plus" ? "A" : ""}">${esc(m[1])}</span></td><td><code>${esc(m[2])}</code></td><td>${esc(m[3])}</td><td style="white-space:nowrap">${esc(m[4])}</td></tr>`).join("")}</table></div>
    <div class="first">${inline(TK.modelAvoid)}</div>
    <h2>BERT vs TF-IDF: rentangnya lebar</h2>
    <div class="table-wrap"><table><tr><th>Data</th><th>Klasik</th><th>Transformer</th><th>Catatan</th></tr>${TK.bertVsTfidf.map(r => `<tr>${r.map(c => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</table></div>
    <p class="small">[Penilaian] TF-IDF dulu sebagai submission aman, BERT sebagai taruhan GPU. Keduanya mengisi blend.</p>`;
  }

  // ---------- Library ----------
  function library() {
    const used = {};
    TK.methods.filter(methodOk).forEach(m => (m.lib || "").split(/\s+/).filter(Boolean).forEach(p => { (used[p] = used[p] || []).push(m); }));
    const rows = Object.keys(used).sort((a, b) => a.localeCompare(b));
    const stat = { in: '<span class="badge good">Ada di bundle</span>', add: '<span class="badge warn">Tambahan disarankan</span>', out: '<span class="badge bad">Tidak ada</span>', ext: '<span class="badge">Di luar pip</span>' };
    const ver = p => (TK.bundle || {})[p.toLowerCase().replace(/_/g, "-")] || (TK.extraPkgs || {})[p.toLowerCase().replace(/_/g, "-")] || "–";
    return `<h1>Library</h1>
    <p class="lead">Setiap paket yang dipakai metode di toolkit, dicocokkan dengan <code>requirements.in</code> bundle <code>persiapan_h1</code>. Versi berasal dari instalasi nyata di venv terpisah (6 Nov 2026, dari wheelhouse bundle dengan <code>--no-index</code>, torch 2.14.1+cu130).</p>
    ${banner(0)}
    <div class="table-wrap"><table><tr><th>Paket</th><th>Versi</th><th>Status</th><th>Dipakai oleh</th></tr>
    ${rows.map(p => `<tr><td><code>${esc(p)}</code></td><td class="small">${esc(ver(p))}</td><td>${stat[libStatus(p)]}</td><td class="small">${used[p].slice(0, 5).map(m => `<a href="#/stage/${m.stage}/${m.id.slice(m.stage.length + 1)}">${inline(m.n)}</a>`).join(", ")}${used[p].length > 5 ? ` <span class="muted">+${used[p].length - 5} lagi</span>` : ""}</td></tr>`).join("")}
    </table></div>
    <h2>Tidak ada di bundle: penggantinya</h2>
    <div class="table-wrap"><table><tr><th>Paket</th><th>Pengganti</th><th>Alasan</th></tr>${TK.notInBundle.map(r => `<tr><td><code>${esc(r[0])}</code></td><td>${inline(r[1])}</td><td class="small">${inline(r[2])}</td></tr>`).join("")}</table></div>`;
  }

  // ---------- Hardware ----------
  function hardware() {
    const H = TK.hardware;
    return `<h1>Hardware & Waktu</h1><p class="lead">PC panitia: i7-12700, 32 GB, RTX 3070 LHR. Throughput belum diukur [Belum teruji]; catat dari <code>03_verify_offline.py</code> saat Set-Up.</p>
    <div class="first"><b>Rumus perencanaan:</b> waktu per epoch = jumlah baris ÷ (sampel per detik). Ukur sampel per detik dari 50 batch, lalu kali dua sebagai cadangan. ${slot("optimization/throughput_timer", { link: true })}</div>
    <div class="grid">${H.profiles.map(p => `<div class="card"><h3>${esc(p.n)}</h3>${list(p.rec)}<p class="small" style="margin:0"><b>Hindari:</b> ${inline(p.avoid)}</p></div>`).join("")}</div>
    <h2>Pengaturan transformer</h2><div class="table-wrap"><table><tr><th>Teknik</th><th>Cara</th><th>Catatan</th></tr>${H.transformer.map(r => `<tr><td><b>${esc(r[0])}</b></td><td>${inline(r[1])}</td><td>${inline(r[2])}</td></tr>`).join("")}</table></div>
    <h2>Skala data dan transformer</h2><div class="table-wrap"><table><tr><th>Baris</th><th>Pendekatan awal</th></tr>${H.scale.map(r => `<tr><td>${esc(r[0])}</td><td>${inline(r[1])}</td></tr>`).join("")}</table></div>
    <h2>Fallback</h2><div class="card small">TF-IDF word+char + LR/LinearSVC sudah menghasilkan submission valid sebelum T70. Bila transformer tidak selesai, kirim baseline itu atau blend dengan yang sudah ada.</div>`;
  }

  // ---------- Readiness ----------
  function readiness() {
    return `<h1>Versi & Offline</h1><p class="lead">Kode live coding sering gagal karena argumen lama. Bundle memasang versi di bawah. Pastikan <code>env.bat</code> atau <code>start_jupyter.bat</code> yang dipakai.</p>
    ${codebox(TK.versionCheck, "python")}
    ${TK.readiness.map((g, gi) => `<h2>${esc(g.group)}</h2><div class="card">${g.items.map((c, i) => `<label class="check"><input type="checkbox" data-check="ready-${gi}-${i}"><span>${inline(c)}</span></label>`).join("")}</div>`).join("")}
    <h2>Jebakan versi</h2><p class="small muted">Diperiksa dengan <code>inspect.signature</code> atau dijalankan pada: ${esc(TK.versionTested)}.</p>
    <div class="table-wrap"><table><tr><th>Library</th><th>Kode lama (gagal)</th><th>Pakai ini</th><th>Catatan</th></tr>${TK.versionTraps.map(v => `<tr><td>${esc(v[0])}</td><td>${inline(v[1])}</td><td>${inline(v[2])}</td><td>${inline(v[3])}</td></tr>`).join("")}</table></div>`;
  }

  // ---------- Dataset ----------
  function datasets() {
    const all = TK.catalog || [];
    const C = all.filter(c => !Filter.task || (c.tasks || []).includes(Filter.task));
    const lic = c => c.licenseOk === true ? `<span class="badge good">${esc(c.license)}</span>` : c.licenseOk === false ? `<span class="badge bad">${esc(c.license)}</span>` : `<span class="badge warn">${esc(c.license)}</span>`;
    return `<h1>Katalog Dataset</h1>
    <p class="lead">Untuk latihan dan demonstrasi notebook, bukan data lomba. Pakai saat lomba hanya bila panitia mengizinkan data eksternal. ${TK.catalogNote || ""} Unduh dengan <code>python tools/download_dataset.py &lt;key&gt;</code>.</p>
    ${banner(all.length - C.length, "dataset")}
    <div class="filters" id="dsFilters"><button class="chip on" data-l="">Semua</button><button class="chip" data-l="id">Bahasa Indonesia</button><button class="chip" data-l="en">Inggris</button><button class="chip" data-l="multi">Multibahasa</button><button class="chip" data-l="local">Sudah diunduh</button></div>
    <div class="table-wrap"><table id="dsTable"><tr><th>Dataset</th><th>Task</th><th>Baris</th><th>Kelas</th><th>Lisensi</th><th>Unduhan</th><th>Karakter</th></tr>
    ${C.map(c => `<tr id="a-${c.key}" data-l="${esc(c.lang)}" data-local="${c.local ? 1 : 0}"><td><b>${esc(c.name)}</b><br><a class="small" href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.source)}</a>${c.local ? ' <span class="badge good">lokal</span>' : ""}<br><code class="small">${esc(c.key)}</code></td>
      <td class="small">${(c.tasks || []).map(taskLink).join(", ")}</td><td class="small">${esc(c.rows)}</td><td>${esc(c.classes ?? "–")}</td><td>${lic(c)}</td><td class="small">${fmtBytes(c.bytes)}</td><td class="small">${inline(c.chars)}</td></tr>`).join("")}
    </table></div>
    <p class="small muted"><span class="badge good">Hijau</span> lisensi terbuka tercatat. <span class="badge warn">Kuning</span> lisensi tidak tercantum atau tidak jelas, cek sumber asli sebelum mendistribusikan ulang. <span class="badge bad">Merah</span> nonkomersial, cukup untuk latihan pribadi.</p>`;
  }

  // ---------- Coverage ----------
  function coverage() {
    const lib = L(), ids = Object.keys(TK.slots).filter(slotOk), empty = ids.filter(i => !lib.snippets[i]);
    const unknown = Object.keys(lib.snippets).filter(i => !TK.slots[i]);
    const ts = TK.tasks.filter(t => !Filter.task || t.id === Filter.task);
    return `<h1>Progres</h1><p class="lead">Terisi otomatis dari isi folder saat <code>tools/build_library.py</code> dijalankan${lib.builtAt ? ` (terakhir: ${esc(lib.builtAt)})` : ""}.</p>
    ${banner(0)}
    <h2>Notebook per task</h2>
    <div class="table-wrap"><table><tr><th>task_id</th><th>Task</th><th>Prioritas</th><th>Notebook</th></tr>
    ${ts.slice().sort((a, b) => a.prio.localeCompare(b.prio)).map(t => { const n = notebooksFor(t.id); return `<tr><td><code>${t.id}</code></td><td>${taskLink(t.id)}</td><td><span class="badge ${t.prio}">${t.prio}</span></td><td>${n.length ? n.map(x => `<a href="#/notebooks/${x.id}">${esc(x.title)}</a> ${nbBadge(x)}`).join("<br>") : '<span class="muted">Belum ada</span>'}</td></tr>`; }).join("")}
    </table></div>
    <div class="between"><h2>Slot snippet kosong (${empty.length} dari ${ids.length})</h2><button class="btn" data-copy-text="${esc(empty.map(i => `snippets/${i}.py  # ${TK.slots[i]}`).join("\n"))}">Salin daftar</button></div>
    <div class="card small">${empty.length ? empty.map(i => `<a class="slot" style="margin:0 .25rem .25rem 0" href="#/snippets/${i}" title="${esc(TK.slots[i])}">○ ${esc(i)}</a>`).join("") : "Semua slot terisi."}</div>
    ${unknown.length ? `<h2>Snippet tambahan</h2><div class="card small">${unknown.map(i => `<code>${esc(i)}</code>`).join(" ")}</div>` : ""}`;
  }

  return { cart, home, lomba, mbg, decision, pipeline, stage, tasks, notebooks, snippets, strategy, diagnostics, metrics, models, library, hardware, readiness, datasets, coverage };
})();
