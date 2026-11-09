// Helper tampilan: escape, inline markup, highlight Python, markdown, notebook.
window.R = (function () {
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const slug = s => String(s).toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

  // `kode` dan **tebal**
  function inline(s) {
    return String(s ?? "").split(/(`[^`]+`)/).map(p => {
      if (p.length > 2 && p[0] === "`" && p[p.length - 1] === "`") return "<code>" + esc(p.slice(1, -1)) + "</code>";
      return esc(p)
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    }).join("");
  }

  const KW = new Set("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(" "));
  function python(code) {
    const re = /("""[\s\S]*?"""|'''[\s\S]*?'''|[rRbBfF]{0,2}"(?:\\.|[^"\\\n])*"|[rRbBfF]{0,2}'(?:\\.|[^'\\\n])*')|(#[^\n]*)|(\b\d[\d_.eE]*\b)|([A-Za-z_][A-Za-z0-9_]*)/g;
    let out = "", last = 0, m, afterDef = false;
    while ((m = re.exec(code))) {
      out += esc(code.slice(last, m.index));
      last = re.lastIndex;
      if (m[1]) out += '<span class="tk-str">' + esc(m[1]) + "</span>";
      else if (m[2]) out += '<span class="tk-com">' + esc(m[2]) + "</span>";
      else if (m[3]) out += '<span class="tk-num">' + esc(m[3]) + "</span>";
      else {
        const w = m[4];
        if (KW.has(w)) { out += '<span class="tk-kw">' + w + "</span>"; afterDef = w === "def" || w === "class"; continue; }
        out += afterDef ? '<span class="tk-fn">' + w + "</span>" : esc(w);
      }
      afterDef = false;
    }
    return out + esc(code.slice(last));
  }

  function codebox(code, lang) {
    const body = lang === "python" ? python(code) : esc(code);
    return '<div class="codebox"><button class="copy" data-copy>Salin</button><pre><code data-raw>' + body + "</code></pre></div>";
  }

  function markdown(src) {
    const lines = String(src).replace(/\r/g, "").split("\n");
    let out = "", i = 0;
    const isRow = l => /^\s*\|.*\|\s*$/.test(l);
    while (i < lines.length) {
      const l = lines[i];
      const fence = l.match(/^\s*```\s*(\w*)/);
      if (fence) {
        const buf = []; i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
        i++; out += codebox(buf.join("\n"), fence[1] === "python" || fence[1] === "py" ? "python" : ""); continue;
      }
      const h = l.match(/^(#{1,6})\s+(.*)/);
      if (h) { const n = Math.min(h[1].length + 1, 4); out += `<h${n} id="h-${slug(h[2])}">${inline(h[2])}</h${n}>`; i++; continue; }
      if (/^\s*(-{3,}|\*{3,})\s*$/.test(l)) { out += "<hr>"; i++; continue; }
      if (isRow(l) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
        const cells = r => r.trim().replace(/^\||\|$/g, "").split("|").map(c => inline(c.trim()));
        let t = "<tr>" + cells(l).map(c => "<th>" + c + "</th>").join("") + "</tr>"; i += 2;
        while (i < lines.length && isRow(lines[i])) t += "<tr>" + cells(lines[i++]).map(c => "<td>" + c + "</td>").join("") + "</tr>";
        out += '<div class="table-wrap"><table>' + t + "</table></div>"; continue;
      }
      if (/^\s*([-*+]|\d+\.)\s+/.test(l)) {
        const ordered = /^\s*\d+\./.test(l); let items = "";
        while (i < lines.length && /^\s*([-*+]|\d+\.)\s+/.test(lines[i])) items += "<li>" + inline(lines[i++].replace(/^\s*([-*+]|\d+\.)\s+/, "")) + "</li>";
        out += ordered ? "<ol>" + items + "</ol>" : "<ul>" + items + "</ul>"; continue;
      }
      if (/^\s*>/.test(l)) {
        const buf = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ""));
        out += '<div class="first">' + inline(buf.join(" ")) + "</div>"; continue;
      }
      if (!l.trim()) { i++; continue; }
      const buf = [];
      while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|\s*```|\s*([-*+]|\d+\.)\s|\s*>)/.test(lines[i]) && !isRow(lines[i])) buf.push(lines[i++]);
      if (!buf.length) { out += "<p>" + inline(lines[i++]) + "</p>"; continue; }
      out += "<p>" + inline(buf.join(" ")) + "</p>";
    }
    return out;
  }

  // Hanya tabel dari output HTML (mis. pandas) yang dirender; selainnya dibuang.
  function safeTable(html) {
    try {
      const doc = new DOMParser().parseFromString(html, "text/html");
      const t = doc.querySelector("table");
      if (!t) return null;
      const copy = n => {
        if (n.nodeType === 3) return esc(n.textContent);
        if (n.nodeType !== 1) return "";
        const tag = n.tagName.toLowerCase();
        const inner = Array.from(n.childNodes).map(copy).join("");
        return ["table", "thead", "tbody", "tr", "th", "td"].includes(tag) ? `<${tag}>${inner}</${tag}>` : inner;
      };
      return copy(t);
    } catch (e) { return null; }
  }

  const src = v => Array.isArray(v) ? v.join("") : String(v ?? "");
  const stripAnsi = s => s.replace(/\u001b\[[0-9;]*[A-Za-z]/g, "");

  function nbOutput(o) {
    if (o.output_type === "stream") return `<div class="output"><pre>${esc(stripAnsi(src(o.text)))}</pre></div>`;
    if (o.output_type === "error") return `<div class="output err"><pre>${esc(stripAnsi((o.traceback || [o.ename + ": " + o.evalue]).join("\n")))}</pre></div>`;
    const d = o.data || {};
    if (d["image/png"]) return `<div class="output"><img alt="Output gambar" src="data:image/png;base64,${esc(src(d["image/png"]).replace(/\s/g, ""))}"></div>`;
    if (d["text/html"]) { const t = safeTable(src(d["text/html"])); if (t) return `<div class="output"><table>${t.replace(/^<table>|<\/table>$/g, "")}</table></div>`; }
    if (d["text/markdown"]) return `<div class="output">${markdown(src(d["text/markdown"]))}</div>`;
    if (d["text/plain"]) return `<div class="output"><pre>${esc(src(d["text/plain"]))}</pre></div>`;
    return "";
  }

  // Mengembalikan {html, toc}
  function notebook(nb) {
    const toc = []; let html = "";
    (nb.cells || []).forEach((c, idx) => {
      const s = src(c.source);
      if (c.cell_type === "markdown") {
        s.split("\n").forEach(l => { const m = l.match(/^(#{1,3})\s+(.*)/); if (m) toc.push({ level: m[1].length, text: m[2].replace(/[`*]/g, ""), id: "h-" + slug(m[2]) }); });
        html += `<div class="cell md">${markdown(s)}</div>`;
      } else if (c.cell_type === "code") {
        html += `<div class="cell"><span class="ec">In [${c.execution_count ?? " "}]</span>${codebox(s, "python")}${(c.outputs || []).map(nbOutput).join("")}</div>`;
      }
    });
    return { html, toc };
  }

  return { esc, slug, inline, python, codebox, markdown, notebook, src };
})();
