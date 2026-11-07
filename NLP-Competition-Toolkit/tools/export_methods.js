// Cetak katalog metode sebagai JSON, untuk validate_api.py, dan cek referensi silang ke stderr.
const fs = require("fs"), path = require("path"), vm = require("vm");
const root = path.join(__dirname, "..", "assets", "js");
const ctx = { console }; ctx.window = ctx; vm.createContext(ctx);
for (const f of ["slots", "tasks", "strategy", "filters", "methods_a", "methods_b", "methods_c", "toolbox", "guide", "lomba"])
  vm.runInContext(fs.readFileSync(path.join(root, "content", f + ".js"), "utf8"), ctx);
vm.runInContext(fs.readFileSync(path.join(root, "generated", "catalog.js"), "utf8"), ctx);
const T = ctx.TK, err = [];
const dup = T.methods.map(m => m.id).filter((x, i, a) => a.indexOf(x) !== i);
if (dup.length) err.push("id metode ganda: " + dup);
for (const m of T.methods) {
  if (!T.stages.some(s => s.id === m.stage)) err.push("stage? " + m.id);
  if (m.slot && !T.slots[m.slot]) err.push("slot? " + m.id + " " + m.slot);
  if (T.badTokens(m.for).length) err.push("for? " + m.id + " " + T.badTokens(m.for));
}
const check = (name, s) => { const b = T.badTokens(s); if (b.length) err.push(name + " for? " + b); };
Object.entries(T.slotFor).forEach(([k, v]) => { if (!T.slots[k]) err.push("slotFor tanpa slot: " + k); check(k, v); });
Object.entries(T.strategyFor).forEach(([k, v]) => check(k, v));
Object.entries(T.diagFor).forEach(([k, v]) => check(k, v));
T.metrics.forEach(g => { check(g.group, g.for); g.items.forEach(i => i.for && check(i.n, i.for)); if (!T.slots[g.slot]) err.push("metric slot? " + g.slot); });
T.bundleModels.forEach(m => check(m[2], m[5]));
const keys = new Set(T.catalog.map(c => c.key)), ids = new Set(T.tasks.map(t => t.id));
T.tasks.forEach(t => { t.datasets.forEach(d => keys.has(d) || err.push("dataset? " + t.id + ":" + d)); t.slots.forEach(s => T.slots[s] || err.push("slot? " + t.id + ":" + s)); });
T.catalog.forEach(c => c.tasks.forEach(t => ids.has(t) || err.push("task? " + c.key + ":" + t)));
T.quickRef.forEach(r => ids.has(r[1]) || err.push("quickRef? " + r[1]));
T.lomba.taskProb.forEach(r => ids.has(r[0]) || err.push("taskProb? " + r[0]));
(function w(n) { if (n.leaf) ids.has(n.task) || err.push("leaf? " + n.task); else n.opts.forEach(o => w(o.to)); })(T.decisionTree);
// Setiap task harus punya metode di tahap inti.
for (const t of T.tasks) for (const st of ["load", "eda", "split", "model", "eval"]) {
  const n = T.methods.filter(m => m.stage === st && T.applies(m.for, t.id)).length;
  if (!n) err.push("task " + t.id + " tanpa metode di " + st);
}
console.error("metode:", T.methods.length, "task:", T.tasks.length, "slot:", Object.keys(T.slots).length, "dataset:", T.catalog.length);
console.error(err.length ? "MASALAH:\n" + err.join("\n") : "referensi silang OK");
process.stdout.write(JSON.stringify(T.methods));
if (err.length) process.exitCode = 1;
