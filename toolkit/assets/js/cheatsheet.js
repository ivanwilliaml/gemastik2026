// Satu alur: baca, ketik, jalankan. Tidak ada keranjang/dependency injector/config JSON.
(function () {
  const data=()=>TK.cheatsheet, esc=R.esc;
  const forTask=task=>data().flows.find(f=>f.tasks.includes(task)||f.id===task);
  const cardOf=id=>data().cards[id];
  const stageNames={load:'Data',eda:'EDA angka',prep:'Preprocessing',split:'Validasi',feat:'Representasi',model:'Model',eval:'Evaluasi',post:'Post-processing',out:'Output'};
  function taskLink(flow) {
    return `<button class="chip" data-set-task="${esc(flow.tasks[0])}" data-go="#/pipeline">${esc(flow.title)}</button>`;
  }
  function card(id,index,core=false) {
    const c=cardOf(id);if(!c)return '';
    const prereq=c.needs.map(k=>cardOf(k)?.title||k);
    const body=`<div class="cheat-note">${esc(c.note)}</div>${R.codebox(c.code,'python')}`;
    const head=`<span class="cheat-number">${index==null?'＋':index+1}</span><b>${esc(c.title)}</b><span class="small muted">${c.code.trimEnd().split('\n').length} baris</span>`;
    const condition=`${c.replaces?`<p class="small"><b>Ganti:</b> ${esc(cardOf(c.replaces)?.title||c.replaces)}. ${esc(c.when==='Wajib'?'Gunakan hanya pada kondisi yang tertulis.':c.when)}</p>`:''}
      ${!core&&prereq.length?`<p class="small muted"><b>Perlu:</b> ${esc(prereq.join(' → '))}. Ketik bagian yang belum ada, bukan mengulang semua sel.</p>`:''}`;
    if(core)return `<section class="card cheat-cell" id="cheat-${esc(id)}"><div class="between">${head}</div>${condition}${body}</section>`;
    return `<details class="stagecell cheat-cell" id="cheat-${esc(id)}"><summary>${head}</summary>${condition}${body}</details>`;
  }
  function select() {
    return `<h1>Cheatsheet per task</h1><p class="lead">Pilih bentuk soal, lalu ikuti tahap dari atas. Kode lengkap asli tersedia beserta fungsi pendukungnya; referensi alternatif ada di bawah.</p>
      <p class="small muted">${esc(data().summary.priority_note)}</p>
      <div class="grid">${data().flows.map(f=>`<article class="card"><span class="badge">Urutan persiapan ${f.priority}</span><h3>${esc(f.title)}</h3><p class="small">${esc(f.note)}</p>${taskLink(f)}</article>`).join('')}</div>`;
  }
  function pipeline(sub) {
    const flow=forTask(sub||Filter.task);
    if(!flow)return select();
    const core=flow.core;
    const extra=flow.extra.filter(k=>!(Filter.noPre&&cardOf(k).pre));
    return `<div class="between"><h1>${esc(flow.title)}</h1><button class="btn" data-cheat-back>Pilih task lain</button></div>
      <p class="lead">${esc(flow.note)}</p><p class="small">${core.length} tahap · kode lengkap dari <code>${esc(flow.pipeline_source)}</code>. Fungsi dan dependensi disertakan sesuai sumber asli.</p>
      <div class="row"><button class="btn" data-cheat-notebook="${esc(flow.id)}">Unduh notebook lengkap</button><button class="btn" data-cheat-print>Cetak cheatsheet ini</button><a class="btn" href="#/audit">Catatan audit</a></div>
      <div class="card cheat-contract">Ubah file, kolom, model dan saklar eksperimen pada sel Setup. Nama variabel mengikuti pipeline asli. Bila pretrained tidak dipakai, matikan saklar pada Setup; sel inisialisasi tetap dijalankan agar tahap sesudahnya memiliki variabel yang diperlukan.</div>
      <h2>Ikuti urutan ini</h2>${core.map((key,i)=>card(key,i,true)).join('')}
      ${extra.length?`<h2>Referensi kode lengkap sesuai kebutuhan</h2><p class="small">Snippet asli beserta dependensinya. Ini referensi alternatif, bukan tahap tambahan yang semuanya wajib dijalankan.</p>
      ${Object.entries(stageNames).map(([stage,title])=>{const items=extra.filter(k=>cardOf(k).stage===stage);return items.length?`<h3>${title}</h3>${items.map(k=>card(k,null)).join('')}`:'';}).join('')}`:''}`;
  }
  function stages(stage,method) {
    const legacy=data().audit.methods.find(m=>m.id===stage+'-'+method || m.id===stage+'/'+method || m.id===method);
    if(legacy)return `<h1>${esc(legacy.title)}</h1><p>${esc(legacy.note)}</p>${legacy.replacements.map(k=>card(k,null)).join('')}<a class="btn" href="#/pipeline">Kembali ke alur task</a>`;
    const map={clean:'eda',imb:'model',opt:'model',train:'model',bert:'model',output:'out'};
    const target=map[stage]||stage;
    const flow=forTask(Filter.task), allowed=flow ? [...flow.core,...flow.extra] : Object.keys(data().cards);
    const keys=[...new Set(allowed)].filter(k=>cardOf(k).stage===target&&!(Filter.noPre&&cardOf(k).pre));
    return `<h1>${esc(stageNames[target]||'Potongan kode')}</h1><p class="lead">${flow?esc(flow.title):'Pilih task di atas untuk menyaring.'} Kode lengkap yang sama dipakai pada alur utama.</p>${keys.map(k=>card(k,null)).join('')}`;
  }
  function snippets(sub,sub2) {
    const key=sub2?sub+'/'+sub2:sub;
    const row=data().audit.snippets.find(r=>r.id===key);
    if(row)return `<h1>${esc(row.id)}</h1><p>${esc(row.note)}</p><p class="small">${esc(row.action)}.</p>${row.replacements.map(k=>card(k,null)).join('')}<a class="btn" href="#/pipeline">Buka dalam urutan task</a>`;
    const flow=forTask(Filter.task),allowed=flow?[...flow.core,...flow.extra]:Object.keys(data().cards);
    return `<h1>Potongan kode</h1><p class="lead">Satu koleksi dengan alur per task. ${flow?esc(flow.title):'Pilih task supaya variabel dan urutannya mudah diikuti.'}</p>${Object.entries(stageNames).map(([key,name])=>`<h2>${name}</h2>${[...new Set(allowed)].filter(k=>cardOf(k).stage===key&&!(Filter.noPre&&cardOf(k).pre)).map(k=>card(k,null)).join('')}`).join('')}`;
  }
  function auditPage() {
    const s=data().summary;
    return `<h1>Audit kode satu per satu</h1><p class="lead">${s.snippets} file snippet dan ${s.methods} saran metode ditinjau. Kode lengkap asli digunakan kembali; seluruh ${s.tasks} task masuk ${s.flows} alur.</p>
      <p>Ini audit struktur dan kondisi pemakaian, bukan janji menang pada data yang belum diketahui. Uji perilaku Kaggle dicatat terpisah.</p>
      <h2>Urutan task</h2><div class="row">${data().flows.map(taskLink).join('')}</div>
      <h2>Temuan per file</h2>${data().audit.snippets.map(r=>`<details class="stagecell"><summary><code>${esc(r.id)}</code> · ${r.lines} baris lama</summary><p>${esc(r.note)}</p><p class="small">${esc(r.verification)}</p><a class="btn mini" href="#/snippets/${esc(r.id)}">Lihat kode pengganti</a></details>`).join('')}
      <h2>Saran metode</h2>${data().audit.methods.map(r=>`<details class="stagecell"><summary>${esc(r.title)}</summary><p>${esc(r.note)}</p><p class="small">Kode rekomendasi: ${esc(r.replacements.join(', '))}. Contoh API lama ${esc(r.api_syntax)}.</p></details>`).join('')}`;
  }
  function notebook(flow) {
    const cells=[{cell_type:'markdown',metadata:{},source:'# '+flow.title+'\n\n'+flow.note}];
    flow.core.forEach(key=>{const c=cardOf(key);cells.push({cell_type:'markdown',metadata:{},source:'## '+c.title+'\n\n'+c.note},
      {cell_type:'code',metadata:{},source:c.code,outputs:[],execution_count:null});});
    cells.forEach((c,i)=>c.id='cheat-'+i);
    return {nbformat:4,nbformat_minor:5,cells,metadata:{kernelspec:{name:'python3',display_name:'Python 3',language:'python'}}};
  }
  document.addEventListener('click',e=>{
    const back=e.target.closest('[data-cheat-back]');if(back){location.hash='#/pipeline';document.getElementById('taskSel').value='';document.getElementById('taskSel').dispatchEvent(new Event('change'));return;}
    const save=e.target.closest('[data-cheat-notebook]');if(save){const flow=data().flows.find(f=>f.id===save.dataset.cheatNotebook),url=URL.createObjectURL(new Blob([JSON.stringify(notebook(flow),null,1)],{type:'application/json'}));
      const a=document.createElement('a');a.href=url;a.download=flow.id+'_cheatsheet.ipynb';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),3000);}
    if(e.target.closest('[data-cheat-print]')){document.querySelectorAll('.cheat-cell').forEach(el=>{if(el.tagName==='DETAILS')el.open=true;});window.print();}
  });
  Pages.pipeline=pipeline;Pages.compose=pipeline;Pages.cart=pipeline;Pages.stage=stages;Pages.snippets=snippets;Pages.audit=auditPage;
  window.CheatSheet={pipeline,stages,snippets,notebook,forTask};
})();
