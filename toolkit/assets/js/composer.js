// Perakit v2: sumber dan dependency graph yang sama dengan tools/compose.py.
(function () {
  const L = () => TK.composer;
  const memory = {};
  const read = (key, fallback) => {
    try { return JSON.parse(localStorage.getItem('nlptk:compose:' + key)) ?? fallback; }
    catch (_) { return fallback; }
  };
  const write = (key, value) => { try { localStorage.setItem('nlptk:compose:' + key, JSON.stringify(value)); } catch (_) {} };
  function resolve(ids) {
    const done = [], active = new Set();
    function visit(id) {
      if (active.has(id)) throw Error('Siklus dependensi: ' + id);
      if (done.includes(id)) return;
      const c = L().components[id];
      if (!c) throw Error('Komponen tidak tersedia: ' + id);
      active.add(id);c.needs.forEach(visit);active.delete(id);done.push(id);
    }
    ids.forEach(visit);return done;
  }
  function taskFor(task) {
    return Object.keys(L().recipes).find(id => id === task || L().recipes[id].aliases.includes(task));
  }
  function state(task) {
    if (!memory[task]) {
      const r = L().recipes[task], saved = read(task, null);
      memory[task] = saved || {selected:r.defaults.slice(), config:JSON.stringify(r.config,null,2), install:true};
      memory[task].selected = memory[task].selected.filter(k => r.choices.includes(k));
    }
    return memory[task];
  }
  function python(value, depth=0) {
    if (value === null) return 'None';
    if (typeof value === 'boolean') return value ? 'True' : 'False';
    if (Array.isArray(value)) return '[' + value.map(v => python(v,depth+1)).join(', ') + ']';
    if (typeof value === 'object') {
      const entries = Object.entries(value);
      return entries.length ? '{\n' + entries.map(([k,v]) => '  '.repeat(depth+1)+JSON.stringify(k)+': '+python(v,depth+1)).join(',\n') + '\n'+'  '.repeat(depth)+'}' : '{}';
    }
    return JSON.stringify(value);
  }
  function assemble(task) {
    const r = L().recipes[task], s = state(task), components=L().components;
    if (!s.selected.length) throw Error('Pilih minimal satu model.');
    if(s.selected.some(k=>!r.choices.includes(k)))throw Error('Pilihan komponen tidak cocok dengan kasus.');
    const chain = resolve([...r.required,...s.selected]);
    const models = chain.map(k=>components[k].model).filter(Boolean);
    if (!models.length) throw Error('Pilih minimal satu model, selain pilihan sentimen.');
    const overrides = JSON.parse(s.config);
    if (!overrides || Array.isArray(overrides) || typeof overrides !== 'object') throw Error('Konfigurasi harus objek JSON.');
    const cfg = {...r.config,...overrides,task,models,sentiment_pretrained:s.selected.includes('sentiment.pretrained')};
    const packages = [...new Set(chain.flatMap(k=>components[k].packages))].sort();
    if(task==='multilabel')packages.push('iterative-stratification>=0.1,<1');
    if (cfg.eda_languages) packages.push('langid==1.1.6');
    const cells = [];
    const md = text => cells.push({cell_type:'markdown',metadata:{},source:text});
    const code = text => cells.push({cell_type:'code',metadata:{},execution_count:null,outputs:[],source:text});
    md('# '+r.title+' · toolkit modular v2\n\n'+r.note+'\n\nAdaptasi: '+r.source+'. Skor sumber tidak diwariskan; jalankan di Kaggle. Ubah konfigurasi sekali, pilih model lewat perakit.');
    md('## 1 · Konfigurasi');code('CFG = '+python(cfg));
    if (s.install) {
      md('## 2 · Lingkungan Kaggle\n\nDependensi hanya untuk blok terpilih. Matikan instalasi pada perakit untuk bundle offline.');
      code("import os, sys, subprocess\nfrom pathlib import Path\nassert Path('/kaggle/input').exists(), 'Jalankan di Kaggle.'\nos.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', "+packages.map(p=>JSON.stringify(p)).join(', ')+"], check=True)");
    }
    md('## 3 · Fungsi komponen\n\nDefinisi sekali; pemakaian ringkas di bawah. Tidak memerlukan folder toolkit saat notebook dipindahkan.');
    chain.forEach(key=>{const c=components[key];if(c.code.trim()){md('### '+c.title+'\n\n'+c.note);code(c.code);}});
    md('## 4 · Jalankan\n\nPemilihan menggunakan validation; target holdout baru dibaca setelah policy dibekukan.');
    code('require_kaggle(gpu='+python(chain.some(k=>components[k].gpu))+ ")\nseed_all(CFG['seed'])");
    L().steps[task].forEach(([name,source])=>{md('### '+name);code(source);});
    cells.forEach((c,i)=>{c.id='tk-'+i;});
    return {nbformat:4,nbformat_minor:5,cells,metadata:{kernelspec:{name:'python3',language:'python',display_name:'Python 3'},
      toolkit:{version:2,task,tasks:r.aliases,selected:s.selected,resolved:chain,provenance:L().provenance[task]}}};
  }
  function download(name,text) {
    const url=URL.createObjectURL(new Blob([text],{type:'application/json'})), a=document.createElement('a');
    a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);
  }
  async function copy(text) {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else { const el=document.createElement('textarea');el.value=text;document.body.appendChild(el);el.select();document.execCommand('copy');el.remove(); }
  }
  function render(task) {
    const esc=R.esc, requested=taskFor(task || Filter.task || '') || read('lastTask','classification');
    const id=L().recipes[requested] ? requested : 'classification';write('lastTask',id);
    const r=L().recipes[id],s=state(id),all=L().components;
    let book,error='';try { book=assemble(id); } catch(e) {error=e.message;}
    const chain=book ? book.metadata.toolkit.resolved : resolve(r.required);
    const auto=chain.filter(k=>!s.selected.includes(k) && all[k].model);
    return `<div id="composerRoot" data-task="${esc(id)}"><h1>Rakit kode · v2</h1>
      <p class="lead">Pilih kasus, centang model, ubah file/kolom sekali. Fungsi dan dependensi ikut otomatis. Unduh notebook utuh atau salin satu komponen.</p>
      <div class="row">${Object.values(L().recipes).map(p=>`<a class="chip ${id===p.id?'on':''}" href="#/compose/${p.id}">${esc(p.title)}</a>`).join('')}</div>
      <p class="small" style="margin-top:.8rem">${esc(r.note)} Sumber: <code>${esc(r.source)}</code>.</p>
      <div class="row"><button class="btn" data-compose-preset="default">Pilihan lengkap</button><button class="btn" data-compose-preset="fast">Cepat tanpa pretrained</button>${['classification','multilabel','nli','similarity'].includes(id)?'<button class="btn" data-compose-preset="pretrained">Lima pretrained</button>':''}</div>
      <h2>1 · Pasang / lepas model</h2><div class="grid">${r.choices.filter(k=>!Filter.noPre || !all[k].gpu).map(k=>`<label class="card"><input type="checkbox" data-compose-choice="${esc(k)}" ${s.selected.includes(k)?'checked':''}> <b>${esc(all[k].title)}</b><span class="small muted" style="display:block">${all[k].gpu?'Kaggle GPU · ':''}${esc(all[k].note||'Input/output mengikuti kontrak yang sama.')}</span></label>`).join('')}</div>
      ${auto.length?`<p class="small">Model prasyarat ikut aktif: ${auto.map(k=>esc(all[k].title)).join(', ')}.</p>`:''}
      <p class="small muted">${chain.length} komponen termasuk helper. Melepas komponen induk akan melepas pilihan yang membutuhkannya. Filter “Tanpa pretrained” juga menghapus pilihan GPU yang aktif.</p>
      <h2>2 · Konfigurasi</h2><p class="small">Edit file, kolom, group_col, normalisasi dan batas eksperimen. Nilai <code>models</code> mengikuti centang di atas. Bahasa: set <code>eda_languages: true</code>.</p>
      <textarea id="composeConfig" spellcheck="false" style="width:100%;min-height:270px;font-family:var(--mono);font-size:13px">${esc(s.config)}</textarea>
      <div class="row"><button class="btn" data-compose-apply>Terapkan konfigurasi</button><label><input type="checkbox" id="composeInstall" ${s.install?'checked':''}> Sertakan instalasi paket</label></div>
      ${error?`<p role="alert" class="badge bad">${esc(error)}</p>`:''}
      <h2>3 · Ambil kode</h2><div class="row"><button class="btn primary" data-compose-download ${error?'disabled':''}>Unduh .ipynb</button><button class="btn" data-compose-copy ${error?'disabled':''}>Salin semua kode</button></div>
      <p class="small muted">Notebook sumber sudah direview di Kaggle. Rakitan ini adalah adaptasi: hasil skor harus dihitung ulang. <span id="composeMessage" role="status"></span></p>
      <h2>Sel pemakaian</h2>${L().steps[id].map(([name,source])=>`<details class="stagecell"><summary>${esc(name)}</summary>${R.codebox(source,'python')}</details>`).join('')}
      <h2>Komponen & dependensi</h2>${chain.filter(k=>all[k].code.trim()).map(k=>`<details class="stagecell"><summary>${esc(all[k].title)} · ${all[k].code.split('\n').length} baris</summary><p class="small">${esc(all[k].note)}</p><button class="btn mini" data-compose-block="${esc(k)}">Salin fungsi + dependensi</button>${all[k].example?R.codebox(all[k].example,'python'):''}${R.codebox(all[k].code,'python')}</details>`).join('')}
      </div>`;
  }
  function repaint(id) {document.getElementById('content').innerHTML=render(id);}
  function saveEditor(id) {
    const s=state(id),el=document.getElementById('composeConfig');if(el)s.config=el.value;
    const install=document.getElementById('composeInstall');if(install)s.install=install.checked;
    write(id,s);
  }
  function removeDependents(id, removed) {
    const s=state(id);
    s.selected=s.selected.filter(key=>!resolve([key]).includes(removed));
  }
  document.addEventListener('change',e=>{
    const root=document.getElementById('composerRoot');if(!root)return;
    const id=root.dataset.task,s=state(id);
    if(e.target.matches('[data-compose-choice]')) {
      saveEditor(id);const key=e.target.dataset.composeChoice;
      if(e.target.checked) {if(!s.selected.includes(key))s.selected.push(key);} else removeDependents(id,key);
      write(id,s);repaint(id);
    }
    if(e.target.id==='composeInstall')saveEditor(id);
  });
  document.addEventListener('click',async e=>{
    const btn=e.target.closest('[data-compose-preset],[data-compose-apply],[data-compose-download],[data-compose-copy],[data-compose-block]');
    if(!btn)return;
    const root=document.getElementById('composerRoot'),id=root.dataset.task,s=state(id),r=L().recipes[id];
    saveEditor(id);
    try {
      if(btn.dataset.composePreset) {
        const preset=btn.dataset.composePreset;
        s.selected=preset==='default'?r.defaults.slice():preset==='fast'?r.choices.filter(k=>!L().components[k].gpu && !k.startsWith('sentiment.')):
          ['indobertweet','indobert_p1','xlmr','indolem','nusabert'].map(k=>'model.'+k);
        write(id,s);
        if((preset==='default' || preset==='pretrained') && Filter.noPre) {
          const toggle=document.getElementById('noPre');toggle.checked=false;toggle.dispatchEvent(new Event('change'));
        } else repaint(id);
        return;
      }
      if(btn.hasAttribute('data-compose-apply')) {JSON.parse(s.config);write(id,s);repaint(id);return;}
      if(btn.dataset.composeBlock) {
        const ids=resolve([btn.dataset.composeBlock]);
        await copy(ids.map(k=>L().components[k].code).filter(Boolean).join('\n\n'));
      } else {
        const book=assemble(id);
        if(btn.hasAttribute('data-compose-download'))download(id+'_modular.ipynb',JSON.stringify(book,null,1));
        else await copy(book.cells.filter(c=>c.cell_type==='code').map(c=>c.source).join('\n\n'));
      }
      document.getElementById('composeMessage').textContent='Kode siap.';
    } catch(error) {document.getElementById('composeMessage').textContent=error.message;}
  });
  function noPreChanged(enabled) {
    if(!enabled)return;
    Object.keys(L().recipes).forEach(id=>{const s=state(id);s.selected=s.selected.filter(k=>!resolve([k]).some(d=>L().components[d].gpu));write(id,s);});
  }
  window.Composer={render,resolve,assemble,taskFor,noPreChanged,state};
  Pages.compose=render;
})();
