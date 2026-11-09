"""Bangun satu cheatsheet, audit per sumber, dan notebook pendek per keluarga."""
import ast
import json
import statistics
import subprocess
from pathlib import Path
from cheatsheet_lib import content, flows
from cheatsheet_audit import ROOT, audit
from original_flow_lib import original_content


def notebook(flow,cards):
    cells=[{'cell_type':'markdown','source':'# '+flow['title']+'\n\n'+flow['note']+
        '\n\nKode lengkap dari pipeline dan snippet asli. Ikuti urutan tahap dan ubah konfigurasi awal sesuai data. Alternatif tersedia pada cheatsheet utama.', 'metadata':{}}]
    for key in flow['core']:
        card=cards[key]
        cells.append({'cell_type':'markdown','metadata':{},'source':'## '+card['title']+'\n\n'+card['note']})
        cells.append({'cell_type':'code','metadata':{},'source':card['code'],'outputs':[],'execution_count':None})
    for i,cell in enumerate(cells):cell['id']='cheat-'+str(i)
    return {'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
        'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'toolkit':{'title':flow['title']+' · cheatsheet','tasks':flow['tasks'],'kind':'cheatsheet','cards':flow['core']}}}


def build():
    rows=audit(content())
    cards,recipes=original_content()
    for row in rows:
        row['replacements']=['snippet:'+row['id']]
        row['action']='Kode lengkap asli ditampilkan; catatan audit dipertahankan'
    # JS katalog lama dibaca sebagai data. Tidak ada kode Python model yang dijalankan.
    result=subprocess.run(['node',str(ROOT/'tools/export_methods.js')],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    methods=json.loads(result.stdout)
    old={row['id']:row for row in rows}
    fallback={'load':['load'],'eda':['profile','length','vocab','noise'],'prep':['clean'],
        'clean':['profile','target'],'feat':['tfidf'],'split':['split'],'model':['lr'],'bert':['bert_setup'],
        'eval':['evaluate'],'post':['class_bias'],'out':['final'],'output':['final'],'opt':['load'],'imb':['lr']}
    reviewed_methods=[]
    for method in methods:
        source=old.get(method.get('slot'))
        targets=source['replacements'] if source else []
        api='\n'.join(filter(None,[method.get('imp'),method.get('api')]))
        try:ast.parse(api);syntax='valid'
        except SyntaxError:syntax='context_example_not_standalone'
        reviewed_methods.append({'id':method['id'],'title':method['n'],'stage':method['stage'],'for':method['for'],
            'source_slot':method.get('slot'),'action':'Kode lengkap asli dan catatan kondisi pemakaian',
            'replacements':targets,'note':source['note'] if source else 'Contoh API ilustratif, bukan sel mandiri. Gunakan input dan langkah eksplisit kartu terkait.',
            'api_syntax':syntax,'old_lines':len(api.splitlines())})
    folder=ROOT/'notebooks/cheatsheet';folder.mkdir(parents=True,exist_ok=True)
    for flow in recipes:
        assert all(key in cards for key in flow['core']+flow['extra']),flow['id']
        book=notebook(flow,cards)
        ast.parse('\n'.join(cards[key]['code'] for key in flow['core']))
        (folder/(flow['id']+'.ipynb')).write_text(json.dumps(book,ensure_ascii=False,indent=1),encoding='utf-8')
    lengths=[len(c['code'].splitlines()) for c in cards.values()]
    data={'cards':cards,'flows':recipes,'audit':{'snippets':rows,'methods':reviewed_methods},
        'summary':{'snippets':len(rows),'methods':len(methods),'flows':len(recipes),
            'tasks':len({t for f in recipes for t in f['tasks']}),'cards':len(cards),
            'median_lines':statistics.median(lengths),'max_lines':max(lengths),
            'priority_note':'Urutan persiapan berdasarkan karakter data teks dan skenario MBG yang sudah dicatat. Bukan probabilitas/fakta soal panitia.'}}
    gen=ROOT/'assets/js/generated/cheatsheet.js'
    gen.write_text('window.TK=window.TK||{};\nTK.cheatsheet='+json.dumps(data,ensure_ascii=False)+';\n',encoding='utf-8')
    (ROOT/'references/cheatsheet_audit.json').write_text(json.dumps(data['audit'],ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Audit cheatsheet offline','',data['summary']['priority_note'],'',
        f"Ditinjau: {len(rows)} file snippet dan {len(methods)} saran metode; {data['summary']['tasks']} task dalam {len(recipes)} alur.",
        f"Kode lengkap: {len(cards)} bagian dari pipeline/snippet asli. Versi kode pendek tidak digunakan pada tampilan utama.",'',
        'Audit struktur/kondisi tidak membuktikan skor model pada data baru. Hasil uji perilaku dipisah dalam `toolkit/references/cheatsheet_validation.json`.',
        '', '## Urutan audit task','', '| Urutan | Alur | Task | Baris kode inti |', '|---|---|---|---|']
    for f in recipes:lines.append(f"| {f['priority']} | {f['title']} | {', '.join(f['tasks'])} | {sum(len(cards[k]['code'].splitlines()) for k in f['core'])} |")
    lines+=['','## Temuan dan keputusan per file','','| Sumber | Baris lama | Kartu pengganti | Keputusan / kondisi |','|---|---|---|---|']
    for row in rows:lines.append(f"| `{row['id']}` | {row['lines']} | {', '.join(row['replacements'])} | {row['note']} |")
    lines+=['','## Saran metode satu per satu','','| Metode | Rujukan | Kartu pendek | Catatan |','|---|---|---|---|']
    for row in reviewed_methods:lines.append(f"| {row['title'].replace('|','/')} | {row['source_slot'] or row['stage']} | {', '.join(row['replacements'])} | {row['note']} |")
    (ROOT.parent/'docs/AUDIT_CHEATSHEET.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(data['summary']))
    return data


if __name__=='__main__':build()
