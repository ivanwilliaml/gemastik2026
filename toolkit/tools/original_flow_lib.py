"""Kode pipeline/snippet lengkap, ditampilkan dalam navigasi task terpadu."""
import re
from pipelines_lib import all_pipelines, ROOT, snippet_body, needs_of, add_chain
from cheatsheet_lib import flows


def original_content():
    pipelines=all_pipelines();cards={};recipes=flows()
    for file in sorted((ROOT/'snippets').rglob('*.py')):
        slot=file.relative_to(ROOT/'snippets').with_suffix('').as_posix()
        source=file.read_text(encoding='utf-8')
        fields=dict(re.findall(r'^# (title|when|avoid|tags):\s*(.*)$',source,re.M))
        chain=add_chain(slot,[])
        code='\n\n'.join(snippet_body(key) for key in chain)+'\n'
        category=slot.split('/')[0]
        stage={'data_loading':'load','eda':'eda','preprocessing':'prep','validation':'split',
            'feature_engineering':'feat','evaluation':'eval','postprocessing':'post','output':'out'}.get(category,'model')
        cards['snippet:'+slot]={'id':'snippet:'+slot,'title':fields.get('title',slot),'stage':stage,
            'code':code,'needs':[],'note':fields.get('when','')+' '+fields.get('avoid',''),
            'when':'Sesuai kondisi data','mode':'referensi','replaces':'','pre':bool(re.search(r'from (transformers|sentence_transformers)',code)),'source':'snippets/'+slot+'.py'}
    for flow in recipes:
        task=flow['tasks'][0]
        name,pipeline=next((name,p) for name,p in pipelines.items() if task in p['tasks'])
        flow['core']=[]
        slots=[]
        for stage in pipeline['stages']:
            key='pipeline:'+name+':'+stage['id']
            cards[key]={'id':key,'title':stage['name'],'stage':{'bert':'model','clean':'prep','output':'out'}.get(stage['id'],stage['id']),
                'code':stage['code'],'needs':[],'note':stage['note'],'when':'Jalankan berurutan','mode':'inti','replaces':'',
                'pre':stage['id']=='bert','source':'pipelines/'+name+'.py'}
            flow['core'].append(key);slots.extend(stage['uses'])
        flow['extra']=list(dict.fromkeys('snippet:'+key for key in slots))
        flow['pipeline_source']='pipelines/'+name+'.py'
    return cards,recipes
