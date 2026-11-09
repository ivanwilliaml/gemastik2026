"""Menulis runner Kaggle; tidak mengeksekusi model secara lokal."""
import base64
import json
from compose_lib import ROOT, library, resolve


def main():
    lib=library()
    folder=ROOT/'tools/composer_kaggle';folder.mkdir(exist_ok=True)
    keys=resolve(lib,list(lib['components']))
    blocks=[lib['components'][key]['code'] for key in keys if lib['components'][key]['code'].strip()]
    payload=base64.b64encode(json.dumps({'blocks':blocks,'configs':{task:r['config'] for task,r in lib['recipes'].items()},
        'smoke':(ROOT/'tools/composer_smoke.py').read_text(encoding='utf-8')}).encode()).decode()
    packages=sorted(set(p for c in lib['components'].values() for p in c['packages']))+['iterative-stratification>=0.1,<1']
    code="import os,sys,subprocess,json,base64\nos.environ.update(USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')\n"
    code+='subprocess.run([sys.executable,"-m","pip","install","-q",'+','.join(map(repr,packages))+'],check=True)\n'
    code+=f"payload=json.loads(base64.b64decode({payload!r}))\nexec(payload['smoke'])\nresults=smoke_test(payload['blocks'],payload['configs'],advanced=True)\nassert all(r['ok'] for r in results), 'Ada kontrak gagal; baca validation.json.'"
    book={'nbformat':4,'nbformat_minor':5,'cells':[{'id':'smoke','cell_type':'code','metadata':{},'source':code,'outputs':[],'execution_count':None}],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}}
    (folder/'runner.ipynb').write_text(json.dumps(book),encoding='utf-8')
    meta={'id':'ivanwllm/gemastik-toolkit-modular-smoke','title':'Gemastik Toolkit Modular Smoke','code_file':'runner.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':True,
        'dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    status_file=ROOT/'references/composer_validation.json'
    if not status_file.exists():
        status_file.write_text(json.dumps({'status':'prepared','kernel':meta['id'],
            'scope':'Kaggle synthetic contract smoke, not a score benchmark','local_ml_execution':False},indent=2),encoding='utf-8')
    print(folder)


if __name__=='__main__':main()
