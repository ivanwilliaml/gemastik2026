"""Rakit notebook self-contained; tidak menjalankan training/inference."""
import argparse
import json
from pathlib import Path
from compose_lib import ROOT, library, build_notebook, execution_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',nargs='?')
    parser.add_argument('--models',help='Nama model dipisah koma; tambahkan sentiment.pretrained untuk tren')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--config',type=Path,help='JSON override CFG, tidak perlu edit kode blok')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--build-all',action='store_true')
    parser.add_argument('--no-install',action='store_true')
    args = parser.parse_args()
    lib = library()
    if args.list:
        for key,recipe in lib['recipes'].items(): print(key+': '+', '.join(recipe['choices']))
        return
    if args.build_all:
        folder = args.out or ROOT/'notebooks/modular'
        folder.mkdir(parents=True,exist_ok=True)
        for task in lib['recipes']:
            book = build_notebook(lib,task,install=not args.no_install)
            (folder/(task+'_modular.ipynb')).write_text(json.dumps(book,ensure_ascii=False,indent=1),encoding='utf-8')
        lib['steps'] = {task:execution_steps(task) for task in lib['recipes']}
        target = ROOT/'assets/js/generated/composer.js'
        target.write_text('window.TK = window.TK || {};\nTK.composer = '+json.dumps(lib,ensure_ascii=False)+';\n',encoding='utf-8')
        print('Dibuat:',len(lib['recipes']),'notebook modular + data perakit browser')
        return
    if args.task not in lib['recipes']: parser.error('Pilih task: '+', '.join(lib['recipes']))
    choices = None if args.models is None else [m if m.startswith(('model.','sentiment.')) else 'model.'+m for m in args.models.split(',')]
    overrides = json.loads(args.config.read_text(encoding='utf-8')) if args.config else None
    book = build_notebook(lib,args.task,choices,overrides,not args.no_install)
    out = args.out or ROOT/'notebooks/modular'/(args.task+'_custom.ipynb')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(book,ensure_ascii=False,indent=1),encoding='utf-8')
    print(out)


if __name__=='__main__':
    main()
