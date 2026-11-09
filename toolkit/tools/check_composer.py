"""Static-only checks: no model fitting/inference locally."""
import ast
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import nbformat
from compose_lib import ROOT, library, resolve, build_notebook


def main():
    lib = library()
    count = 0
    reports = {}
    with tempfile.TemporaryDirectory(prefix='composer_static_') as temp:
        paths = []
        for task,r in lib['recipes'].items():
            choices = [key for key in r['choices'] if key.startswith('model.')]
            combinations = [r['defaults'],choices]+[[key] for key in choices]+[list(pair) for pair in itertools.combinations(choices,2)]
            for selected in combinations:
                book = build_notebook(lib,task,selected)
                nbformat.validate(nbformat.from_dict(book))
                order = book['metadata']['toolkit']['resolved']
                assert len(order)==len(set(order))
                for key in order:
                    assert all(order.index(dep)<order.index(key) for dep in lib['components'][key]['needs'])
                code = '\n\n'.join(cell['source'] for cell in book['cells'] if cell['cell_type']=='code')
                ast.parse(code)
                path = Path(temp)/f'{task}_{count}.py';path.write_text(code,encoding='utf-8');paths.append(str(path))
                count += 1
            reports[task] = {'combinations':len(combinations),'defaults':r['defaults']}
        # Undefined names matter for plug-in dependencies. This does not execute generated code.
        result = subprocess.run([sys.executable,'-m','ruff','check','--select','F821,F822,F823',*paths],capture_output=True,text=True)
        assert result.returncode==0, result.stdout+result.stderr
    result = subprocess.run(['node',str(ROOT/'tools/check_composer_js.cjs')],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    browser = json.loads(result.stdout)
    for task,book in browser.items():
        nbformat.validate(nbformat.from_dict(book))
        python = build_notebook(lib,task)
        combined = lambda nb:'\n\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code')
        assert ast.dump(ast.parse(combined(book)))==ast.dump(ast.parse(combined(python))),f'JS/CLI berbeda: {task}'
    # Fail loud on broken dependency graphs.
    try: resolve(lib,['not_existing'])
    except ValueError: pass
    else: raise AssertionError('Unknown component tidak ditolak')
    report = {'static':'passed','assemblies_checked':count,'browser_cli_default_equivalence':len(browser),
        'runtime':'not_run_locally','tasks':reports}
    (ROOT/'references/composer_static.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Static PASS: {count} rakitan, 8 default browser = CLI; tidak ada training/inference lokal.')


if __name__=='__main__': main()
