"""Static source equivalence: full original pipeline in the unified task flow."""
import ast,json
import nbformat
from build_cheatsheet import build,notebook,ROOT
from pipelines_lib import all_pipelines,snippet_body


def main():
    data=build();pipelines=all_pipelines()
    assert len({task for flow in data['flows'] for task in flow['tasks']})==43
    for flow in data['flows']:
        pipeline=pipelines[flow['pipeline_source'].split('/')[-1].removesuffix('.py')]
        assert [data['cards'][key]['code'] for key in flow['core']]==[stage['code'] for stage in pipeline['stages']]
        book=notebook(flow,data['cards']);nbformat.validate(nbformat.from_dict(book))
        ast.parse('\n'.join(c['source'] for c in book['cells'] if c['cell_type']=='code'))
    for row in data['audit']['snippets']:
        assert snippet_body(row['id']) in data['cards']['snippet:'+row['id']]['code']
    report={'static':'passed','tasks':43,'flows':len(data['flows']),'full_original_pipeline_equivalence':True,
        'full_snippets_checked':len(data['audit']['snippets']),'local_ml_execution':False,
        'runtime':'No new runtime claim; earlier short-code tests do not validate this restored full-code version.'}
    (ROOT/'references/cheatsheet_static.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report)


if __name__=='__main__':main()
