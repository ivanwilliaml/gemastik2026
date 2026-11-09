"""Download existing Kaggle outputs once; never wait, train, or run inference locally."""
from pathlib import Path
import json
import shutil
import tempfile
import nbformat
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
KERNEL = 'ivanwllm/gemastik-pair-time-review'
FILES = json.loads((ROOT/'_build/pair_time_kaggle/suite_files.json').read_text(encoding='utf-8'))
RESULTS = {'similarity':'hasil_similarity_review','nli3':'hasil_nli3_review',
           'retrieval':'hasil_retrieval_review','temporal':'hasil_temporal_review'}

def main():
    api=KaggleApi();api.authenticate()
    status=json.loads(str(api.kernels_status(KERNEL)))
    print('Kaggle:',status)
    if status['status'] not in {'COMPLETE','ERROR','CANCELLED'}:
        print('Belum selesai. Tidak menunggu; file sumber lokal tetap tersedia.')
        return
    with tempfile.TemporaryDirectory(prefix='gemastik_pair_sync_') as tmp:
        download=Path(tmp)
        api.kernels_output(KERNEL,str(download),force=True,quiet=True,
            file_pattern=r'(^[^/]+_review\.ipynb$|^hasil_[^/]+/[^/]+\.(csv|json)$|^suite_status\.json$)')
        report={}
        for task,relative in FILES.items():
            source=download/(task+'_review.ipynb');target=ROOT/relative
            if not source.exists():report[task]='belum ada output notebook';continue
            executed=nbformat.read(source,4);nbformat.validate(executed)
            errors=[o for c in executed.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
            local=nbformat.read(target,4)
            same=[(c.cell_type,c.source) for c in local.cells]==[(c.cell_type,c.source) for c in executed.cells]
            if not errors:
                destination=target if same else target.with_name(target.stem+'_executed.ipynb')
                shutil.copy2(source,destination)
                report[task]='output tersimpan: '+str(destination)
            else:
                failed=target.parent/'hasil_review'/'failed';failed.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,failed/source.name)
                report[task]='run gagal; sumber tidak ditimpa, output error tersimpan'
            source_reports=download/RESULTS[task]
            if source_reports.exists():
                destination=target.parent/'hasil_review'/RESULTS[task]
                destination.mkdir(parents=True,exist_ok=True)
                for path in source_reports.iterdir():
                    if path.is_file():shutil.copy2(path,destination/path.name)
        (ROOT/'_build/pair_time_kaggle/sync_status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        for task,message in report.items():print(task+': '+message)

if __name__=='__main__':main()
