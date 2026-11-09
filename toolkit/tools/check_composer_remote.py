"""Nonblocking status / bounded live log; download the final JSON report only."""
import argparse
import json
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest

ROOT=Path(__file__).resolve().parents[1]
KERNEL='ivanwllm/gemastik-toolkit-modular-smoke'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',default='v1')
    parser.add_argument('--logs',action='store_true')
    parser.add_argument('--kernel',default=KERNEL)
    parser.add_argument('--prefix',default='composer_smoke')
    parser.add_argument('--report',default='composer_validation.json')
    args=parser.parse_args()
    api=KaggleApi();api.authenticate()
    status=json.loads(str(api.kernels_status(args.kernel)))
    report={'kernel':args.kernel,'version':args.version,**status,'local_ml_execution':False,'scope':'Synthetic contract smoke, not benchmark accuracy'}
    if args.logs:
        request=ApiGetKernelSessionLogsStreamRequest()
        request.user_name,request.kernel_slug=args.kernel.split('/')
        request.version_label=args.version;request.wait_for_logs_url_seconds=5
        with api.build_kaggle_client() as client:
            response=client.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
            try:response.raw._fp.fp.raw._sock.settimeout(2)
            except AttributeError:pass
            start=time.monotonic();chunks=[]
            try:
                while time.monotonic()-start<5:
                    block=response.raw.read1(65536)
                    if not block:break
                    chunks.append(block)
            except Exception:pass
            response.close()
        payload=[]
        for line in b''.join(chunks).decode('utf-8','replace').splitlines():
            if line.startswith('data: '):
                try:payload.append(json.loads(line[6:]).get('data',''))
                except (ValueError,AttributeError):pass
        rows=[]
        for line in ''.join(payload).splitlines():
            if line.startswith('{"test":'):
                try:rows.append(json.loads(line))
                except ValueError:pass
        report['live_tests']=rows
        report['live_summary']={'passed':sum(row['ok'] for row in rows),'failed':sum(not row['ok'] for row in rows)}
        print(json.dumps(rows,indent=2))
    if str(status.get('status','')).upper() in {'COMPLETE','ERROR'}:
        dest=ROOT/'references'/(args.prefix+'_remote');dest.mkdir(exist_ok=True)
        api.kernels_output(args.kernel,path=str(dest),file_pattern='^'+args.prefix+r'/validation\.json$',force=True,quiet=True)
        file=dest/args.prefix/'validation.json'
        if file.exists():report['validation']=json.loads(file.read_text(encoding='utf-8'))
    (ROOT/'references'/args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Status:',status.get('status'))


if __name__=='__main__':main()
