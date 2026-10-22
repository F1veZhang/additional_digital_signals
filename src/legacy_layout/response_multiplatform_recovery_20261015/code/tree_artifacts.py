"""Resolve preserved primary tree outputs plus non-destructive budget recovery."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def receipt_path(jid):
    candidates=[ROOT/'results/tree_resume/cells'/jid/'completion.json',ROOT/'results/tree/cells'/(jid+'.json')]
    return next((p for p in candidates if p.exists() and read(p).get('success')),candidates[-1])
def receipt(jid):
    p=receipt_path(jid);return read(p) if p.exists() else {}
def predictions(jid):
    p=receipt_path(jid)
    return p.parent/'predictions.csv' if p.name=='completion.json' else p.parent/(jid+'_predictions.csv')
def curve(jid,seed):
    p=receipt_path(jid);r=read(p)
    result=p.parent/f'seed{seed}_curve.csv' if p.name=='completion.json' else p.parent/f'{jid}_seed{seed}_curve.csv'
    if result.exists():return result
    if r.get('alias_of'):return curve(r['alias_of'],seed)
    raise FileNotFoundError(str(result))
def pending(jobs):return [j['job_id'] for j in jobs if not receipt(j['job_id']).get('success')]
