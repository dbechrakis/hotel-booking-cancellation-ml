"""Validate committed evidence without pretending to retrain external-data models."""
from pathlib import Path
import ast, csv, json, math
ROOT=Path(__file__).resolve().parents[1]
for p in ROOT.rglob('*.py'):
    if not any(x in p.parts for x in ['.git','.venv','node_modules']): ast.parse(p.read_text())
for p in ROOT.rglob('*.ipynb'):
    n=json.loads(p.read_text()); assert n['nbformat']==4
    for c in n['cells']:
        assert c['cell_type'] in ['code','markdown','raw']
        assert not any(o.get('output_type')=='error' for o in c.get('outputs',[])), str(p)
def table(path):
    with (ROOT/path).open(newline='',encoding='latin1' if str(path).startswith('data/') else 'utf-8') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-6): assert math.isclose(float(a),float(b),abs_tol=tol,rel_tol=tol),(a,b)
v=json.loads((ROOT/'outputs/validation.json').read_text())
assert v['raw_rows']==v['train_rows']+v['test_rows']+v['purged_rows']
assert not {'reservation_status','reservation_status_date','is_canceled','assigned_room_type'} & set(v['features'])
rows=table('outputs/classification_metrics.csv')
for r in rows:
    for k in ['accuracy','precision','recall','f1','average_precision','roc_auc']:assert 0<=float(r[k])<=1
    p,q=float(r['precision']),float(r['recall']);close(r['f1'],2*p*q/(p+q) if p+q else 0)
close(rows[0]['average_precision'],v['test_cancellation_rate'])

print('Committed evidence and syntax checks passed; see VALIDATION.md for rerun scope.')
