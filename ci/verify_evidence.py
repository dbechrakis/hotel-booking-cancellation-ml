"""Validate committed evidence without pretending to retrain external-data models."""
from pathlib import Path
import ast, csv, gzip, hashlib, json, math
ROOT=Path(__file__).resolve().parents[1]
for p in ROOT.rglob('*.py'):
    if not any(x in p.parts for x in ['.git','.venv','node_modules']): ast.parse(p.read_text())
for p in ROOT.rglob('*.ipynb'):
    n=json.loads(p.read_text()); assert n['nbformat']==4
    for c in n['cells']:
        assert c['cell_type'] in ['code','markdown','raw']
        assert not any(o.get('output_type')=='error' for o in c.get('outputs',[])), str(p)
def table(path):
    target=ROOT/path
    encoding='latin1' if str(path).startswith('data/') else 'utf-8'
    if target.suffix=='.gz':
        handle=gzip.open(target,mode='rt',newline='',encoding=encoding)
    else:
        handle=target.open(mode='r',newline='',encoding=encoding)
    with handle as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-6): assert math.isclose(float(a),float(b),abs_tol=tol,rel_tol=tol),(a,b)
v=json.loads((ROOT/'outputs/validation.json').read_text())
assert v['raw_rows']==v['train_rows']+v['test_rows']+v['purged_rows']
assert not {'reservation_status','reservation_status_date','is_canceled','assigned_room_type'} & set(v['features'])
rows=table('outputs/classification_metrics.csv')
for r in rows:
    for k in ['accuracy','precision','recall','f1','average_precision','roc_auc']:assert 0<=float(r[k])<=1
    p,q=float(r['precision']),float(r['recall']);close(r['f1'],2*p*q/(p+q) if p+q else 0)
close(rows[0]['average_precision'],v['test_cancellation_rate'])
metadata=json.loads((ROOT/'artifacts/model_metadata.json').read_text())
artifact=ROOT/'artifacts/cancellation_logistic.joblib'
assert hashlib.sha256(artifact.read_bytes()).hexdigest()==metadata['artifact_sha256']
assert hashlib.sha256((ROOT/'artifacts/cancellation_calibrator.joblib').read_bytes()).hexdigest()==metadata['calibrator_sha256']
lr=[r for r in rows if r['model']=='Logistic Regression'][0]
for k,value in metadata['holdout_metrics'].items(): close(lr[k],value)
cal=metadata['calibration']
assert cal['calibration_rows']+cal['evaluation_rows']==v['test_rows']
assert cal['evaluation_calibrated']['brier']<cal['evaluation_uncalibrated']['brier']
assert metadata['features']==v['features']
predictions=table('outputs/holdout_predictions.csv.gz')
assert len(predictions)==v['test_rows']
assert sum(int(r['is_canceled']) for r in predictions)/len(predictions)==v['test_cancellation_rate']
assert all(0<=float(r['risk_score'])<=1 and 0<=float(r['cancellation_probability'])<=1 for r in predictions)
assert sum(r['calibration_role']=='calibration' for r in predictions)==cal['calibration_rows']
policy=metadata['policy']
assert policy['threshold']==metadata['threshold']
assert sum(float(r['risk_score'])>=policy['threshold'] for r in predictions)==policy['holdout_flagged']

weekly=table('outputs/monitoring/weekly_replay.csv')
summary=json.loads((ROOT/'outputs/monitoring/summary.json').read_text())
assert summary['weeks']==len(weekly) and summary['bookings']==sum(int(r['bookings']) for r in weekly)
assert summary['weeks_below_target_recall']==sum(float(r['realised_recall'])<0.8 for r in weekly)
assert set(metadata['monitoring_reference'])>={'bin_edges','bin_shares','flagged_share'}
close(sum(metadata['monitoring_reference']['bin_shares']),1)

print('Committed evidence and syntax checks passed; see VALIDATION.md for rerun scope.')
