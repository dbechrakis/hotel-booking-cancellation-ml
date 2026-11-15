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
for k,value in metadata['notebook_reference_metrics'].items(): close(lr[k],value)
assert set(metadata['holdout_metrics'])==set(metadata['notebook_reference_metrics'])
cal=metadata['calibration']
assert cal['calibration_rows']+cal['evaluation_rows']==v['test_rows']
assert cal['evaluation_calibrated']['brier']<cal['evaluation_uncalibrated']['brier']
assert metadata['features']==[f for f in v['features'] if f not in metadata['removed_inputs']]
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

ob=json.loads((ROOT/'outputs/overbooking/summary.json').read_text())
counts=json.loads((ROOT/'outputs/overbooking/counts.json').read_text())
assert counts['source_sha256']==metadata['source_sha256'] and counts['gate_passed'] and counts['evaluation_nights']==ob['evaluation_nights']
agg={}
for r in table('outputs/overbooking/night_forecasts.csv'):
    a=agg.setdefault((r['method'],int(r['horizon'])),[0,0.0,0]); a[0]+=1; a[1]+=float(r['crps']); a[2]+=int(r['in80'])
readme=(ROOT/'README.md').read_text()
for r in ob['forecast_scores_all_hotels']:
    n,crps,cover=agg[(r['method'],r['horizon'])]
    assert n==r['nights']; close(crps/n,r['crps']); close(cover/n,r['coverage80'])
for m in ['flat','segment','at_booking','model','model_od']:
    row='| `%s` | '%m+' | '.join('%.2f'%(agg[(m,h)][1]/agg[(m,h)][0]) for h in (30,14,7,3))+' | '+' | '.join('%.1f%%'%(100*agg[(m,h)][2]/agg[(m,h)][0]) for h in (30,14,7,3))+' |'
    assert row in readme,row
net={}
for r in table('outputs/overbooking/scenarios.csv'):
    if r['capacity_shift']=='0' and r['cost_ratio']=='2':
        a=net.setdefault((r['method'],int(r['horizon']),float(r['demand_share'])),[0,0.0]); a[0]+=int(r['nights']); a[1]+=float(r['net'])
for h in (30,7):
    for m in ['flat','segment','at_booking','model','model_od']:
        row='| `%s` | %d days | '%(m,h)+' | '.join('%.1f'%(100*net[(m,h,s)][1]/net[(m,h,s)][0]) for s in (0.25,0.5,1.0))+' |'
        assert row in readme,row
validation=(ROOT/'VALIDATION.md').read_text()
for q,g in ob['expectations'].items():
    assert '- %s. %s.'%(q,{True:'Passed',False:'Failed',None:'Not graded'}[g['passed']]) in validation,q

print('Committed evidence and syntax checks passed; see VALIDATION.md for rerun scope.')
