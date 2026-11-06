import json,zipfile,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
P=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
seeds=[123,456,789];conditions=['standard_poisson','clipping_poisson','dp_sigma_0p1','dp_sigma_0p25','dp_sigma_0p5'];scores=['mean_token_entropy','mean_answer_nll','discrete_semantic_entropy']
frames=[];outcomes=[];protocols=[];validities=[];checks=[];thresholds={}
for seed in seeds:
 names={123:'llama_low_noise_test_seed123_reports_6886d9037735.zip',456:'llama_low_noise_test_seed456_reports_dceb7c16585b.zip',789:'llama_low_noise_test_seed789_reports_c1f11bb6eebd.zip'};f=P/'upload'/names[seed];z=zipfile.ZipFile(f)
 df=pd.read_csv(z.open('per_question_scores.csv'),float_precision='round_trip');o=pd.read_csv(z.open('outcomes.csv'));protocol=json.loads(z.read('frozen_protocol.json'));v=pd.read_csv(z.open('question_validity.csv'),keep_default_na=False);inputs=json.loads(z.read('analysis_inputs.json'))
 reviewed=P/f'llama_test{seed}_human_reviewed/blind_answer_review.csv';assert hashlib.sha256(reviewed.read_bytes()).hexdigest()==inputs['review'],'Review hash mismatch'
 protocols.append(protocol);validities.append(v.sort_values('question_id').reset_index(drop=True));frames.append(df);outcomes.append(o)
 thresholds[seed]={(r['condition'],r['score']):r for r in json.loads(z.read('applied_development_thresholds.json'))}
 assert set(df.seed)=={seed} and not df.duplicated(['condition','score','question_id']).any()
 checks.append({'seed':seed,'input':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'merged_review_hash_verified':True,'human_checked_eligible_answers':int(pd.read_csv(z.open('review_provenance.csv')).human_checked.sum())})
assert all(p==protocols[0] for p in protocols)
assert all(v.equals(validities[0]) for v in validities)
ids=sorted(validities[0].loc[validities[0].decision=='include','question_id']);assert len(ids)==945
data=pd.concat(frames);out=pd.concat(outcomes);prepared=[];metrics=[]
for seed in seeds:
 for c in conditions:
  labels=[]
  for s in scores:
   a=data[(data.seed==seed)&(data.condition==c)&(data.score==s)].set_index('question_id').loc[ids];assert len(a)==945
   x=a.value.to_numpy();y=a.incorrect.to_numpy();assert np.isfinite(x).all() and set(y)=={0,1};labels.append(y)
   t=thresholds[seed][c,s];fr=next(r for r in protocols[0]['threshold_sources'][str(seed)]['thresholds'] if r['condition']==c and r['score']==s)
   assert t['threshold']==fr['threshold'] and t['abstain_all']==fr['abstain_all']
   accept=np.zeros(len(y),bool) if t['abstain_all'] else x<=t['threshold'];assert np.array_equal(accept,a.accepted.to_numpy())
   auc=roc_auc_score(y,x);acc=1-y.mean();cov=accept.mean();risk=y[accept].mean() if accept.any() else None
   saved=out[(out.seed==seed)&(out.condition==c)&(out.score==s)].iloc[0]
   for k,val in [('AUROC',auc),('accuracy',acc),('test_coverage_at_frozen_threshold',cov),('test_risk_at_frozen_threshold',risk)]:assert val is None or np.isclose(saved[k],val,atol=1e-12), (seed,c,s,k,saved[k],val)
   metrics.append({'seed':seed,'condition':c,'score':s,'accuracy':float(acc),'AUROC':float(auc),'coverage':float(cov),'risk':None if risk is None else float(risk),'epsilon':None if pd.isna(saved.accounted_epsilon) else float(saved.accounted_epsilon)})
   order=np.argsort(x,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(x[order]))+1];prepared.append((seed,c,s,order,starts,y[order]))
  assert all(np.array_equal(labels[0],v) for v in labels)
print('Verified 45 outcome rows, frozen threshold application, shared protocol/questions and merged review hashes.',flush=True)
# Paired nonparametric question bootstrap. Each draw resamples the SAME question indices for all models and seeds.
# Weighted Mann-Whitney AUROC with exact half-credit for tied scores avoids repeated sklearn calls.
B=20000;rng=np.random.default_rng(20261024);boot=np.empty((B,3,5,3));n=len(ids)
for start in range(0,B,200):
 end=min(B,start+200);w=rng.multinomial(n,np.full(n,1/n),size=end-start)
 for seed,c,s,order,groups,y in prepared:
  a=w[:,order];pos=np.add.reduceat(a*y,groups,axis=1);neg=np.add.reduceat(a*(1-y),groups,axis=1)
  numerator=(pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(axis=1)
  boot[start:end,seeds.index(seed),conditions.index(c),scores.index(s)]=numerator/(pos.sum(axis=1)*neg.sum(axis=1))
meanboot=boot.mean(axis=1);m=pd.DataFrame(metrics);means=m.groupby(['condition','score'],sort=False)[['accuracy','AUROC','coverage','risk','epsilon']].mean().reset_index();contrasts=[]
alpha=.05/9
for ci,c in enumerate(conditions[2:],2):
 for si,s in enumerate(scores):
  d=meanboot[:,ci,si]-meanboot[:,1,si];lo,hi=np.quantile(d,[alpha/2,1-alpha/2]);point=float(m[(m.condition==c)&(m.score==s)].AUROC.mean()-m[(m.condition==conditions[1])&(m.score==s)].AUROC.mean())
  contrasts.append({'condition':c,'score':s,'delta_AUROC':point,'adjusted_lower':float(lo),'adjusted_upper':float(hi)})
results={'checks':checks,'n_questions':n,'bootstrap_replicates':B,'bootstrap_seed':20261024,'interval':'percentile, individual nominal 1-0.05/9 (99.4444%), Bonferroni family of nine; conditional on fitted adapters; fixed human/AI labels','per_seed':metrics,'means':json.loads(means.to_json(orient='records')),'primary_contrasts':contrasts}
(D/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
print(means.to_string(index=False));print(pd.DataFrame(contrasts).to_string(index=False))
lines=['# Three-seed Llama low-noise test analysis','',f'945 shared included questions; 55 held. Seeds 123, 456, 789. All 45 reported metric rows reproduced from per-question data. Review hashes match the merged human-review packages. Frozen thresholds and their acceptance decisions verified. Shared frozen protocols and question-validity tables match.','', '## Mean results across seeds','', '| Condition | Accuracy | Token AUROC | NLL AUROC | Semantic AUROC |','|---|---:|---:|---:|---:|']
for c in conditions:
 a={r['score']:r for r in results['means'] if r['condition']==c};lines.append(f"| {c} | {a[scores[0]]['accuracy']:.2%} | "+' | '.join(f"{a[s]['AUROC']:.4f}" for s in scores)+' |')
lines+=['','## Frozen-threshold selective answering','', 'Coverage and risk are separately averaged across seeds. Target was at most 10% observed DEVELOPMENT error; this is not a guarantee of TEST error.','', '| Condition | Token coverage / risk | NLL coverage / risk | Semantic coverage / risk |','|---|---:|---:|---:|']
for c in conditions:
 a={r['score']:r for r in results['means'] if r['condition']==c};lines.append('| '+c+' | '+' | '.join(f"{a[s]['coverage']:.2%} / {a[s]['risk']:.2%}" for s in scores)+' |')
lines+=['','## Primary comparisons: noise minus clipping','', '20,000 paired question-bootstrap draws, equal-weight mean of three seed-specific AUROCs. Each draw uses the same resampled questions for every seed, condition and estimator. Percentile intervals have nominal 99.4444% individual coverage, using Bonferroni alpha=0.05/9. No seeds are resampled.','', '| Condition | Score | AUROC change | Adjusted interval |','|---|---|---:|---|']
for r in contrasts:lines.append(f"| {r['condition']} | {r['score']} | {r['delta_AUROC']:+.4f} | [{r['adjusted_lower']:+.4f}, {r['adjusted_upper']:+.4f}] |")
lines+=['','## Limits and remaining work','', '- Labels remain AI-assisted with partial human review. Audit agreement after corrections is adjudicated agreement, not independent first-pass agreement. Question-wording concerns and incomplete extra reviews remain documented in review packages.','- Human decisions were merged per seed; this analysis verifies exact supplied review hashes, but does not establish semantic consistency across all differently spelled answers between seeds. A cross-seed label consistency audit remains appropriate before publication.','- These bootstrap intervals quantify question-sampling uncertainty conditional on the 15 fitted adapters and fixed labels. They do not establish variation across the full population of training runs or account for label/cluster uncertainty.','- This test split was previously used for Qwen, and the noise range was selected after development exploration. Describe it as the Llama test evaluation with prior exposure disclosed.','- These nine comparisons test noise versus clipping only, not direct differences between noise levels or superiority of one estimator over another.','- Original generation environments cannot be revalidated from these report bundles alone: analysis_inputs contains hashes, not the full generation identity files. Keep the original evaluation_identity.json and analysis_environment files in the reproducibility record.','- Epsilon values are per-run accounting, not privacy accounting for jointly releasing multiple adapters. Low noise can imply very weak privacy.','', '## Reproduction','', 'Place the three source ZIPs in ../upload relative to this folder, then run python analyze.py with numpy, pandas and scikit-learn. results.json contains per-seed metrics, aggregate values, hashes and all primary contrasts.']
(D/'report.md').write_text('\n'.join(lines)+'\n')
