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

# Fixed policies and labels; uncertainty is over shared test-question sampling only.
B=20000;n=len(ids);rng=np.random.default_rng(20261024)
accepted=[];errors=[]
for seed in seeds:
 for c in conditions:
  for s in scores:
   a=data[(data.seed==seed)&(data.condition==c)&(data.score==s)].set_index('question_id').loc[ids]
   accepted.append(a.accepted.to_numpy(dtype=float));errors.append(a.accepted.to_numpy(dtype=float)*a.incorrect.to_numpy())
accepted=np.array(accepted).T;errors=np.array(errors).T
cov_boot=np.empty((B,15));risk_boot=np.empty((B,15))
for start in range(0,B,200):
 end=min(B,start+200);w=rng.multinomial(n,np.full(n,1/n),size=end-start)
 ac=w@accepted;er=w@errors
 assert (ac>0).all(), 'Undefined bootstrap risk: zero accepted answers.'
 cov_boot[start:end]=(ac/n).reshape(-1,3,15).mean(axis=1)
 risk_boot[start:end]=(er/ac).reshape(-1,3,15).mean(axis=1)
m=pd.DataFrame(metrics);rows=[];alpha=.05/30
for ci,c in enumerate(conditions):
 for si,s in enumerate(scores):
  row={'condition':c,'score':s}
  subset=m[(m.condition==c)&(m.score==s)]
  for key,draws in [('coverage',cov_boot[:,ci*3+si]),('risk',risk_boot[:,ci*3+si])]:
   lo,hi=np.quantile(draws,[.025,.975]);alo,ahi=np.quantile(draws,[alpha/2,1-alpha/2])
   row[key]={'mean':float(subset[key].mean()),'lower_95':float(lo),'upper_95':float(hi),'adjusted_lower':float(alo),'adjusted_upper':float(ahi)}
  rows.append(row)
result={'n_questions':n,'seeds':seeds,'bootstrap_replicates':B,'bootstrap_seed':20261024,'estimand':'Equal-weight mean of seed-specific coverage and risk, separately; fixed development thresholds','adjusted_family_size':30,'adjusted_individual_level':1-alpha,'checks':checks,'results':rows,'per_seed':metrics}
(D/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
lines=['# RQ3: selective answering at frozen development thresholds','', '945 shared included test questions; seeds 123, 456 and 789; five conditions and three estimators. The acceptance decisions and original point estimates were verified against saved reports and the frozen development protocol. No thresholds were retuned.','', '## Method','', '20,000 paired question-bootstrap draws (seed 20261024), using the same question resampling weights across every seed, condition and estimator. In each draw, compute coverage and accepted-answer error separately for each seed, then average equally over the three seeds. Risk is not pooled across seeds. No draw had zero accepted answers.','', 'Both ordinary 95% percentile intervals and multiplicity-adjusted intervals are reported. The adjusted family contains all 30 quantities (15 coverage and 15 risk), using alpha=0.05/30, or nominal 99.8333% individual intervals. Approximate family-wise coverage is 95% under bootstrap assumptions. Extreme endpoints use about 17 draws per tail and have Monte Carlo uncertainty. This is a secondary analysis after inspecting results.','', '## Results: ordinary 95% intervals','', '| Condition | Estimator | Coverage [95% interval] | Error risk [95% interval] |','|---|---|---|---|']
for r in rows:
 cells=[]
 for key in ['coverage','risk']:
  v=r[key];cells.append(f"{v['mean']:.2%} [{v['lower_95']:.2%}, {v['upper_95']:.2%}]")
 lines.append('| '+r['condition']+' | '+r['score']+' | '+' | '.join(cells)+' |')
lines+=['','## Multiplicity-adjusted intervals','', '| Condition | Estimator | Coverage interval | Risk interval |','|---|---|---|---|']
for r in rows:
 lines.append('| '+r['condition']+' | '+r['score']+' | '+' | '.join(f"[{r[k]['adjusted_lower']:.2%}, {r[k]['adjusted_upper']:.2%}]" for k in ['coverage','risk'])+' |')
lines+=['','## Interpretation','', 'All point-estimate mean test risks are below the 10% target. This alone does not establish that population risk is at most 10%. The target was an observed development-error criterion, not a formal risk-control guarantee. An upper interval below 10% provides conditional statistical evidence for the mean risk of these three fixed policies; it does not certify every seed or future run.','', 'Semantic entropy retains numerically greater coverage in every noisy condition. No pairwise coverage superiority tests are performed here. More coverage at different achieved risks is not an equal-risk comparison.','', 'These intervals condition on fitted adapters, fixed development thresholds, fixed correctness labels and clustering. They do not account for training variability, development-threshold selection uncertainty, label or clustering errors. Human review is partial and AI-assisted. The test split had prior Qwen exposure. Three seeds do not establish population-of-training-runs behavior.','', '## Reproduce','', 'Extract the full archive and run `python rq3_test_analysis/analyze.py` with numpy, pandas and scikit-learn. No model reruns are needed. Source report ZIPs and reviewed label files are included for hash validation. Package versions are in analysis_environment.json.']
(D/'report.md').write_text('\n'.join(lines)+'\n')
import importlib.metadata as md
(D/'analysis_environment.json').write_text(json.dumps({p:md.version(p) for p in ['numpy','pandas','scikit-learn']},indent=2))
for r in rows:print(r)
