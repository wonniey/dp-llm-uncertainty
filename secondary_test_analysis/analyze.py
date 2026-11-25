import json,zipfile,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
P=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
seeds=[123,456,789];conditions=['standard_poisson','clipping_poisson','dp_sigma_0p1','dp_sigma_0p25','dp_sigma_0p5'];scores=['mean_token_entropy','mean_answer_nll','discrete_semantic_entropy']
frames=[];outcomes=[];protocols=[];validities=[];checks=[];thresholds={}
for seed in seeds:
 names={123:'llama_low_noise_test_seed123_reports_6886d9037735.zip',456:'llama_low_noise_test_seed456_reports_dceb7c16585b.zip',789:'llama_low_noise_test_seed789_reports_38433cca7401.zip'};f=P/'upload'/names[seed];z=zipfile.ZipFile(f)
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

meanboot=boot.mean(axis=1)
point=np.array([r['AUROC'] for r in metrics]).reshape(3,5,3).mean(axis=0)
contrasts=[]
alpha=.05/19
def add(kind,label,terms):
    value=sum(coef*point[c,s] for coef,c,s in terms)
    draws=sum(coef*meanboot[:,c,s] for coef,c,s in terms)
    lo,hi=np.quantile(draws,[alpha/2,1-alpha/2])
    contrasts.append(dict(kind=kind,label=label,delta_AUROC=float(value),lower=float(lo),upper=float(hi)))
for ci,c in enumerate(conditions):
    for si in (0,1):add('Direct estimator comparison',c+': semantic minus '+scores[si],[(1,ci,2),(-1,ci,si)])
for ci,c in enumerate(conditions[2:],2):
    for si in (0,1):add('Change in semantic advantage versus clipping',c+': semantic versus '+scores[si],[(1,ci,2),(-1,ci,si),(-1,1,2),(1,1,si)])
for si,s in enumerate(scores):add('Clipping minus standard',s,[(1,1,si),(-1,0,si)])
assert len(contrasts)==19 and np.isfinite(boot).all()
# Ensure this run reproduces the primary analysis exactly.
primary=json.loads((P/'final_test_analysis_corrected/results.json').read_text())
for r in primary['primary_contrasts']:
    ci=conditions.index(r['condition']);si=scores.index(r['score'])
    lo,hi=np.quantile(meanboot[:,ci,si]-meanboot[:,1,si],[.05/18,1-.05/18])
    assert np.allclose([point[ci,si]-point[1,si],lo,hi],[r['delta_AUROC'],r['adjusted_lower'],r['adjusted_upper']],atol=1e-12,rtol=0)
result=dict(status='Exploratory, post-results secondary analysis',n_questions=n,seeds=seeds,bootstrap_replicates=B,bootstrap_seed=20261024,interval_level=1-alpha,family_size=19,checks=checks,primary_results_reproduced=True,contrasts=contrasts)
(D/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
lines=['# Secondary RQ1 and RQ2 analysis','', 'Exploratory comparisons selected after inspecting primary results. These supplement, and do not replace, the nine primary comparisons.','', '945 shared included questions; equal-weight average of seed-specific AUROCs for seeds 123, 456 and 789. 20,000 paired question-bootstrap draws resample the same questions for all models and scores. Seeds are not resampled. All original primary estimates and intervals were reproduced exactly.','', 'One family of 19 secondary contrasts uses Bonferroni-adjusted percentile intervals: nominal individual coverage 1 − 0.05/19 = 99.7368%. This gives approximate family-wise 95% coverage under bootstrap assumptions for this specified family, not for every analysis considered during the project. Extreme-tail endpoints have Monte Carlo uncertainty (about 26 draws in each tail).','', '## Definitions','', '- Direct comparison: semantic AUROC minus token-entropy or answer-NLL AUROC within a condition. Positive values favor semantic entropy.','- Change in advantage: (semantic minus token-based AUROC under noise) minus (semantic minus token-based AUROC under clipping). Positive values mean semantic entropy changes more favorably when adding noise.','- Clipping effect: clipping-only AUROC minus standard-training AUROC for the same score.','']
for kind in dict.fromkeys(r['kind'] for r in contrasts):
    lines+=['## '+kind,'','| Comparison | AUROC difference | Adjusted interval |','|---|---:|---|']
    for r in contrasts:
        if r['kind']==kind:lines.append(f"| {r['label']} | {r['delta_AUROC']:+.4f} | [{r['lower']:+.4f}, {r['upper']:+.4f}] |")
    lines+=['']
lines+=['## Interpretation limits','', 'Intervals describe question-sampling uncertainty conditional on the 15 fitted adapters and fixed correctness labels. They do not account for training-run population variability, label errors or semantic-clustering uncertainty. Labels are AI-assisted with partial human review. The test split had prior Qwen exposure, and the noise range was selected after development exploration. No direct noise-level contrasts are tested here, so this analysis does not establish a monotonic dose response. AUROC does not measure answer accuracy.','', 'The lower-noise conditions can have weak privacy: reported per-run epsilon is approximately 916.27, 53.72 and 9.17 for sigma 0.1, 0.25 and 0.5 respectively at delta 1e-5. These are not joint-release privacy guarantees.','', '## Reproduction','', 'Extract the complete ZIP, install numpy, pandas and scikit-learn, then run `python secondary_test_analysis/analyze.py`. Included source report ZIPs and reviewed CSVs are hash-verified. The original primary results JSON is included as a regression check. No model inference or training is required. See analysis_environment.json for the analysis package versions.']
(D/'report.md').write_text('\n'.join(lines)+'\n')
import importlib.metadata as md
(D/'analysis_environment.json').write_text(json.dumps({p:md.version(p) for p in ['numpy','pandas','scikit-learn']},indent=2))
print(json.dumps(contrasts,indent=2))
