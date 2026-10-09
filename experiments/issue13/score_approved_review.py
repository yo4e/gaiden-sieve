"""方針承認された4件だけを既定の保存モデルで初回採点する。fitは呼ばない。"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import joblib
from sklearn.metrics import roc_auc_score
from support import HERE,REPO,read_json,read_jsonl,write_json,digest,verify_protected_files
from gaiden_sieve.profile import load_profile
from gaiden_sieve.data import build_input_text
from gaiden_sieve.classify import predict_one
from gaiden_sieve.evaluate import calculate_metrics
from dataclasses import asdict

REFERENCES={'MS01':'not_relevant','MS02':'relevant','MS03':'not_relevant','MS04':'relevant'}
APPROVAL=dict(approved_at='2026-10-09T10:45:00+09:00',review_method='owner_approval_of_assistant_summaries_and_proposed_editorial_labels',input_independently_read=False,blind_review=False,model_predictions_shown_before_approval=False,reference_origin='assistant_proposal_with_owner_policy_approval',provenance='Owner approval relayed by authorized task instruction; private message text is not republished')

def locked_core(path,manifest):
    if digest(path.read_bytes())!=manifest['private_jsonl_sha256']:raise ValueError('private review input hash mismatch')
    all_rows=read_jsonl(path);rows=[r for r in all_rows if r['evaluation_role']=='core']
    if len(rows)!=4 or {r['review_id'] for r in rows}!=set(REFERENCES):raise ValueError('expected exactly the four locked core IDs')
    expected={r['review_id']:r for r in manifest['rows']}
    for r in rows:
        m=expected[r['review_id']]
        if digest(r['title'])!=m['title_sha256'] or digest(r['summary'])!=m['summary_sha256']:raise ValueError('core text hash mismatch')
    return rows

def score_model(model,profile,rows):
    results=[]
    for r in rows:
        reference=REFERENCES[r['review_id']]
        text=build_input_text(title=r['title'],summary=r['summary'],text_fields=profile.text_fields)
        prediction=predict_one(model,profile,title=r['title'],summary=r['summary'])
        vectorizer=model.named_steps['tfidf'];tokens=vectorizer.build_analyzer()(text);oov=sum(t not in vectorizer.vocabulary_ for t in tokens)
        results.append(dict(review_id=r['review_id'],id=r['id'],source=r['source'],reference_label=reference,reference_basis=APPROVAL['reference_origin'],binary_prediction=str(model.predict([text])[0]),**asdict(prediction),oov_features=oov,total_features=len(tokens),title_sha256=digest(r['title']),summary_sha256=digest(r['summary'])))
    labels=[r['reference_label'] for r in results];predicted=[r['binary_prediction'] for r in results]
    metrics=asdict(calculate_metrics(labels,predicted,positive_label='relevant'))
    positive=[r for r in results if r['reference_label']=='relevant'];accepted=[r for r in results if r['classification']=='relevant'];tp=sum(r['reference_label']=='relevant' for r in accepted)
    total=sum(r['total_features'] for r in results);indexed={r['review_id']:r for r in results}
    return dict(rows=results,binary_metrics=metrics,binary_confusion=dict(tp=sum(a=='relevant' and b=='relevant' for a,b in zip(labels,predicted)),fp=sum(a=='not_relevant' and b=='relevant' for a,b in zip(labels,predicted)),tn=sum(a=='not_relevant' and b=='not_relevant' for a,b in zip(labels,predicted)),fn=sum(a=='relevant' and b=='not_relevant' for a,b in zip(labels,predicted))),roc_auc=float(roc_auc_score([l=='relevant' for l in labels],[r['probability_relevant'] for r in results])),operational=dict(distribution=dict(Counter(r['classification'] for r in results)),precision=tp/len(accepted) if accepted else None,recall=tp/len(positive),retained_for_review_recall=sum(r['classification']!='not_relevant' for r in positive)/len(positive)),oov_feature_rate=sum(r['oov_features'] for r in results)/total if total else 0,pair_score_gaps={negative+'_'+relevant:indexed[relevant]['probability_relevant']-indexed[negative]['probability_relevant'] for negative,relevant in [('MS01','MS02'),('MS03','MS04')]},source_only_all_relevant_baseline=dict(precision=.5,recall=1,f1=2/3,roc_auc=.5))

def main():
    p=argparse.ArgumentParser();p.add_argument('--private-root',type=Path,required=True);p.add_argument('--review-input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    verify_protected_files();manifest=read_json(HERE/'lineage/independent-review-candidates.json');rows=locked_core(args.review_input,manifest)
    folder=args.private_root/'iteration2/reviewed/artifacts/ai-gaiden/candidate';models=list(folder.glob('*.joblib'));metadata=list(folder.glob('*.metadata.json'))
    if len(models)!=1 or len(metadata)!=1:raise ValueError('expected one frozen iteration2 candidate')
    meta=read_json(metadata[0])
    if meta['model_version']!='20261008T231148972272Z-291d65aa' or meta['training_data_hash']!='sha256:291d65aa8fac6aaf53812e4ac584b448fe8d2c258a9ef646c2f5b2acc76e2b18':raise ValueError('frozen model metadata mismatch')
    if digest(models[0].read_bytes())!=meta['model_artifact_hash']:raise ValueError('frozen model artifact hash mismatch')
    profile=load_profile(REPO/'profiles/ai-gaiden/config.yml');model=joblib.load(models[0]);scores=score_model(model,profile,rows)
    report=dict(evaluation_role='small_policy_approved_nonblind_probe',approval=APPROVAL,count=4,source_count=1,approved_reference_labels=dict(Counter(REFERENCES.values())),model_version=meta['model_version'],model_artifact_hash=meta['model_artifact_hash'],training_data_hash=meta['training_data_hash'],model_code_commit=meta['code_commit'],dependencies=meta['dependencies'],review_input_sha256=digest(args.review_input.read_bytes()),thresholds=dict(relevant=profile.relevant_threshold,not_relevant=profile.not_relevant_threshold),model_refit=False,quality_gate_evaluated=False,gold_modified=False,limitations=['owner approved assistant descriptions and proposals; not blind independent input annotation','single source Microsoft n4 and historical article dates','cannot prove independent human accuracy, calibration, source shortcut removal or future generalization','not used for fitting, parameter selection, quality gate or promotion'],**scores)
    write_json(args.output,report);print('Saved frozen-model policy-approved probe: '+str(args.output))
if __name__=='__main__':main()
