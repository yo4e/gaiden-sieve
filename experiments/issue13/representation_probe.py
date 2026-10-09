"""語と文字部分表現の2条件を、同じ教師とgroup分割で局所比較する。"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from dataclasses import asdict
import platform,joblib,sklearn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from support import HERE,REPO,read_json,read_jsonl,write_json,digest,verify_rows,verify_protected_files,verify_boundaries,canonical_url,overlaps
from evaluation_design import groups
from causal_probe import measure,EXPECTED
from score_approved_review import REFERENCES,APPROVAL
from gaiden_sieve.data import load_labeled_jsonl,labeled_dataset_hash,build_text
from gaiden_sieve.profile import load_profile
from gaiden_sieve.evaluate import calculate_metrics

CONDITIONS={'word12_c1':dict(analyzer='word',ngram_range=(1,2)),'charwb35_c1':dict(analyzer='char_wb',ngram_range=(3,5))}

def build_model(name):
    return Pipeline([('tfidf',TfidfVectorizer(**CONDITIONS[name])),('classifier',LogisticRegression(C=1,max_iter=1000,random_state=42))])

def grouping_rows(items,raw_rows):
    # URLは記事同一性の検査metadataとして保持し、モデル特徴には渡さない。
    metadata={r['id']:r for r in raw_rows}
    if len(metadata)!=len(raw_rows):raise ValueError('duplicate IDs in grouping metadata')
    rows=[]
    for item in items:
        raw=metadata.get(item.id)
        if raw is None or not isinstance(raw.get('source_url'),str) or not raw['source_url'].strip():
            raise ValueError('source_url missing in grouping metadata: '+item.id)
        rows.append(dict(id=item.id,source=item.source,title=item.title,summary=item.summary,label=item.label,source_url=raw['source_url']))
    return rows

def grouped_splits(rows):
    ids,edges=groups(rows);keys=[ids[r['id']] for r in rows];labels=[r['label'] for r in rows]
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(np.zeros(len(rows)),labels,keys))
    for train,test in folds:
        if {keys[i] for i in train}&{keys[i] for i in test}:raise ValueError('group overlap in OOF split')
        if len({labels[i] for i in train})!=2:raise ValueError('training fold misses label')
    return folds,keys,edges

def summarize_oof(rows):
    labels=[r['label'] for r in rows];predicted=[r['binary_prediction'] for r in rows]
    result=dict(count=len(rows),binary_metrics=asdict(calculate_metrics(labels,predicted,positive_label='relevant')),roc_auc=float(roc_auc_score([l=='relevant' for l in labels],[r['p_relevant'] for r in rows])),operational_distribution=dict(Counter(r['classification'] for r in rows)),rows=rows)
    result['negative_recall_by_source']={s:sum(r['binary_prediction']=='not_relevant' for r in rows if r['label']=='not_relevant' and r['source']==s)/sum(r['label']=='not_relevant' and r['source']==s for r in rows) for s in sorted({r['source'] for r in rows if r['label']=='not_relevant'})}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--private-root',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);args=p.parse_args()
    out=args.output_root.resolve()
    if not out.is_relative_to(REPO/'.diagnosis'):raise ValueError('model and private output must stay in ignored .diagnosis')
    verify_protected_files();training_path=args.private_root/'iteration2/reviewed-bootstrap';audit_path=args.private_root/'iteration2/independent-llm-audit.jsonl'
    if labeled_dataset_hash(training_path)!=EXPECTED:raise ValueError('original training hash mismatch')
    verify_rows(training_path/'05-reviewed-cross-source.jsonl','iteration2');verify_rows(audit_path,'llm_audit');verify_rows(args.private_root/'experiment/incoming.jsonl','frozen_rss')
    manifest=read_json(HERE/'lineage/representation-probe.json')
    if digest(args.probe.read_bytes())!=manifest['file_sha256']:raise ValueError('new probe hash mismatch')
    training,holdout,gold_review,audit=verify_boundaries(training_path,audit_path);new_probe=read_jsonl(args.probe)
    approved_all=read_jsonl(REPO/'.diagnosis/issue13-independent-eval/review/private-review-candidates.jsonl')
    approved=[dict(r,label=REFERENCES[r['review_id']]) for r in approved_all if r['review_id'] in REFERENCES]
    old_probe=read_jsonl(REPO/'.diagnosis/issue13-causal/prospective-validation.jsonl');quarantine=read_jsonl(args.private_root/'iteration2/quarantined.jsonl');rss=read_jsonl(args.private_root/'experiment/incoming.jsonl')
    protected=training+holdout+gold_review+audit+approved_all+old_probe+quarantine+rss
    group_ids,_=groups(protected+new_probe);protected_ids={group_ids[r['id']] for r in protected}
    if any(group_ids[r['id']] in protected_ids for r in new_probe):raise ValueError('new probe overlaps protected group')
    profile=load_profile(REPO/'profiles/ai-gaiden/config.yml');items=load_labeled_jsonl(training_path);texts=[build_text(i,profile.text_fields) for i in items];labels=[i.label for i in items]
    ordered=grouping_rows(items,training);folds,keys,edges=grouped_splits(ordered)
    unseen=[];seen=set()
    for r in rss:
        url=canonical_url(r)
        if not any(overlaps(r,q) for q in training+holdout+gold_review) and url not in seen:unseen.append(r);seen.add(url)
    report=dict(plan=read_json(HERE/'representation-plan.json'),training_hash=EXPECTED,probe_sha256=manifest['file_sha256'],approved_ms4_reference_method=APPROVAL['review_method'],runtime=dict(python=platform.python_version(),sklearn=sklearn.__version__,joblib=joblib.__version__),group_count=len(set(keys)),max_group_size=max(Counter(keys).values()),group_edges=edges,folds=[dict(fold=k,train_count=len(train),test_count=len(test),test_labels=dict(Counter(labels[i] for i in test)),test_ids=[ordered[i]['id'] for i in test],group_overlap=0) for k,(train,test) in enumerate(folds)],conditions={},limitations=['OOF assesses provisional teacher labels, not independent human accuracy','gold and approved MS4 are known audits, not tuning sets','new probe is balanced RSS-only assistant labeling, historical not future','feature OOV units differ between word and character; lower char OOV alone is not semantic accuracy','no threshold change, search, adoption or production connection'])
    out.mkdir(parents=True,exist_ok=True)
    for name in CONDITIONS:
        model=build_model(name);model.fit(texts,labels)
        if name=='word12_c1':
            original=joblib.load(next((args.private_root/'iteration2/reviewed/artifacts').rglob('*.joblib')));delta=float(np.max(np.abs(original.predict_proba(texts)-model.predict_proba(texts))))
            if delta>1e-12:raise ValueError('original word model parity failed')
            report['original_word_parity_max_delta']=delta
        joblib.dump(model,out/(name+'.joblib'))
        scores={s:measure(model.named_steps['tfidf'],model.named_steps['classifier'],rows) for s,rows in [('training',ordered),('known_holdout',holdout),('known_gold',read_jsonl(REPO/'profiles/ai-gaiden/gold.jsonl')),('known_llm_audit',audit),('known_approved_ms4',approved),('new_llm_probe',new_probe),('rss100',rss),('unseen62',unseen)]}
        oof=[]
        for fold,(train,test) in enumerate(folds):
            partial=build_model(name);partial.fit([texts[i] for i in train],[labels[i] for i in train])
            result=measure(partial.named_steps['tfidf'],partial.named_steps['classifier'],[ordered[i] for i in test])
            prediction=partial.predict([texts[i] for i in test])
            for row,binary in zip(result['rows'],prediction):oof.append(dict(row,binary_prediction=str(binary),fold=fold))
        summary=summarize_oof(oof)
        summary['per_fold']={str(k):{key:value for key,value in summarize_oof([r for r in oof if r['fold']==k]).items() if key!='rows'} for k in range(3)}
        scores['new_llm_probe']['by_source_auc']={source:float(roc_auc_score([r['label']=='relevant' for r in scores['new_llm_probe']['rows'] if r['source']==source],[r['p_relevant'] for r in scores['new_llm_probe']['rows'] if r['source']==source])) for source in sorted({r['source'] for r in new_probe})}
        confident=[r for r in scores['rss100']['rows'] if r['classification']=='relevant']
        rss_lookup={r['id']:r for r in rss}
        scores['rss100']['confident_relevant_known_training_count']=sum(any(overlaps(rss_lookup[r['id']],q) for q in training) for r in confident)
        report['conditions'][name]=dict(saved_local_model_sha256=digest((out/(name+'.joblib')).read_bytes()),local_artifact_only=True,feature_count=len(model.named_steps['tfidf'].vocabulary_),analyzer=CONDITIONS[name]['analyzer'],oov_unit='word_unigram_bigram_occurrences' if name.startswith('word') else 'within_word_character_3_to_5gram_occurrences',group_oof=summary,evaluations=scores)
    write_json(out/'results.json',report);print('Saved bounded representation comparison: '+str(out/'results.json'))
if __name__=='__main__':main()
