"""正則化と反復テンプレートの局所診断。promotionは行わない。"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support
from support import REPO,read_json,read_jsonl,write_json,verify_protected_files,verify_boundaries,verify_rows,overlaps,digest
from gaiden_sieve.data import load_labeled_jsonl,labeled_dataset_hash,build_text,build_input_text
from gaiden_sieve.profile import load_profile

EXPECTED='sha256:291d65aa8fac6aaf53812e4ac584b448fe8d2c258a9ef646c2f5b2acc76e2b18'

def family_weights(texts,labels,analyzer):
    groups=defaultdict(list)
    for i,text in enumerate(texts):groups[tuple(sorted(analyzer(text)))].append(i)
    weights=np.ones(len(texts))
    for indices in groups.values():
        if len({labels[i] for i in indices})!=1:raise ValueError('mixed-label analyzer family')
        weights[indices]=1/len(indices)
    for label in set(labels):
        indices=np.flatnonzero(np.asarray(labels)==label)
        weights[indices]*=len(indices)/weights[indices].sum()
    return weights,sorted((len(v) for v in groups.values()),reverse=True)

def decompose(matrix,clf):
    contributions=matrix.multiply(clf.coef_[0]);positive=np.asarray(contributions.maximum(0).sum(axis=1)).ravel();negative=np.asarray(contributions.minimum(0).sum(axis=1)).ravel()
    logits=clf.intercept_[0]+positive+negative
    if not np.allclose(logits,clf.decision_function(matrix),atol=1e-12):raise ValueError('logit decomposition mismatch')
    return positive,negative,logits

def measure(vectorizer,clf,rows):
    texts=[build_input_text(title=r['title'],summary=r.get('summary',''),text_fields=('title','summary')) for r in rows];x=vectorizer.transform(texts);p=clf.predict_proba(x)[:,list(clf.classes_).index('relevant')]
    positive,negative,logits=decompose(x,clf);analyzer=vectorizer.build_analyzer();vocab=vectorizer.vocabulary_
    counts=Counter();sources=defaultdict(Counter);records=[];oov=total=0
    for i,row in enumerate(rows):
        label='relevant' if p[i]>=.8 else 'not_relevant' if p[i]<.4 else 'uncertain';counts[label]+=1;sources[row['source']][label]+=1
        tokens=analyzer(texts[i]);missing=sum(t not in vocab for t in tokens);oov+=missing;total+=len(tokens)
        records.append(dict(id=row['id'],source=row['source'],label=row.get('label'),p_relevant=float(p[i]),classification=label,positive_logit_contribution=float(positive[i]),negative_logit_contribution=float(negative[i]),nonzero_features=int(x[i].nnz),oov_features=missing,total_features=len(tokens)))
    result=dict(count=len(rows),distribution=dict(counts),source_distribution={s:dict(c) for s,c in sources.items()},oov_feature_rate=oov/total if total else 0,rows=records)
    if rows and all(r.get('label') in ['relevant','not_relevant'] for r in rows):
        y=np.array([r['label']=='relevant' for r in rows]);precision,recall,f1,_=precision_recall_fscore_support(y,clf.predict(x)=='relevant',average='binary',zero_division=0)
        result['binary']=dict(precision=float(precision),recall=float(recall),f1=float(f1));result['auc']=float(roc_auc_score(y,p)) if len(set(y))>1 else None
        predicted=p>=.8;tp=int(np.sum(y&predicted));result['operational']=dict(precision=tp/int(predicted.sum()) if predicted.sum() else None,recall=tp/int(y.sum()) if y.sum() else None)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--private-root',type=Path,required=True);parser.add_argument('--validation',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    verify_protected_files();training_path=args.private_root/'iteration2/reviewed-bootstrap';audit_path=args.private_root/'iteration2/independent-llm-audit.jsonl'
    if labeled_dataset_hash(training_path)!=EXPECTED:raise ValueError('original training hash mismatch')
    verify_rows(training_path/'05-reviewed-cross-source.jsonl','iteration2')
    verify_rows(audit_path,'llm_audit')
    verify_rows(args.private_root/'experiment/incoming.jsonl','frozen_rss')
    training,holdout,gold_review,audit=verify_boundaries(training_path,audit_path)
    expected_validation=read_json(Path(__file__).parent/'lineage/prospective-ms4.json')['file_sha256']
    if digest(args.validation.read_bytes())!=expected_validation:raise ValueError('locked validation hash mismatch')
    validation=read_jsonl(args.validation)
    if any(overlaps(r,p) for r in validation for p in training+holdout+gold_review+audit):raise ValueError('prospective validation overlaps protected data')
    profile=load_profile(REPO/'profiles/ai-gaiden/config.yml');items=load_labeled_jsonl(training_path);texts=[build_text(i,profile.text_fields) for i in items];labels=[i.label for i in items]
    vectorizer=TfidfVectorizer(ngram_range=(1,2));x=vectorizer.fit_transform(texts);weights,families=family_weights(texts,labels,vectorizer.build_analyzer())
    # 再代入診断の確率を、教師loaderと同じ行順で記録する。
    ordered=[dict(id=i.id,source=i.source,title=i.title,summary=i.summary,label=i.label) for i in items]
    gold=read_jsonl(REPO/'profiles/ai-gaiden/gold.jsonl');rss=read_jsonl(args.private_root/'experiment/incoming.jsonl')
    unseen=[];seen=set()
    from support import canonical_url
    for r in rss:
        key=canonical_url(r)
        if not any(overlaps(r,p) for p in training+holdout+gold_review) and key not in seen:unseen.append(r);seen.add(key)
    result=dict(training_hash=EXPECTED,validation_sha256=digest(args.validation.read_bytes()),feature_count=x.shape[1],family_sizes=families,conditions={},limitations=['bounded diagnostic interventions only; no promotion','human gold and historical audits already inspected; no candidate selection','new validation n=4 single source and provisional assistant labels','RSS is frozen historical snapshot; not newly fetched current100'])
    for name,c,w in [('c1',1,None),('c10',10,None),('family_c1',1,weights)]:
        clf=LogisticRegression(C=c,max_iter=1000,random_state=42).fit(x,labels,sample_weight=w)
        if name=='c1':
            import joblib
            saved=list((args.private_root/'iteration2/reviewed/artifacts').rglob('*.joblib'))
            if len(saved)!=1:raise ValueError('expected one original candidate artifact')
            original=joblib.load(saved[0]);delta=float(np.max(np.abs(original.predict_proba(texts)-clf.predict_proba(x))))
            if delta>1e-12:raise ValueError('original C1 parity failed')
            result['original_c1_max_probability_delta']=delta
        condition=dict(C=c,intercept=float(clf.intercept_[0]),zero_feature_probability=float(1/(1+np.exp(-clf.intercept_[0]))),coefficient_l2=float(np.linalg.norm(clf.coef_)),iterations=int(clf.n_iter_[0]),class_weight_sums={l:float(np.sum((weights if w is not None else np.ones(len(labels)))[np.array(labels)==l])) for l in set(labels)},evaluations={})
        for split,rows in [('training',ordered),('holdout',holdout),('gold',gold),('known_llm_audit',audit),('prospective_ms4',validation),('rss100',rss),('unseen62',unseen)]:condition['evaluations'][split]=measure(vectorizer,clf,rows)
        result['conditions'][name]=condition
    write_json(args.output,result);print('Saved bounded diagnostic: '+str(args.output))
if __name__=='__main__':main()
