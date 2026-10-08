"""固定条件で再評価する独立診断用CLI。公開結果に私有本文を埋め込まない。"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import numpy as np
from support import HERE,REPO,read_json,read_jsonl,write_json,verify_rows,verify_boundaries,verify_protected_files,overlaps
from gaiden_sieve.profile import load_profile
from gaiden_sieve.artifacts import load_candidate_model
from gaiden_sieve.data import load_labeled_jsonl,labeled_dataset_hash,build_text
from gaiden_sieve.train import train_candidate,build_pipeline,evaluate_labeled_items
from gaiden_sieve.classify import predict_one
from gaiden_sieve.reproducibility import verify_candidate_reproducibility
from gaiden_sieve.shadow import run_shadow_candidate

def score(model,items,profile):
    rows=[]
    for item in items:
        p=predict_one(model,profile,title=item.title,summary=item.summary)
        rows.append(dict(id=item.id,source=item.source,label=item.label,binary_prediction=str(model.predict([build_text(item,profile.text_fields)])[0]),**asdict(p)))
    positives=sum(r['label']=='relevant' for r in rows);predicted=[r for r in rows if r['classification']=='relevant'];tp=sum(r['label']=='relevant' for r in predicted)
    return dict(count=len(items),binary_metrics=asdict(evaluate_labeled_items(model,items,profile)),operational_metrics=dict(precision=tp/len(predicted) if predicted else None,recall=tp/positives,distribution=dict(Counter(r['classification'] for r in rows))),rows=rows)

def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--baseline-only',action='store_true');group.add_argument('--local-results-root',type=Path);group.add_argument('--hydrated-root',type=Path)
    parser.add_argument('--input',type=Path,help='optional historical frozen input; exact hash required')
    parser.add_argument('--output',type=Path,default=REPO/'.diagnosis/issue13-replay');args=parser.parse_args()
    verify_protected_files();expected=read_json(HERE/'results.json');profile=load_profile(REPO/'profiles/ai-gaiden/config.yml')
    dataset_paths={'baseline':REPO/'profiles/ai-gaiden/bootstrap'};audit_path=None;input_path=args.input
    if args.local_results_root:
        root=args.local_results_root.resolve();dataset_paths.update(iteration1=root/'experiment/augmented-bootstrap',iteration2=root/'iteration2/reviewed-bootstrap');audit_path=root/'iteration2/independent-llm-audit.jsonl';input_path=input_path or root/'experiment/incoming.jsonl'
    elif args.hydrated_root:
        root=args.hydrated_root.resolve();dataset_paths.update(iteration1=root/'iteration1',iteration2=root/'iteration2');audit_path=root/'llm_audit'/'independent-llm-audit.jsonl'
    if audit_path:verify_rows(audit_path,'llm_audit')
    if input_path:verify_rows(input_path,'frozen_rss')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True);summary={}
    for name,path in dataset_paths.items():
        expected_model=expected[name]
        if name!='baseline':
            manifest=read_json(HERE/'lineage'/f'{name}.json');verify_rows(path/manifest['filename'],name)
        if labeled_dataset_hash(path)!=expected_model['candidate_metadata']['training_data_hash']:raise ValueError(f'{name}: training dataset hash mismatch')
        training_rows,holdout_rows,gold_rows,audit_rows=verify_boundaries(path,audit_path)
        model_out=output/name;artifacts=model_out/'artifacts';items=load_labeled_jsonl(path);holdout=REPO/'profiles/ai-gaiden/real-holdout'
        metadata=train_candidate(items=items,data_path=path,profile=profile,artifacts_dir=artifacts,evaluation_items=load_labeled_jsonl(holdout),evaluation_data_path=holdout)
        write_json(model_out/'train.json',metadata)
        verification=verify_candidate_reproducibility(data_path=path,profile=profile,artifacts_dir=artifacts,model_version=metadata['model_version'],evaluation_data_path=holdout)
        if not verification['passed']:raise ValueError(f'{name}: candidate verification failed')
        write_json(model_out/'verification.json',verification)
        model=load_candidate_model(artifacts_dir=artifacts,profile=profile.profile,model_version=metadata['model_version'])
        gold=score(model,load_labeled_jsonl(REPO/'profiles/ai-gaiden/gold.jsonl'),profile);write_json(model_out/'gold.json',gold)
        texts=[build_text(item,profile.text_fields) for item in items];retrained=build_pipeline();retrained.fit(texts,[item.label for item in items]);delta=float(np.max(np.abs(model.predict_proba(texts)-retrained.predict_proba(texts))))
        if delta>1e-12:raise ValueError('refit probabilities differ')
        checks=dict(holdout_metrics_match=metadata['metrics']==expected_model['holdout_metrics'],gold_binary_metrics_match=gold['binary_metrics']==expected_model['gold']['binary_metrics'],gold_operational_metrics_match=gold['operational_metrics']==expected_model['gold']['operational_metrics'],refit_max_probability_difference=delta)
        if audit_path:
            audit=score(model,load_labeled_jsonl(audit_path),profile);write_json(model_out/'llm-audit.json',audit);checks['llm_audit_binary_metrics_match']=audit['binary_metrics']==expected_model['llm_audit']['binary_metrics'];checks['llm_audit_operational_metrics_match']=audit['operational_metrics']==expected_model['llm_audit']['operational_metrics']
        train_score=score(model,items,profile);write_json(model_out/'training-diagnostic.json',train_score)
        if input_path:
            input_rows=read_jsonl(input_path);unseen=[r for r in input_rows if not any(overlaps(r,p) for p in training_rows+holdout_rows+gold_rows)];unique=[];seen=set()
            from support import canonical_url
            for row in unseen:
                key=canonical_url(row)
                if key not in seen:unique.append(row);seen.add(key)
            for split,rows in [('full',input_rows),('unseen-unique-url',unique)]:
                folder=model_out/split;folder.mkdir(parents=True,exist_ok=True);snapshot=folder/'private-input.jsonl';snapshot.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
                run_shadow_candidate(profile=profile,artifacts_dir=artifacts,model_version=metadata['model_version'],input_path=snapshot,classified_output=folder/'private-classified.jsonl',drift_output=folder/'drift.json',uncertain_output=folder/'private-uncertain.jsonl',comparison_output=folder/'comparison.json')
                drift=read_json(folder/'drift.json');target=expected_model['shadow'][split];checks[f'{split}_distribution_match']=drift['classification_distribution']==target['classification_distribution'];checks[f'{split}_oov_match']=abs(drift['oov_feature_rate']-target['oov_feature_rate'])<1e-12
        failures=[k for k,v in checks.items() if k.endswith('_match') and not v]
        summary[name]=dict(checks=checks,passed=not failures,shadow_replayed=input_path is not None)
        if failures:raise ValueError(f'{name}: historical result mismatches: {failures}')
    write_json(output/'replay-checks.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
