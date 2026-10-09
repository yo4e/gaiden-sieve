"""元59件で、URL欠落の実影響と保存OOFの割当をfitなしで監査する。"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'issue13'))
from support import HERE,REPO,read_json,read_jsonl,write_json,digest,canonical_url,verify_protected_files,verify_boundaries
from representation_probe import grouping_rows,grouped_splits
from evaluation_design import groups
from gaiden_sieve.data import load_labeled_jsonl,labeled_dataset_hash

EXPECTED='sha256:291d65aa8fac6aaf53812e4ac584b448fe8d2c258a9ef646c2f5b2acc76e2b18'

def main():
    p=argparse.ArgumentParser();p.add_argument('--training-data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    verify_protected_files()
    if labeled_dataset_hash(args.training_data)!=EXPECTED:raise ValueError('original59 training hash mismatch')
    raw,holdout,gold,_=verify_boundaries(args.training_data);items=load_labeled_jsonl(args.training_data)
    fixed=grouping_rows(items,raw);old=[{k:r[k] for k in ['id','source','title','summary','label']} for r in fixed]
    old_groups,_=groups(old);fixed_groups,_=groups(fixed)
    old_folds,_,_=grouped_splits(old);fixed_folds,_,_=grouped_splits(fixed)
    saved=read_json(HERE/'representation-results.json')['folds']
    assignments=[];url_duplicates=[];old_url_leak_pairs=set()
    for i,a in enumerate(fixed):
        for b in fixed[:i]:
            if canonical_url(a)==canonical_url(b):url_duplicates.append([a['id'],b['id']])
    for fold,((old_train,old_test),(new_train,new_test)) in enumerate(zip(old_folds,fixed_folds)):
        old_ids=[old[i]['id'] for i in old_test];new_ids=[fixed[i]['id'] for i in new_test]
        for i in old_train:
            for j in old_test:
                if canonical_url(fixed[i])==canonical_url(fixed[j]):old_url_leak_pairs.add(tuple(sorted([fixed[i]['id'],fixed[j]['id']])))
        train_urls={canonical_url(fixed[i]) for i in new_train};test_urls={canonical_url(fixed[i]) for i in new_test}
        assignments.append(dict(fold=fold,test_ids=new_ids,old_fixed_assignment_equal=old_ids==new_ids,saved_assignment_equal=new_ids==saved[fold]['test_ids'],url_overlap_count=len(train_urls&test_urls)))
    changed=sum((old_groups[a['id']]==old_groups[b['id']])!=(fixed_groups[a['id']]==fixed_groups[b['id']]) for i,a in enumerate(fixed) for b in fixed[:i])
    output=dict(training_hash=EXPECTED,rows=len(raw),url_present=sum(bool(r.get('source_url')) for r in raw),normalized_url_unique=len({canonical_url(r) for r in fixed}),same_url_pairs=url_duplicates,old_group_count=len(set(old_groups.values())),fixed_group_count=len(set(fixed_groups.values())),pair_membership_changes=changed,folds=assignments,actual_url_leak_pairs_across_oof_folds=len(old_url_leak_pairs),model_fit_performed=False,protected_files_unchanged=True,limitations=['audit is original59 OOF URL identity; not all semantic/template/time leakage','lack of impact in original59 does not invalidate synthetic regression bug','no model retraining or metrics recomputation'])
    write_json(args.output,output);print('Saved original59 URL audit: '+str(args.output))
if __name__=='__main__':main()
