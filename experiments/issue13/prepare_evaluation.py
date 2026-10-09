"""私有RSSから少数の独立評価レビュー候補を作る。学習・採点は行わない。"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from collections import Counter
from support import REPO,read_jsonl,write_json,digest,verify_protected_files,verify_boundaries,verify_rows
from gaiden_sieve.data import labeled_dataset_hash
from evaluation_design import groups,related,temporal_role

# 選択は記事内容と配信元内の対比による。モデル確率は参照しない。
SPECS=[
 ('MS01','ms',9,'core','not_relevant','量子チップの発表。提供されたRSSではAI用途を説明していない。'),
 ('MS02','ms',10,'core','relevant','生成AIとCopilotの企業活用事例。MS01と同日の同一配信元。'),
 ('MS03','ms',16,'core','not_relevant','セキュリティ会議の参加案内。RSSではAI用途を説明していない。'),
 ('MS04','ms',15,'core','relevant','言語モデルアプリのRAG手法を説明。MS03の翌日の同一配信元。'),
 ('GH01','gh',12,'expansion_unpaired','relevant','コード補完の独自モデル学習、強化学習、モデル更新。対比負例は未確保。'),
 ('AWS01','aws',74,'expansion_unpaired','relevant','Bedrockでの言語モデル提供開始。対比負例は未確保。'),
 ('BOUNDARY_UI','ui',64,'policy_only',None,'Copilot CLIのASCIIアニメーション、端末互換性とアクセシビリティの実装。'),
 ('BOUNDARY_INFRA','aws',71,'policy_only',None,'HyperPodの権限、共有容量、管理境界を扱う。以前隔離した既知例。'),
]

def verify_pool_hashes(paths,expected):
    if set(paths)!=set(expected):raise ValueError('locked pool names mismatch')
    for name,path in paths.items():
        if digest(path.read_bytes())!=expected[name]:raise ValueError('locked pool hash mismatch: '+name)

def main():
    p=argparse.ArgumentParser();p.add_argument('--private-root',type=Path,required=True);p.add_argument('--new-pool-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    verify_protected_files()
    if labeled_dataset_hash(args.private_root/'iteration2/reviewed-bootstrap')!='sha256:291d65aa8fac6aaf53812e4ac584b448fe8d2c258a9ef646c2f5b2acc76e2b18':raise ValueError('training hash mismatch')
    verify_rows(args.private_root/'iteration2/independent-llm-audit.jsonl','llm_audit')
    training,holdout,gold,audit=verify_boundaries(args.private_root/'iteration2/reviewed-bootstrap',args.private_root/'iteration2/independent-llm-audit.jsonl')
    probe=read_jsonl(REPO/'.diagnosis/issue13-causal/prospective-validation.jsonl');quarantine=read_jsonl(args.private_root/'iteration2/quarantined.jsonl')
    protected=training+holdout+gold+audit+probe+quarantine
    paths={'ms':args.new_pool_root/'ms-pool/candidate-pool.jsonl','gh':args.new_pool_root/'candidate-pool.jsonl','aws':args.private_root/'experiment/candidate-pool.jsonl','ui':REPO/'.diagnosis/issue13-causal/candidate-pool.jsonl'}
    from support import read_json
    lock=Path(__file__).parent/'lineage/independent-review-candidates.json'
    if lock.exists():verify_pool_hashes(paths,read_json(lock)['source_pool_sha256'])
    pools={k:read_jsonl(v) for k,v in paths.items()};selected=[];manifest=[]
    for review_id,pool,index,role,proposal,reason in SPECS:
        r=dict(pools[pool][index]);r.update(review_id=review_id,evaluation_role=role,proposed_label=proposal,proposal_source='assistant_rss_only',proposal_reason=reason,review_status='awaiting_human',human_label=None)
        selected.append(r)
    grouped,edges=groups(protected+selected)
    for r in selected:
        matches=[dict(id=q['id'],reason=kind,similarity=sim) for q in protected if (kind:=related(r,q)[0]) for sim in [related(r,q)[1]]]
        if r['evaluation_role']!='policy_only' and (matches or grouped[r['id']] in {grouped[q['id']] for q in protected}):raise ValueError('candidate overlaps protected group: '+r['review_id'])
        manifest.append(dict(review_id=r['review_id'],id=r['id'],source=r['source'],source_url=r['source_url'],published_at=r['published_at'],feed_url=r['feed_url'],raw_feed_sha256=r['raw_feed_sha256'],retrieved_at=r.get('retrieved_at'),ai_gaiden_commit='977c9a6b4be35e57f8d945eae7a84c65e5afa03f',title_sha256=digest(r['title']),summary_sha256=digest(r['summary']),title_chars=len(r['title']),summary_chars=len(r['summary']),evaluation_role=r['evaluation_role'],review_status=r['review_status'],human_label=None,proposed_label=r['proposed_label'],proposal_source=r['proposal_source'],proposal_reason=r['proposal_reason'],group_id=grouped[r['id']],protected_matches=matches,temporal_role=temporal_role(r,'2026-10-09T00:00:00Z')))
    args.output.mkdir(parents=True,exist_ok=True)
    private=args.output/'private-review-candidates.jsonl';private.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in selected))
    # レビューワークシートには予測確率・助手ラベル・由来sourceフィールドを表示しない。
    worksheet=['# 評価候補の人間レビュー（未採点）','title+summaryだけを読み、AI外電として採用/除外/情報不足を記入。記事本文やモデル予測は判断に加えない。','core4件だけが最小再開対象。残りは任意の拡張・政策例。']
    for r in selected:
        worksheet.extend(['\n## '+r['review_id'],r['title'],r['summary'],'判断: ______　理由: ______'])
    (args.output/'private-human-review.md').write_text('\n\n'.join(worksheet)+'\n')
    report=dict(status='awaiting_human_not_evaluation_gold',model_scoring_performed=False,model_training_performed=False,private_jsonl_sha256=digest(private.read_bytes()),source_pool_sha256={k:digest(v.read_bytes()) for k,v in paths.items()},core_count=4,core_proposed_labels=dict(Counter(r['proposed_label'] for r in selected if r['evaluation_role']=='core')),all_count=len(selected),near_template_threshold=.60,core_protected_overlap_count=0,core_source_count=1,pair_time_gap_hours={'MS01_MS02':.5,'MS03_MS04':23},future_cutoff='2026-10-09T00:00:00Z',embargo_days=7,future_evaluation_start='2026-10-16T00:00:00Z',limitations=['core is Microsoft-only n4, assistant-proposed labels pending human','all candidates predate frozen cutoff, not future holdout','pair balance must be recomputed from human labels','near-template heuristic does not prove semantic independence','unpaired expansion and policy-only rows excluded from primary metrics'],rows=manifest)
    write_json(args.output/'review-manifest.json',report)
    print('Prepared 4 core, 2 unpaired expansion, 2 policy-only candidates; no fitting/scoring')
if __name__=='__main__':main()
