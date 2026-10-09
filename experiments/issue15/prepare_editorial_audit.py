"""6件・3境界の提案非提示票を作る。編集判断や新採点は行わない。"""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'issue13'))
from support import HERE,REPO,read_json,read_jsonl,write_json,digest,verify_protected_files

FORBIDDEN={'label','proposed_label','human_label','label_reason','proposal_reason','probability_relevant','p_relevant','pair','dataset_role'}

def visible_records(rows):
    # 文章はそのまま保ち、提案・予測・由来役割・ペア関係は回答票へ混ぜない。
    ordered=list(rows);random.Random(42).shuffle(ordered)
    return [dict(case_id=f'E{i+1:02}',title=r['title'],summary=r['summary']) for i,r in enumerate(ordered)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);args=p.parse_args()
    out=args.output_root.resolve()
    if not out.is_relative_to(REPO/'.diagnosis'):raise ValueError('raw editorial input must remain ignored/private')
    verify_protected_files()
    probe=read_jsonl(REPO/'.diagnosis/issue13-representation/new-probe.jsonl');probe_m=read_json(HERE/'lineage/representation-probe.json')
    if digest((REPO/'.diagnosis/issue13-representation/new-probe.jsonl').read_bytes())!=probe_m['file_sha256']:raise ValueError('representation probe hash mismatch')
    old=read_jsonl(REPO/'.diagnosis/issue13-independent-eval/review/private-review-candidates.jsonl');old_m=read_json(HERE/'lineage/independent-review-candidates.json')
    if digest((REPO/'.diagnosis/issue13-independent-eval/review/private-review-candidates.jsonl').read_bytes())!=old_m['private_jsonl_sha256']:raise ValueError('review pool hash mismatch')
    aws=read_jsonl(REPO/'profiles/ai-gaiden/bootstrap/01-relevant.jsonl')
    rows=[next(r for r in old if r['review_id']=='BOUNDARY_UI'),next(r for r in probe if r['review_id']=='NEW_GH02'),next(r for r in old if r['review_id']=='BOUNDARY_INFRA'),next(r for r in aws if r['id'].startswith('aws-machine-learning:url:95f314d9')),next(r for r in probe if r['review_id']=='NEW_MS01'),next(r for r in probe if r['review_id']=='NEW_MS02')]
    descriptions={r['id']:r for r in old_m['rows']}
    descriptions.update({r['record_metadata']['id']:r for r in probe_m['rows']})
    for r in rows:
        if r['id'] in descriptions:
            expected=descriptions[r['id']]
            if digest(r['title'])!=expected['title_sha256'] or digest(r['summary'])!=expected['summary_sha256']:raise ValueError('original title/summary hash mismatch')
    approved=set(__import__('score_approved_review').REFERENCES)
    if any(r.get('review_id') in approved for r in rows):raise ValueError('approved MS4 must not be re-requested')
    visible=visible_records(rows);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'private-input-sheet.json',visible)
    worksheet=['# 入力同条件の編集判断: 6件','まずこの票だけを読み、提案ラベル・予測・ペア対応は開かず記入してください。企業名は原入力内に残ります。既知例を含む編集監査で、未見の性能試験ではありません。','各例: 拾う / 拾わない / 情報不足 / 方針未決。根拠箇所や必要な背景も書いてください。URL先の本文を読んだ場合は、その追加情報を別に記録してください。']
    for r in visible:worksheet.extend(['## '+r['case_id'],'title: '+r['title'],'summary: '+r['summary'],'初回判断: ______','根拠箇所/必要な背景: ______','根拠の状態: 入力内根拠あり / 背景依存 / 情報不足 / 方針未決','境界方針確認後の判断（変化があれば）: ______'])
    (out/'private-input-sheet.md').write_text('\n\n'.join(worksheet)+'\n')
    write_json(out/'private-response-template.json',[dict(case_id=r['case_id'],initial_decision=None,evidence_status=None,supporting_text_or_required_background=None,after_policy_decision=None,reviewed_at=None) for r in visible])
    # 対応鍵はphase1の回答後に技術者が使う。提案ラベルやモデルスコアは鍵にも出さない。
    mapping=[]
    for v in visible:
        matches=[r for r in rows if r['title']==v['title'] and r['summary']==v['summary']]
        if len(matches)!=1:raise ValueError('ambiguous visible input identity')
        r=matches[0];mapping.append(dict(case_id=v['case_id'],id=r['id'],source=r['source'],source_url=r['source_url'],original_role='existing_teacher' if r['id'].startswith('aws-machine-learning:url:95f314d9') else 'previously_selected_or_scored_audit'))
    write_json(out/'private-after-response-key.json',mapping)
    manifest=dict(count=6,policy_boundaries=3,proposal_labels_shown=False,model_scores_shown=False,pairs_shown=False,original_text_hashes_verified=True,approved_ms4_excluded=True,human_review_performed=False,policy_decided_by_assistant=False,model_scoring_performed=False,model_training_performed=False,private_visible_json_sha256=digest((out/'private-input-sheet.json').read_bytes()),limitations=['known teacher/quarantine/probe examples; not unseen test or independent accuracy','source/brand names remain in original text; not source-blind','previous exposure cannot be undone by hiding labels now','initial decision and after-policy decision must remain separate'],rows=[dict(id=r['id'],source=r['source'],source_url=r['source_url'],title_sha256=digest(r['title']),summary_sha256=digest(r['summary']),title_chars=len(r['title']),summary_chars=len(r['summary'])) for r in rows])
    write_json(out/'preparation-manifest.json',manifest);print('Prepared six original-input items with no proposal/score/pair shown; human review pending')
if __name__=='__main__':main()
