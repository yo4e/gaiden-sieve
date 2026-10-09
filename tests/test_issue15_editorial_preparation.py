"""提案・予測・ペア対応を出さず、原入力だけを保持する票を検証する。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/issue15'))
from prepare_editorial_audit import visible_records,FORBIDDEN

def test_visible_sheet_removes_proposals_scores_and_pair_keys():
    rows=[dict(title='Exact input '+str(i),summary='Original summary\n'+str(i),label='relevant',proposed_label='not_relevant',label_reason='hidden',probability_relevant=.9,pair='hidden',source='hidden',id=str(i)) for i in range(6)]
    visible=visible_records(rows)
    assert len(visible)==6
    assert visible==visible_records(rows)
    assert {r['title'] for r in visible}=={r['title'] for r in rows}
    assert all(set(r)=={'case_id','title','summary'} for r in visible)
    assert all(not set(r)&FORBIDDEN for r in visible)
    original={r['title']:r['summary'] for r in rows}
    assert all(r['summary']==original[r['title']] for r in visible)
