"""編集方針承認をblind人間評価と区別し、凍結採点の境界を検証する。"""
import sys,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/issue13'))
from score_approved_review import score_model,locked_core,APPROVAL,REFERENCES
from gaiden_sieve.profile import load_profile
from support import digest

class FrozenModel:
    named_steps={'classifier':SimpleNamespace(classes_=['not_relevant','relevant']),'tfidf':SimpleNamespace(vocabulary_={'known':0},build_analyzer=lambda:lambda text:text.split())}
    def fit(self,*args,**kwargs):raise AssertionError('scoring must never fit')
    def predict(self,texts):return np.array(['relevant']*len(texts))
    def predict_proba(self,texts):
        p=[{'MS01':.70,'MS02':.75,'MS03':.69,'MS04':.74}[text.split()[0]] for text in texts]
        return np.array([[1-v,v] for v in p])

def test_score_keeps_binary_operational_and_ranking_separate():
    profile=load_profile(Path(__file__).resolve().parents[1]/'profiles/ai-gaiden/config.yml')
    rows=[dict(review_id=i,id=i,source='same',title=i,summary='known') for i in REFERENCES]
    result=score_model(FrozenModel(),profile,rows)
    assert result['binary_confusion']==dict(tp=2,fp=2,tn=0,fn=0)
    assert result['binary_metrics']['precision']==.5
    assert result['roc_auc']==1
    assert result['operational']==dict(distribution={'uncertain':4},precision=None,recall=0,retained_for_review_recall=1)
    assert result['oov_feature_rate']==.5

def test_policy_approval_is_explicitly_not_blind_annotation():
    assert APPROVAL['blind_review'] is False
    assert APPROVAL['input_independently_read'] is False
    assert APPROVAL['reference_origin']=='assistant_proposal_with_owner_policy_approval'

def test_changed_review_input_fails_before_scoring(tmp_path):
    path=tmp_path/'review.jsonl';rows=[dict(review_id=i,evaluation_role='core',title=i,summary='known') for i in REFERENCES]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest=dict(private_jsonl_sha256=digest(path.read_bytes()),rows=[dict(review_id=r['review_id'],title_sha256=digest(r['title']),summary_sha256=digest(r['summary'])) for r in rows])
    assert len(locked_core(path,manifest))==4
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='input hash mismatch'):locked_core(path,manifest)
