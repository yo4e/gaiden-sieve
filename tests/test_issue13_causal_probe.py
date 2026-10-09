"""クラス総重みの保持と、切片を含めたlogit分解を検証する。"""
import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/issue13'))
from causal_probe import family_weights,decompose

def test_families_preserve_class_mass_and_equalize_duplicate_family():
    labels=['not_relevant']*3+['relevant']*2
    weights,sizes=family_weights(['same template','same template','other negative','positive one','positive two'],labels,TfidfVectorizer().build_analyzer())
    assert sizes==[2,1,1,1]
    assert weights[:3].sum()==pytest.approx(3)
    assert weights[3:].sum()==pytest.approx(2)
    assert weights[0]+weights[1]==pytest.approx(weights[2])

def test_conflicting_family_fails():
    with pytest.raises(ValueError,match='mixed-label'):
        family_weights(['same','same'],['relevant','not_relevant'],TfidfVectorizer().build_analyzer())

def test_logit_sum_includes_intercept_and_zero_feature_document():
    x=TfidfVectorizer().fit_transform(['positive helpful','negative irrelevant','positive useful','negative useless','unknown'])
    clf=LogisticRegression().fit(x[:4],['relevant','not_relevant','relevant','not_relevant'])
    pos,neg,logits=decompose(x,clf)
    assert np.all(pos>=0) and np.all(neg<=0)
    np.testing.assert_allclose(logits,clf.decision_function(x))
    assert logits[-1]==pytest.approx(clf.intercept_[0])
