"""文字の未知語耐性と、表現比較の条件・群分割を検証する。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/issue13'))
from representation_probe import build_model,grouped_splits

def test_partial_word_reuses_character_fragments_without_word_match():
    word=build_model('word12_c1').named_steps['tfidf'];char=build_model('charwb35_c1').named_steps['tfidf']
    texts=['prediction predictive','regularization regularized']
    word.fit(texts);char.fit(texts)
    assert word.transform(['predictions']).nnz==0
    assert char.transform(['predictions']).nnz>0

def test_only_representation_changes_between_conditions():
    a=build_model('word12_c1');b=build_model('charwb35_c1')
    assert a.named_steps['classifier'].get_params()==b.named_steps['classifier'].get_params()
    assert a.named_steps['classifier'].C==1
    assert a.named_steps['classifier'].random_state==42
    assert b.named_steps['tfidf'].analyzer=='char_wb'
    assert b.named_steps['tfidf'].ngram_range==(3,5)

def test_grouped_oof_never_splits_version_siblings():
    topics=['apple pear grape plum fig peach','star orbit comet planet galaxy nebula','violin piano cello flute drum trumpet','river ocean lake rain cloud snow','socket packet network router cable switch','grain wheat rice bread corn oats']
    rows=[]
    for k,topic in enumerate(topics):
        for version in [1,2]:rows.append(dict(id=f'{k}-{version}',source_url=f'https://example.com/{k}/{version}',source='one',title=f'{topic} release {version}',summary=topic,label='relevant' if k<3 else 'not_relevant'))
    folds,keys,_=grouped_splits(rows)
    assert len(folds)==3 and len(set(keys))==6
    seen=[]
    for train,test in folds:
        assert not {keys[i] for i in train}&{keys[i] for i in test}
        seen.extend(test)
    assert sorted(seen)==list(range(12))

def test_fitting_training_only_does_not_add_evaluation_word():
    model=build_model('word12_c1');model.fit(['positive alpha','negative beta'],['relevant','not_relevant'])
    model.predict_proba(['unseenholdoutonly'])
    assert 'unseenholdoutonly' not in model.named_steps['tfidf'].vocabulary_
