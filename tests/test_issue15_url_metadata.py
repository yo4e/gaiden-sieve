"""OOF呼び出しでURLを保持し、特徴へは混ぜない回帰検証。"""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/issue13'))
from representation_probe import grouping_rows
from evaluation_design import groups
from gaiden_sieve.data import build_input_text

def example_rows():
    return [dict(id='a',source='one',source_url='https://example.invalid/article',title='Geometry model reasoning',summary='Research into proof systems.',label='relevant'),dict(id='b',source='two',source_url='https://example.invalid/article?utm_source=two',title='Laboratory bulletin',summary='Quarterly institutional updates.',label='relevant')]

def test_oof_projection_preserves_cross_source_url_identity():
    raw=example_rows();items=[SimpleNamespace(**{k:r[k] for k in ['id','source','title','summary','label']}) for r in raw]
    old=[dict(vars(i)) for i in items]
    assert len(set(groups(old)[0].values()))==2
    fixed=grouping_rows(items,raw)
    assert len(set(groups(fixed)[0].values()))==1
    assert fixed[1]['source_url']==raw[1]['source_url']

def test_missing_url_fails_instead_of_silent_projection():
    raw=example_rows();items=[SimpleNamespace(**r) for r in raw];raw[0].pop('source_url')
    with pytest.raises(ValueError,match='source_url missing'):grouping_rows(items,raw)

def test_grouping_metadata_does_not_become_model_text():
    raw=example_rows();fixed=grouping_rows([SimpleNamespace(**r) for r in raw],raw)
    for r in fixed:
        text=build_input_text(title=r['title'],summary=r['summary'],text_fields=('title','summary'))
        assert 'example.invalid' not in text
