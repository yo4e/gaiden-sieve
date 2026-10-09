"""配信元を跨ぐ重複、版番号違いと時刻境界の漏洩防止を検証する。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/issue13'))
from evaluation_design import groups,related,temporal_role

def row(i,title,summary='',url=None):
    return dict(id=i,title=title,summary=summary,source_url=url or 'https://example.com/'+i)

def test_cross_source_same_url_stays_in_one_group():
    a=row('a','first',url='https://example.com/post?utm_source=x')
    b=row('b','different',url='https://example.com/post')
    ids,edges=groups([a,b]);assert ids['a']==ids['b'];assert edges[0]['reason']=='identity'

def test_version_only_updates_stay_in_one_group():
    a=row('a','Patch version 4.1.2','We fixed local permissions and improved storage performance.')
    b=row('b','Patch version 4.1.3','We fixed local permissions and improved storage performance.')
    assert related(a,b)[0]=='normalized_content'

def test_near_template_and_disjoint_content():
    a=row('a','Platform alpha','alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi')
    b=row('b','Platform beta','alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron changed')
    c=row('c','Different subject','green forest carbon trees wildlife birds river')
    assert related(a,b)[0]=='near_template';assert related(a,c)[0] is None

def test_temporal_embargo_boundaries():
    cutoff='2026-10-09T00:00:00Z'
    for value,expected in [('2026-10-08T23:59:59Z','train_before_cutoff'),('2026-10-09T00:00:00Z','embargo'),('2026-10-15T23:59:59Z','embargo'),('2026-10-16T00:00:00Z','future_evaluation')]:
        assert temporal_role(dict(published_at=value),cutoff)==expected

def test_template_group_crossing_cutoff_is_purged():
    from evaluation_design import group_temporal_roles
    a=dict(row('a','Repeated template'),published_at='2026-10-08T12:00:00Z')
    b=dict(row('b','Repeated template'),published_at='2026-10-17T12:00:00Z')
    c=dict(row('c','Unrelated new story'),published_at='2026-10-18T12:00:00Z')
    ids,_=groups([a,b,c]);roles=group_temporal_roles([a,b,c],ids,'2026-10-09T00:00:00Z')
    assert roles=={'a':'purged','b':'purged','c':'future_evaluation'}

def test_changed_private_pool_hash_is_rejected(tmp_path):
    import pytest
    from prepare_evaluation import verify_pool_hashes
    from support import digest
    path=tmp_path/'pool.jsonl';path.write_text('original')
    expected={'source':digest(path.read_bytes())}
    verify_pool_hashes({'source':path},expected)
    path.write_text('different')
    with pytest.raises(ValueError,match='locked pool hash mismatch'):
        verify_pool_hashes({'source':path},expected)
