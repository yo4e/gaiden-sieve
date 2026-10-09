"""Issue 15のレビュー用検算。学習・API呼び出し・元データ変更は行わない。"""
from __future__ import annotations
import hashlib
import json
import math
import sys
from pathlib import Path
from fractions import Fraction

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = REPO / "experiments" / "issue13"
OUTPUT = REPO / ".diagnosis" / "issue15-review-checks"
COMMIT = '59166ad4c2a70a54355e0c34dcfd3ea9f91b5e24'
# Connectorで読んだ固定コミットの公開スコアを、原文なしで転記した小さな監査用抜粋。
PROBE_IDS = ['NEW_MS01', 'NEW_MS02', 'NEW_MS03', 'NEW_MS04', 'NEW_GH01', 'NEW_GH02']
LABELS = [0, 1, 0, 1, 0, 1]
WORD = [0.7239523574861466, 0.7327932854900033, 0.6891421484270277, 0.7372970697690677, 0.7235573712670058, 0.74622868132238]
CHAR = [0.7307515322760967, 0.7188703861597883, 0.6676869263591947, 0.7068032705587973, 0.7029465791078086, 0.7626718360369785]
APPROVED = [0.7017094225753593, 0.7501415621769115, 0.6964905392462379, 0.7446836948862532]
GOLD_CHAR = [0.7265741223393425, 0.7424138841001778, 0.747638639013498, 0.700014192919883, 0.7197518595548258, 0.7236253763523831, 0.24639237708864814, 0.24639237708864814, 0.2241553005755935, 0.22505590937727593]
GOLD_LABELS = [0, 1, 0, 1, 1, 1, 0, 0, 0, 0]


def auc(y: list[int], p: list[float]) -> dict:
    positive = [v for t, v in zip(y, p) if t == 1]
    negative = [v for t, v in zip(y, p) if t == 0]
    if not positive or not negative:
        return {'auc': None, 'pairs': 0, 'wins': 0, 'ties': 0}
    wins = sum(a > b for a in positive for b in negative)
    ties = sum(a == b for a in positive for b in negative)
    pairs = len(positive) * len(negative)
    return {'auc': (wins + ties / 2) / pairs, 'pairs': pairs, 'wins': wins, 'ties': ties}


def metrics(y: list[int], p: list[float]) -> dict:
    assert len(y) == len(p) and all(math.isfinite(x) and 0 <= x <= 1 for x in p)
    predicted = [x > .5 for x in p]
    tp = sum(t == 1 and v for t, v in zip(y, predicted))
    fp = sum(t == 0 and v for t, v in zip(y, predicted))
    tn = sum(t == 0 and not v for t, v in zip(y, predicted))
    fn = sum(t == 1 and not v for t, v in zip(y, predicted))
    state = ['R' if x >= .8 else 'N' if x < .4 else 'U' for x in p]
    accepted = [i for i, s in enumerate(state) if s == 'R']
    return dict(n=len(y), tp=tp, fp=fp, tn=tn, fn=fn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                distribution={s: state.count(s) for s in ['R','U','N']},
                operational_precision=sum(y[i] for i in accepted)/len(accepted) if accepted else None,
                operational_recall=sum(y[i] for i in accepted)/sum(y) if sum(y) else None,
                retained_recall=sum(t == 1 and s != 'N' for t,s in zip(y,state))/sum(y) if sum(y) else None,
                **auc(y,p))


def main() -> None:
    expected = {'support.py':'fd3a88c2f8a7158528b8fbc9dc08cd8d58283ea9', 'evaluation_design.py':'03af183cb1bb2272ba57322e8fbb299934ea847f'}
    verified = {}
    for name, sha in expected.items():
        b = (SOURCE/name).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        assert actual == sha, (name, actual, sha)
        verified[name] = actual
    sys.path.insert(0, str(SOURCE))
    from evaluation_design import groups, related, group_temporal_roles
    # 実記事ではない合成例で、OOF入力への射影でURL境界が消えることを確認する。
    rows = [dict(id='feed-a:url:shared',source='feed-a',source_url='https://example.invalid/article',title='A model can reason about geometrical structures',summary='Research evaluation investigates theorem proving.',label='relevant'),
            dict(id='feed-b:url:shared',source='feed-b',source_url='https://example.invalid/article?utm_source=b',title='A fresh announcement from the laboratory',summary='Quarterly project bulletin and institutional updates.',label='relevant')]
    full = groups(rows)[0]
    projected = groups([{k:r[k] for k in ['id','source','title','summary','label']} for r in rows])[0]
    assert len(set(full.values())) == 1
    assert len(set(projected.values())) == 2
    # 同一文章ならURLなしでも検出される。欠落の影響範囲を狭める対照。
    same_text = [rows[0], dict(rows[0], id='feed-c:url:shared', source='feed-c')]
    assert len(set(groups([{k:r[k] for k in ['id','source','title','summary','label']} for r in same_text])[0].values())) == 1
    temporal = [dict(rows[0], published_at='2026-10-08T23:59:59Z'), dict(rows[1], published_at='2026-10-16T00:00:00Z')]
    assert set(group_temporal_roles(temporal,full,'2026-10-09T00:00:00Z').values()) == {'purged'}

    new_probe = {name:metrics(LABELS,p) for name,p in [('word',WORD),('char',CHAR)]}
    for name,p in [('word',WORD),('char',CHAR)]:
        new_probe[name]['microsoft_auc'] = auc(LABELS[:4],p[:4])
        new_probe[name]['github_auc'] = auc(LABELS[4:],p[4:])
        new_probe[name]['paired_correct'] = sum(p[i+1]>p[i] for i in [0,2,4])
        new_probe[name]['pair_gaps'] = [p[i+1]-p[i] for i in [0,2,4]]
        new_probe[name]['micro_oov'] = (sum([76,51,40,46,88,47])/sum([107,109,69,89,145,99]) if name=='word' else sum([219,101,119,115,285,72])/sum([814,810,529,610,804,649]))
    assert new_probe['word']['auc'] == 1
    assert math.isclose(new_probe['char']['auc'],7/9)
    assert new_probe['char']['microsoft_auc']['auc'] == .5
    assert new_probe['char']['paired_correct'] == 2
    assert all(x['distribution'] == {'R':0,'U':6,'N':0} for x in new_probe.values())
    approved=metrics(LABELS[:4], APPROVED)
    approved['pair_gaps']=[APPROVED[1]-APPROVED[0],APPROVED[3]-APPROVED[2]]
    approved['micro_oov']=(76+88+53+91)/(117+151+115+139)
    assert math.isclose(approved['micro_oov'],.5900383141762452)
    gold_char=metrics(GOLD_LABELS,GOLD_CHAR)
    assert math.isclose(gold_char['auc'],17/24)
    assert math.isclose(gold_char['precision'],2/3)
    # ここは保存されたfold集計からの計算。59行の再学習/全OOFスコア再計算ではない。
    fold_counts=[dict(fold=0,positive=13,sourcegraph_negative=7,other_negative=0),dict(fold=1,positive=13,sourcegraph_negative=0,other_negative=7),dict(fold=2,positive=12,sourcegraph_negative=5,other_negative=2)]
    assert sum(r['positive'] for r in fold_counts)==38
    pair_counts=[r['positive']*(r['sourcegraph_negative']+r['other_negative']) for r in fold_counts]
    within_fold={}
    for name,wins in [('word',[91,55,84]),('char',[91,57,84])]:
        within_fold[name] = {'same_fold_pairs':sum(pair_counts),'pair_credit_including_half_ties':sum(wins),'pair_weighted_auc':float(Fraction(sum(wins),sum(pair_counts))), 'macro_fold_auc':sum(float(Fraction(w,n)) for w,n in zip(wins,pair_counts))/3}
    output=dict(commit=COMMIT,source_blob_checks=verified,
        new_probe=new_probe,approved_ms4=approved,known_gold_char=gold_char,
        grouping_synthetic_checks={'with_url_groups':1,'oof_projected_groups':2,'identical_text_still_detected':True,'cross_time_group_purged':True,'actual_59_row_url_leakage':'not_tested'},
        fold_counts=fold_counts,within_fold_auc_from_published_fold_summaries=within_fold,
        source_rule_binary_from_counts={'tp':38,'fp':9,'tn':12,'fn':0,'precision':38/47,'recall':1.0,'f1':76/85},
        limitations=['No model fit, artifact load or full pytest run', 'Numeric input subset transcribed from fixed-commit connector reads', 'Full historical source inputs and current sklearn environment are not identical', 'Synthetic grouping failure is not proof of leakage in original data'])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'review-checks.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUTPUT/'score-extract.json').write_text(json.dumps(dict(commit=COMMIT,probe_ids=PROBE_IDS,labels=LABELS,word=WORD,char=CHAR,approved_ms4=APPROVED,gold_char=GOLD_CHAR,gold_labels=GOLD_LABELS),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
