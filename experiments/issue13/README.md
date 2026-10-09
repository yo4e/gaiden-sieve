# Issue #13 — 別モデルによる独立診断の入口

**2回の局所データ追加では改善を確認できなかった。3回目の同じ追加反復は止める。** このディレクトリは、別モデルが「特徴の拾い方」「少量・偏ったデータ」「評価の前提」を独立に疑い、次の実験を設計するための引き継ぎである。

学習者向けの短い説明は [learning-notes.md](learning-notes.md)、実作成した独立評価候補と最小確認は [independent-evaluation-plan.md](independent-evaluation-plan.md)。

対象: [Issue #13](https://github.com/yo4e/gaiden-sieve/issues/13)。まずこの文書、次に [results.json](results.json)、[prediction-comparison.json](prediction-comparison.json)、既存 [DESIGN.md](../../DESIGN.md)、[train.py](../../src/gaiden_sieve/train.py)、[data.py](../../src/gaiden_sieve/data.py)、[drift.py](../../src/gaiden_sieve/drift.py) を読む。

## 固定した条件

| 条件 | 値 |
| --- | --- |
| SIEVE core | `6e1da50285a753c5ffef4ca6d3bfe43fafcfd1f5` |
| AI外電 read-only ingestion | `977c9a6b4be35e57f8d945eae7a84c65e5afa03f` |
| input features | title + summary。label/reason/source/URLは特徴へ混ぜない |
| model | TF-IDF word unigram/bigram + LogisticRegression(C=1, max_iter=1000, seed=42) |
| operational threshold | relevant: P≥0.80、not_relevant: P<0.40、それ以外uncertain |
| quality gate | binary precision≥0.95 / recall≥0.80 |
| external holdout | 10件、LLM R6/N4、全て評価専用 |
| human gold | 10件、人間R4/N6、学習から分離、監査用でgateの代替ではない |
| environment | Python3.12.13、scikit-learn1.9.1、joblib1.6.0。他は[requirements.txt](requirements.txt) |

profile、gold、gold-review、holdout、既存bootstrapとcoreのhashは [protected-file-hashes.json](protected-file-hashes.json)。本PRではこれらを変更していない。運用model/threshold/gate/promotion policyの変更、Phase4C、本番接続、外部AIサービスへの送信はしていない。追加診断では局所的なC値・重みの比較だけを行い、採用はしていない。

## 何を試したか

1. **baseline**: 既存bootstrap36件（R24/N12）を全件fitし、明示holdoutで評価。
2. **iteration1**: AI系source内のhard negative / boundary例を15件（R8/N7）追加し、51件（R32/N19）でfit。Google AI・AWS ML・GitHub AI & ML。追加は助手自身のRSS-only判断で `label_source=llm / review_status=provisional_llm`。別AI API・人間確認はない。
3. **iteration2**: iteration1の3曖昧例（映画紹介、画像作成編集、ML基盤一般管理）と、情報不足2例（caption、一般インタビュー）を隔離。残り10件＋新規13件（R7/N6）で59件（R38/N21）。Microsoft Cloudの同sourceにR5/N5、GitHub内の一般Git機能とAI tooling等を対比させた。
4. 新規LLM監査8件（R6/N2）をiteration2のfit前に固定した。教師/gold/holdoutからid・正規化URL・title・title+summaryで分離。これは同じ助手の暫定ラベルであり、human precisionとは呼ばない。

各回の選定を結果を見る前に固定し、各回1候補だけ学習した。評価後にgoldに合わせたラベル変更・再学習はしていない。ただしgoldの既知誤判定を課題として知っているため、goldを未見の最終testとは呼ばない。

## 何が失敗したか

同じ2026-10-09早朝取得のRSS100行で比較。R/U/Nはrelevant/uncertain/not_relevant。

| 指標 | baseline | iteration1 | iteration2 |
| --- | ---: | ---: | ---: |
| train R/N | 24/12 | 32/19 | 38/21 |
| holdout binary P/R/F1 | 1/1/1 | 1/1/1 | 1/1/1 |
| human gold binary P/R/F1 | 0.6667/1/0.8 | 0.6667/1/0.8 | 0.6667/1/0.8 |
| gold operational relevant recall | 1.0 | 0 | 0 |
| RSS100行 R/U/N | 80/0/20 | 0/80/20 | 0/80/20 |
| uncertain rate | 0% | 80% | 80% |
| OOV feature rate | 0.808165 | 0.790072 | 0.790934 |
| 独立URL62件 R/U/N | 62/0/0 | 0/62/0 | 0/62/0 |
| 独立URL62件 OOV | 0.826365 | 0.808439 | 0.811661 |
| 非Sourcegraph not_relevant | 0 | 0 | 0 |
| LLM監査8件 binary P/R/F1 | 0.75/1/0.8571 | 0.75/1/0.8571 | 0.75/1/0.8571 |

100行の33行は既存train/holdout/goldと一致し、残り67行を別評価。Microsoft AI/Cloud間の同一URL5組を除いて62件。追加教師は元100行と重複0。100/67/62は同じスナップショットの部分集合で、独立した3回の取得ではない。

Sourcegraph20件は全て既知のtrain/holdout/gold記事。全20件N、他の全80件はiteration1/2でUというsource内一律分類が続いた。16非空sourceの分布・平均確率はresults.jsonに保存。17source取得成功・失敗0、admission accepted75/rejected25。

iteration2で取り直した最新RSSは2件（Hugging Face / LangChain）が入れ替わったが、3candidateとも上表と同じ分布。iteration2の最新OOVは0.790690。前回比較の主表は、日付の入力差を混ぜず元スナップショットで測った。

**binaryとthresholdを混同しない。** 既存evaluateは`model.predict`の二値判定。goldは3候補ともTP4/FP2/FN0/TN4で、recall=4/4=1。0.80の固定thresholdでは追加後の真のR4件は全てUで、確信Rとしてのrecall=0/4。R予測0なのでprecisionは未定義。Uを人間レビュー候補として保持するなら4件とも候補には残る。

## 未解明な点と診断で疑う入口

### 観測済み

**訂正:** 教師59件は正例min .750557 > 負例max .619505、AUC1.0で順位分離している。以下のN9件がbinary Rという事実は、固定境界での失敗を表し、順位を学べていない証拠ではない。追加の局所診断は [causal-diagnosis.md](causal-diagnosis.md) と [causal-results.json](causal-results.json) を参照。

- iteration2の教師上でも非SourcegraphのN9件が全てbinary Rのまま。P(R)約0.587–0.620、thresholdでは全てU。教師上のconfusionはTP38/FP9/TN12/FN0で、TN12はSourcegraphだけ。この値はresubstitution診断であり汎化成績ではない。
- 補助監査のN2件はP(R)0.7241/0.6921。真のRと暫定ラベルしたAI bakery例は0.6962。正負のrankingが重なり、単にthresholdを下げればきれいに分かれる状態ではない。
- 現行featureはtitle+summaryのみ。source shortcutの疑いはsource名フィールドへの直接依存ではなく、本文内のsource固有語彙・定型文・データ構成に関する仮説。
- holdoutのN4件はSourcegraph patchへ偏り、類似テンプレートがtrainにもある。gate passはhard negativeへの汎化を証明しない。

### 原因として未検証

Sourcegraph近似テンプレートの反復、少数・広範な語彙の各記事、OOV約0.79、既定正則化、クラス比率、RSSの疎いcaption、bootstrapラベルの意味的整合性。当初の2回ではどれが支配的かのablationはしていない。その後のC値と同一analyzer-family重みの局所比較は [causal-diagnosis.md](causal-diagnosis.md) に追記した。別モデルなら改善するとの証拠もない。

### 次のモデルへの依頼

1. まずデータ境界・metric定義・score処理を独立に点検する。公開された暫定ラベルを無批判にgold扱いしない。
2. 上の仮説を切り分ける最小の診断計画を提案する。少数追加で同じ観測を繰り返すことは避ける。templateの実効重複、source内対比、時間/source/内容クラスによる評価分割を疑う。
3. 現行方式の教師上の順位分離と、固定境界での判定・評価集合の弱さを別々に説明する。
4. モデル/特徴量/正則化/thresholdの変更、追加の学習・データ取得、外部AI送信、本番接続は、本引き継ぎの診断開始権限と混同しない。必要な次の実験は目的と条件を示して別判断へ戻す。

山田さんへ15件/23件全件の確認を戻すことは前提にしない。資料で扱える例は助手が整理し、隔離例の編集方針が実験に必要になった時だけ狭い政策判断や少数のspot checkを求める。

## 公開・非公開境界

**新規第三者title/summary本文、生RSS、私有cache、model/joblib、ZIP、端末絶対パス、Slackログ・tokenは本PRに含めない。** 既存repoのbootstrap/gold等は本PRでは変更・再コピーしていない。追加候補と入力は、URL、id、文字数、title/summary SHA256、由来feed URL/hash、日時、助手のlabel/reason、article別確率を公開する。文章hashは文章の配布権利の代替ではない。

- [lineage/iteration1.json](lineage/iteration1.json)、[iteration2.json](lineage/iteration2.json): 追加教師の本文なしmanifest。
- [quarantine.json](lineage/quarantine.json): 学習に使わない5例。
- [llm_audit.json](lineage/llm_audit.json): 独立補助監査8例のmanifest。
- [frozen_rss.json](lineage/frozen_rss.json)、[latest_rss.json](lineage/latest_rss.json): 2スナップショットの本文なしmanifest。
- [feed-observations.json](lineage/feed-observations.json): 取得先、取得観測、元feed hash。Google/AWSの無効なpagination・同一feed返却も記録。

**公開metadataだけから消えた過去RSS本文を逆算することはできない。** 独立モデルはすぐにcode/metrics/確率/由来の診断とbaseline再現を開始できる。2回の全データの完全再生には、権利上共有可能な私有入力、または公式RSSからhash一致の本文が再取得できることが必要。取得不能な行を別記事・新summary・新ラベルで埋めて「同じ実験」と扱わない。

## 再現手順と確認済み範囲

repo rootでPython3.12のvenvを作り、以下を実行する。`.diagnosis/`はgitignore対象で、復元本文・モデル・cacheはそこに保存する。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r experiments/issue13/requirements.txt
.venv/bin/python -m pip install -e '.[dev]'
# 公開repoだけで開始できるbaselineのholdout/gold/再現性診断
.venv/bin/python experiments/issue13/replay.py --baseline-only
```

2回の追加データを公式RSSから復元できる場合:

```bash
git clone https://github.com/yo4e/AI-gaiden.git .diagnosis/AI-gaiden
git -C .diagnosis/AI-gaiden checkout 977c9a6b4be35e57f8d945eae7a84c65e5afa03f
.venv/bin/python experiments/issue13/hydrate.py \
  --ai-gaiden-root .diagnosis/AI-gaiden --download-feeds
.venv/bin/python experiments/issue13/replay.py \
  --hydrated-root .diagnosis/issue13-hydrated
```

hydrateは原文記事ページを取得せず公式RSSだけを取得する。title/summaryのhashと追加JSONLのfile hashが一致しない行があれば、そのdatasetを完全復元扱いしない。`hydration-status.json`のmissing_idsを確認する。RSSが変わり、このコマンドが今後失敗する可能性は残る。

原実験の私有cacheがある場合、hydrateの`--download-feeds`を`--feed-cache <private-cache-directory>`に置き換えられる。原実験の私有結果ディレクトリ（experiment/とiteration2/を含む）がある場合は、次のコマンドでfrozen100行と62独立URLのshadowまで再現する。

```bash
.venv/bin/python experiments/issue13/replay.py \
  --local-results-root <private-results-directory>
```

hydrate-only replayにはhistorical RSS100行がないためshadowを実行しない。公開manifestにはGitHub Releases API行もあり、教師向けRSS cacheだけでは全100行を復元できない。元`incoming.jsonl`を正当に利用できる場合は`--input <private-frozen-incoming.jsonl>`を追加できる。違う入力hashは拒否する。

新しいcurrent RSSで別実験する場合は既存 `gaiden_sieve.ai_gaiden_shadow` のread-only取得を使い、日時/commit/input hashを別記録する。過去再現CLIに違うcurrent inputを渡して成功扱いにはしない。

### このPRの検証

- baseline-onlyを公開ファイルだけで実行し、holdout/gold/threshold metricsが記録と一致。
- 私有保存RSSから4manifestをhydrateし、JSONL hash全て一致。
- hydrateした教師で3candidateを再fitし、holdout/gold/LLM監査metricsが記録と一致。
- 私有元入力を使った再生で3candidateの100行/62独立URLの分類分布・OOVが記録と一致。
- 毎回保存candidateを再ロードして既存verify全項目pass。同じdata/seedでrefitして最大確率差0。
- コアの初回26テスト成功。公開用harness追加後の回帰チェックと負のhash検証は [validation.json](validation.json) に記録。
- 公開用harnessの追加は診断の再現処理であり、3つの新しい学習仮説を試したものではない。モデル方式と条件は同じ。

初回の実作業は2026-10-09約05:32–05:42 JST（約10分）、説明補足約05:43–05:45。第2回は08:08:20–08:17:47 JST（約9.5分）。取得・インストール待ちを含む。GitHub引き継ぎ整備は別作業としてPRに記録する。
