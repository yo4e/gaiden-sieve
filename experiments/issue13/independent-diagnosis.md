# Issue #13 独立診断: 学習内の順位、判定境界、汎化を分ける

2026-10-09 / ChatGPT側の月野テンプレクスによる引き継ぎ診断。

対象は PR #14 の元コミット `2bc2d93eca74eed446b77ce4fed09a438f08e4bc`。SIEVE core は `6e1da50285a753c5ffef4ca6d3bfe43fafcfd1f5`。元の [引き継ぎ](README.md)、[保存済み予測](prediction-comparison.json)、[結果](results.json) を変更せずに調べた。予測ファイルの Git blob SHA は `41a767ca059ea8b4a282392e1be250c311a26c3d`。

## 結論

**3回目の同種のデータ追加に進む前に、問題の言い方を修正する。**

- iteration2 の学習59件では、正例38件の点数が負例21件をすべて上回っている。「学習例すら順位で区別できない」は誤り。固定の二値判定境界では負例9件を誤判定し、運用境界では正例38件も全件保留になる。
- しかし既知の human gold では順位が混ざる。**学習内の点数の分離を、未学習記事への汎化や確率校正の成功と呼べない。** 判定線だけの変更でも、このgoldの目標には届かない。
- 現行holdoutの満点は非常に弱い証拠。本文を一切見ない「Sourcegraphだけ負例」という比較器でも満点になるうえ、学習とholdoutに特徴量化前処理後の完全同一例がある。

確率の読み違い、labelやsourceフィールドの直接混入、明示holdoutをfitする誤りは、読んだコード経路では見つからなかった。支配的な原因を正則化・偏った反復・ラベル品質のいずれかに断定できる段階ではない。

## 1. 新しく確認したこと

### 1.1 学習59件ではランキングが完全に分離している

`prediction-comparison.json` の `iteration2_training` 全59行を集計すると:

| 集合 | 件数 | P(relevant) 最小 | 最大 |
| --- | ---: | ---: | ---: |
| 正例 | 38 | 0.750557 | 0.796775 |
| Sourcegraph以外の負例 | 9 | 0.587340 | 0.619505 |
| Sourcegraphの負例 | 12 | 0.212245 | 0.267443 |

最も低い正例と最も高い負例の差は `0.13105219972895676`。全 `38×21=798` 正負ペアで正例が上にあり、**学習内ROC-AUCは1.0**。全正例と全負例を分ける単一の点数境界は、学習内に限れば存在する。

一方、二値判定は TP38 / FP9 / TN12 / FN0。運用の R/U/N は 0/47/12。したがって「非Sourcegraph負例9件を全部Rにする」という既存の観測は正しいが、そこから「特徴が正負をまったく分けていない」とは言えない。

これは保存済みの学習内スコアの診断であり、性能評価ではない。正則化が強すぎる可能性を否定するものでもない。切片、特徴寄与、確率の縮み、校正状態は別途確認が必要。

### 1.2 goldでは単なる判定境界の問題ではなくなる

Sourcegraphの簡単な負例4件を除くと、既知goldは正例4件・負例2件の6件となる。正例が負例を上回ったペア数を直接数えた。

| 保存済みモデル | gold全10件 ROC-AUC | 非Sourcegraph6件 ROC-AUC | recall≥0.80で到達可能な最大precision |
| --- | ---: | ---: | ---: |
| baseline | 0.875000 | 5/8 = 0.625 | 0.800000 |
| iteration1 | 0.875000 | 5/8 = 0.625 | 0.800000 |
| iteration2 | 0.791667 | 3/8 = 0.375 | 0.666667 |

正例が4件なので、recall≥0.80には4件すべてを採用する必要がある。iteration2ではそのための境界以下にすると負例2件も入り、precisionは4/6となる。**3モデルとも、この既知gold上でprecision≥0.95 / recall≥0.80を同時達成する単一境界は存在しない。**

この計算はしきい値を採用するための探索ではなく、「しきい値を下げれば解決」という説明の反証。profileやgateは変更していない。goldは既に何度も見た10件で、6件の部分集合も小さい。AUC低下を統計的に確かな一般性能低下とは呼ばない。将来のモデル選択には別のvalidationと未見testが必要。

### 1.3 holdoutには特徴レベルの同一例がある

bootstrapのSourcegraph負例のうち、summaryが空の7件と、holdoutの同種2件を確認した。版番号は違うが、現行TF-IDFの `build_analyzer()` 出力は9件すべて完全一致する。

既定のword token patternは1文字の数字を取り除く。そのため各桁が1文字の版番号の違いは消え、unigram/bigramの出現列が同じになる。同じ学習済みvectorizerへ渡せば、これらは同じベクトルとなる。これはURLやidの重複ではなく、**評価例に新規性がないという意味での特徴レベルの重複**。

確認対象は [bootstrap/03-mixed.jsonl](../../profiles/ai-gaiden/bootstrap/03-mixed.jsonl)、[04-not-relevant.jsonl](../../profiles/ai-gaiden/bootstrap/04-not-relevant.jsonl)、[holdout/01.jsonl](../../profiles/ai-gaiden/real-holdout/01.jsonl)。patch例の全ベクトルや全データの近似クラスタまで調べたという主張ではない。

coreのid重複拒否も、実験用support.pyのid・URL・文字列照合も、この前処理後の同一性までは検出しない。元データを勝手に消すのでなく、後続実験ではこの同一家族を分割境界の単位として扱う。

### 1.4 「sourceだけを見る」悪い比較器でも評価を通る

`SourcegraphならN、それ以外ならR` という反証用ルールは、現行holdoutの正例6件・負例4件を全件正解する。同じルールはgoldの TP4/FP2/TN4/FN0、iteration2の学習内 TP38/FP9/TN12/FN0 も再現する。

これは、モデルがsourceフィールドを読んでいる証拠ではない。現行 `build_text()` はtitle+summaryだけを使用している。**現行の二値評価だけでは、内容分類とsource由来の単純な規則を見分けられない**ことを示す負の対照であり、採用すべき分類器の提案ではない。

## 2. 壊れていることと、まだ証明していないこと

| 点検対象 | 今回の確認 | 限界 |
| --- | --- | --- |
| 特徴入力 | title+summaryのホワイトリスト。label/reason/sourceフィールドは直接含まない | 本文内のsource固有語彙や定型文は残る |
| 確率列 | `classes_` からpositive_labelの列を選ぶ | 元joblibの全挙動を今回再実行したわけではない |
| holdoutへのfit | 明示holdoutの経路ではbootstrapのみをfitする | 特徴レベルの重複は別問題 |
| binaryと運用指標 | `predict()` の二値評価と0.40/0.80の運用判定は別 | binary gate passは運用の確信Rの精度や件数を保証しない |
| 未定義precision | R予測0件はprecision未定義。追加後goldの確信R recallは0/4 | 保留もレビューに残せば正例4件は失っていない |
| OOV | analyzerのunigramとbigramの出現回数を合計した未知特徴率 | OOV約0.79は「記事の79%を理解できない」という意味ではない |
| 暫定ラベル | 追加例はLLMラベルのまま | 人間の正解に置き換えた扱いをしない |

したがって、現時点の診断は「学習内の点数順位は分離するが、固定境界での決定と既知goldへの汎化は良くない。さらに評価集合が単純規則でも通る構成」である。これを単純なデータ不足、単純なCの不足、単純なsource shortcutのどれか一つに縮めない。

## 3. 次に行う最小実験案（未実施・別判断）

### 先に測る: 固定モデルの点数内訳

原59件のhash一致入力と記録された環境が確保できる場合、既存の固定条件で再現し、切片 `b` と記事ごとの `w·x` を分けて記録する。fit結果を変えずに、正負教師と既知goldで次を見る。

- 既知語をまったく持たないゼロ特徴ベクトルの `sigmoid(b)`。これは教師の正例比率そのものとは限らない。
- source内の正負対比で、どの特徴群が正負の点数差を作っているか。本文や長い特徴引用は公開せず、集計・id・hashを中心に残す。
- unigram/bigram別のOOV、記事別の既知特徴数、定型文の実効反復数。

これは「学習中に何を覚えたか」と「新しい文でその手掛かりが消えるか」を切り分ける診断。消えたRSSを別記事や新summaryで埋めて、元実験を再現したと呼ばない。

### 比較はまず2条件だけ: C=1とC=10

実験権限を別途得たら、同じ59件、同じラベル・重み・seed・TF-IDF語彙/IDFを固定し、Logistic RegressionのCだけを1と10で比べる。運用しきい値0.40/0.80、gate、goldの役割、production状態は変えない。大量のgrid searchや、goldを見ての繰り返し選別はしない。

| 観測結果 | 読み方 |
| --- | --- |
| 学習内の余裕と確信度だけが増え、未学習の同source正負順位が改善しない | 正則化は点数の縮みに関与し得るが、汎化の解決ではない |
| 判定件数だけが増えて誤採用も増える | uncertain削減を改善と数えない |
| 事前固定したvalidationの同source正負順位・運用precision/recall・保留負担が改善 | 続ける候補。ただし未見testで確認するまで採用しない |
| Cを変えてもほぼ同じ、または大きく不安定 | 特徴と教師分布、ラベル整合性の検証へ優先度を移す |

定型文の反復を次に調べる場合は、source/template familyの重みを変える対照を独立に行う。単純にSourcegraph行を削るだけではクラス比率と総重みも変わるため、正負の総重みを保った比較を設計する。今回この対照は実行していない。

### 評価の立て直しは性能比較の前提

現行holdoutやgoldは履歴を守って保持する。別途、新しい検証集合を予測結果を見る前に固定する。単なるランダム分割でなく、同一URL・正規化文・template familyを跨がせず、時間での分割も考慮する。同じsource内に正例と負例を置くテストと、未知sourceへのテストは目的を分けて報告する。

source-onlyの負の対照、source別・内容クラス別の成績、確信Rのprecision/recall、保留率、レビューに残した正例のrecallを併記する。ラベルは由来を保存し、曖昧・情報不足例を無理に二値へ押し込まない。山田さんに追加教師全件の確認を戻すことは前提にしない。方針の曖昧さが実験の結論を左右する少数例だけ、必要時に編集判断へ戻す。

## 4. 再計算と確認範囲

追加した [audit_predictions.py](audit_predictions.py) はPython標準ライブラリだけで動く。モデルをロードも学習もせず、保存済み予測から指標を独立に再計算する。CLIは標準出力だけを使い、repoやモデルを変更しない。

```bash
python experiments/issue13/audit_predictions.py
pytest -q tests/test_issue13_audit.py
```

テストは運用境界、NaN/不正値、同点、未定義precision、保存済み判定との矛盾、idの重複を確認する。さらにGit管理された元予測ファイルを直接読み、上の59件/10件の結論を検算する回帰テストと、実際のcore analyzerで7+2件の特徴同一性およびholdoutの負の対照を検証するテストを含む。

ローカルでは単体テスト17件、保存済み69件のスコアを手動転記した必要列の検算、9件の空summaryタイトルのanalyzer一致、独立なscikit-learn実装とのAUC照合100ケースを確認した。手動転記したprojectionは元ファイルの完全な複製ではなく、公開データセットとして追加していない。ローカル環境はPython3.13.5 / scikit-learn1.8.0で、元実験の環境と異なる。元ファイルを直接使う2テストを含め、追加19件を載せたコミット `6c882f68f7097422e198133111bf438b70893886` の [MLOps CI #67](https://github.com/yo4e/gaiden-sieve/actions/runs/37863974747) は成功。Test、既存baselineのTrain candidate、Evaluate/reproducibility、reportの全工程成功を確認した。これは元環境での59件の再学習成功を意味しない。

原59件の本文を使った学習、原100件の完全shadow再生、C比較、校正の検証、新規評価データ取得は未実施。既存CIが走らせるbaselineのtrain/evaluateと、今回の新たな学習実験は区別する。

## 5. 変更しなかったもの

core、profile、gold、gold-review、holdout、bootstrap、元の予測・結果、threshold、promotion policy、AI外電の本番連携は変更していない。新規の第三者記事本文、生RSS、Slackログ、私有cache、joblib、秘密情報は追加していない。Issue #13は未完了のまま、PR #14はDraftのままにする。次の実験案は実施済みとして扱わない。

## 一次資料

リポジトリ内: [train.py](../../src/gaiden_sieve/train.py)、[data.py](../../src/gaiden_sieve/data.py)、[classify.py](../../src/gaiden_sieve/classify.py)、[evaluate.py](../../src/gaiden_sieve/evaluate.py)、[drift.py](../../src/gaiden_sieve/drift.py)、[support.py](support.py)。

ライブラリ仕様: [scikit-learn text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)、[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。ライブラリのstable文書だけを元実験の完全再現証明には使わない。
