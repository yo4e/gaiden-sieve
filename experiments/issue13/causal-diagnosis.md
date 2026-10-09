# Issue 13 — 追加の局所診断と採用判断

**現行候補も今回の比較候補も採用しない。** 教師59件は確率の順位では完全分離している。以前の「教師上でも区別を学べていない」という説明は訂正する。問題は固定境界での判定と未知記事への一般化であり、uncertainを減らすだけでは改善を示せない。

## 固定条件と由来

元の59件（R38/N21）のtraining hashは `sha256:291d65aa8fac6aaf53812e4ac584b448fe8d2c258a9ef646c2f5b2acc76e2b18`。教師、監査、frozen RSSのmanifest hashと保護対象hashを照合した。原データ・ラベル・gold・運用設定を変更していない。

教師だけで一度fitしたTF-IDF unigram/bigramの語彙・IDF（2,426特徴）、seed42、運用境界R≥.8/N<.4を共通にした。新しいスコアを見る前に私有planへ3条件を記録した: C1、C10、同一analyzer-familyを逆数重み付けしたC1。重みは各クラスの総量R38/N21を保持する。goldでのparameter探索や結果を見た後のラベル修正はない。C10は局所診断であり運用設定への適用ではない。

教師には同一analyzer出力の7件familyが一つあり、残り52件は単独。この重み付けは厳密な同一familyだけを対象にし、全ての近似テンプレートを除去するものではない。

追加の評価用Microsoft Cloud4件（R2/N2）は、公式RSSを読んで助手がスコア計算前に暫定ラベルを固定した。教師、gold-review、holdout、既知LLM監査、frozen RSSとID/URL/本文identityで非重複。本文なし由来は [lineage/prospective-ms4.json](lineage/prospective-ms4.json)、本文と取得XMLは私有保存のみ。単一配信元・4件・助手ラベル・過去ページの小さな評価であり、人間の精度や新しい時間holdoutではない。取得parserは既存AI外電checkout `977c9a6b4be35e57f8d945eae7a84c65e5afa03f` を使用した。

## 比較結果

| 局所条件 | 教師AUC | 教師二値precision/recall | frozen RSS R/U/N | gold二値precision/recall | gold運用precision/recall | gold AUC |
|---|---:|---|---|---|---|---:|
| 原C1 |1.000|.8085 / 1|0 / 80 / 20|.6667 / 1|未定義 / 0|.7917|
| C10 |1.000|1 / 1|64 / 16 / 20|.6667 / 1|.3333 / .25|.7500|
| family重みC1 |1.000|.8085 / 1|0 / 80 / 20|.6667 / 1|未定義 / 0|.7500|

全条件のfrozen RSS OOVは.790934、未知62独立URLのOOVは.811661。語彙を固定しているためCや重みだけではOOVは変わらない。未知62件は原C1・重みC1でR0/U62/N0、C10でR53/U9/N0。原C1と重みC1はSourcegraph20件だけNで他sourceは全U。C10でもNはSourcegraphだけ。[causal-results.json](causal-results.json) にsource別分布・全行確率・logit寄与を保存した。RSSは正解ラベルがなく、この変化から精度やsource偏り低減を断定できない。この追加診断は既存100/62スナップショットを再利用し、新規current100取得とは扱っていない。

既知LLM holdoutの二値precision/recallは全条件1/1。C10の運用成績は1/.8333だが、人間goldでは確信R3件のうち正解1件、precision1/3。弱いholdoutの好成績だけでは判断できない。既知LLM監査8件の二値成績は全条件.75/1、C10運用成績は1/.6667。

追加Microsoft4件は全条件AUC1、二値成績.5/1。原C1・重みC1は全U、C10はR1/U3/N0（precision1、recall.5）。4件の暫定ラベルをgateや候補選定に使わない。

## logit分解で確認できたこと

原C1の教師正例は.750557–.796775、負例は.212245–.619505で、全59件の順位は分離する。切片1.028590は特徴が全てゼロの入力にP=.736642を与える。教師正例の平均logit寄与は正の特徴+.301377、負の特徴-.049906、負例では+.011443/−1.533391。切片と特徴寄与の和がdecision_functionと一致することを検証した。

C10では係数L2 normが3.398772→10.713564、切片が1.028590→1.652064（ゼロ特徴P=.839170）となる。教師正例は.901617–.935620、負例は.039996–.281192となり既存境界で教師を判定できる。しかし既知goldの誤りが残って確信度だけが増す。正則化への感度は示せたが、既定Cが唯一の原因だとも、高OOVが誤りを引き起こしたとも証明していない。ゼロ特徴Pは fitted modelの診断であり、OOVの多い全記事がその確率になる意味ではない。

family重みはクラス総量を保ったまま切片を.898035（ゼロ特徴P=.710545）へ下げたが、運用分布は変わらずgold順位も改善しなかった。厳密な同一family反復だけを抑える処置では解決しない。近似テンプレート、配信元、時間による一般化は未検証。

## 次の担当と判断の分担

**助手・次の実験担当で進められる設計:** 複数のAI系sourceごとに正例/負例を揃え、title+summaryだけで内容を判定できる候補を集める。情報不足と意味的境界は別キューへ隔離する。ID/正規化URL/本文hash/analyzer-familyに加え近似テンプレートと記事時刻でtrain/evaluationを分割する。既知goldは監査専用のまま固定し、新しい評価を独立の人間ラベルで確定する手順・レビュー候補数を事前に提案する。その評価確定後に表現・学習条件を少数の事前宣言された比較にする。無方向な少量追加の反復や既知goldでの調整はしない。この文書は設計案であり、次の採用やPhase4C実行許可ではない。

**本人・編集担当の政策判断が必要な境界:** (1) AIブランドのツールについて、記事の内容が一般的な端末UI/ASCII描画の実装だけの場合にAI関連記事とするか。(2) ML基盤の一般管理・権限・費用運用で、学習や推論の具体的AI機構の説明がない場合の扱い。これらは単なる取得・重複判定では解決できない。短い定義と数件のspot checkを求める対象であり、暫定教師全件の承認を戻す前提にはしない。映画紹介やcaptionだけの情報不足例は引き続き学習から隔離し、既存goldラベルを変えない。

未検証: 新しい独立人間評価での精度、十分な標本でのcalibration、character/subword方式、全ての近似テンプレート効果、時間/sourceを跨ぐ一般化、本番挙動。運用model・thresholdの変更、Phase4C、本番接続、mergeは行わない。

## 再現と検証

Python3.12.13、scikit-learn1.9.1、numpy2.5.3、scipy1.18.1、joblib1.6.0。その他は既存requirementsを参照。

```bash
python experiments/issue13/causal_probe.py \
  --private-root <private-results-directory> \
  --validation <private-prospective-validation.jsonl> \
  --output .diagnosis/issue13-causal/results.json
python -m pytest -q
```

私有結果directoryには元iteration2教師/監査/保存joblibとexperiment/incoming.jsonlが必要。新規validationはmanifestのfile hash一致を要求する。公開metadataだけから消えたRSS本文を復元できるとは主張しない。

元保存joblibと原C1の教師59件確率は最大差2.22e-16。全診断を再実行して結果JSONが完全一致。保護対象hash不変。3新規テストでクラス総量保持、family単位均等化、矛盾ラベル拒否、切片を含むlogit分解・ゼロ特徴入力を検証し、既存45件を含む48テストが成功した。公開JSONにtitle/summary、生RSS、joblib、私有絶対pathがないことを検査した。CIは公開コードのテスト・既存MLOpsを検証し、私有追加データ実験の再実行を検証するものではない。

追加診断の実作業は2026-10-09 09:54:10–10:01:27 JST、約7分17秒（取得・テスト待ちを含む）。GitHub保存・最終CI確認は別途行う。
