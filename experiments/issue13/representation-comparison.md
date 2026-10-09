# 単語と文字部分表現: 固定2条件の一巡

**文字表現C1を採用する根拠は得られなかった。** 文字断片の未知率は低くなるが、新しい6件の二値・運用判定は変わらず、暫定ラベルに対する順位は悪化した。条件を追加探索せず、この一巡を完了として記録する。

## 仮説と採点前の固定

単語が少し変化するとTF-IDF word表現では別の未知語になる。単語内の文字部分を拾えば、既存の語と共通部分を使って未知語へ対応し、内容判定を助けられるか疑った。

[representation-plan.json](representation-plan.json) を採点前に固定した。比較は元の59件（R38/N21）・同じラベル・LR C1/max_iter1000/seed42・一様重み・R≥.8/N<.4のまま、以下の2つだけ。

| 条件 | 表現 | 全59件fitの語彙数 |
|---|---|---:|
|word12_c1|単語1語と連続2語|2,426|
|charwb35_c1|単語内の3～5文字、語の端を空白で表すchar_wb|9,476|

hybrid、他の文字長、C値、thresholdの探索はしない。既知goldや承認済みMS4に合わせて設定を選び直さない。文字表現の最適設定を見つけたとの主張もない。source/URL/理由/ラベルは特徴へ混ぜない。運用profileと原データは変更していない。

## 評価の役割を分ける

**内部の群分割評価:** 元教師59件をidentity、版番号共通化、近似templateで49群（最大7件）にし、StratifiedGroupKFoldで3分割、seed42を固定。各foldのテストは20/20/19件、Nは各7件。群がtrain/testを跨ぐ組は0。各foldのTF-IDFはそのfoldのtrainだけでfitする。OOF（各記事を、その記事を学習しなかったモデルで採点した結果）は、練習ラベルに対する内部診断で独立人間精度ではない。群検出が全ての意味的類似を見つけた保証もなく、同sourceの異なる群は両側に残る。

**新しい暫定6件:** 採点前にRSS-onlyの助手ラベルを固定。Microsoft R2/N2、GitHub R1/N1。同一sourceの対比を3組用意し、既存教師/holdout/gold/監査/承認MS4/選択済み候補/隔離例/frozen RSSの保護群と一致0。MicrosoftはAI企業活用とデータセンター冷却、AI採用とセキュリティ啓発、GitHubはcoding agent設定と一般ハッカソン案内。本文なしの由来は [representation-probe.json](lineage/representation-probe.json)。人間確認済みgoldとして扱わない。

取得は公式RSS6ページ・60候補、HTML取得0。GitHubで2組目の明確な独立Nを確保できず、予定上限R2/N2を無理に埋めず1組にした。Microsoftの一組は既存の未採点poolから補った。各組は約1～2日差だが記事自体は2024–2025年で、future holdoutではない。データセンター冷却等のNは「与えられたRSSではAI利用の説明がない」という暫定編集解釈で、原記事にAIがないという事実の保証ではない。採点後に不都合なラベルを隔離し直していない。

**既知集合:** 人間gold10、LLM holdout10、既知LLM監査8、方針承認MS4は既知の監査。MS4承認は助手要約への方針承認で、blind独立採点ではない。全100行と未知62独立URLは過去の凍結RSS観測で正解ラベルなし。今回新しくcurrent100を取ったとは扱わない。

## 何が変わったか

| 指標 | 単語 | 文字部分 |
|---|---:|---:|
|内部OOF binary precision / recall|.8085 / 1|.8085 / 1|
|内部OOF pooled AUC|.7155|.7293|
|OOF非Sourcegraph N9のbinary除外recall|0/9|0/9|
|新暫定6 binary precision / recall|.5 / 1|.5 / 1|
|新暫定6 運用R/U/N|0/6/0|0/6/0|
|新暫定6 pooled AUC|1.0000|.7778|
|新暫定Microsoft4のAUC|1.0000|.5000|
|新暫定GitHub2のAUC|1.0000|1.0000|
|既知human gold binary precision / recall|.6667 / 1|.6667 / 1|
|既知human gold AUC|.7917|.7083|
|既知LLM監査8 AUC|.9167|.5833|
|方針承認MS4 運用R/U/N / AUC|0/4/0 / 1|0/4/0 / 1|
|frozen RSS100 R/U/N|0/80/20|3/77/20|
|未知62URL R/U/N|0/62/0|0/62/0|

文字で増えたRSS採用候補3件は、全て既存教師と重なる記事。未知62件はどちらも全Uのままなので、100件のUが3件減ったことを未知記事の改善と扱えない。NはどちらもSourcegraph20件だけ。

OOFのSourcegraph N12は両条件ともbinaryで12/12除外でき、他sourceのN9は0/9。source型の偏りは残る。fold別AUCは単語1/.6044/1、文字1/.6264/1。pooled AUCは異なるfoldモデルのスコアをまとめた値で、モデル間のスコア尺度差も含む。.7155→.7293だけを改善の有意な証拠とはしない。

OOVは単語のRSSで.790934、文字で.515212、新暫定6で.563107/.216082。ただし前者は「単語/2語の出現」、後者は「3～5文字断片の出現」で、分母も単位も異なる。文字の低い値を、同じ単位の未知語改善率や内容精度として引き算しない。文字の共通部分を使えることはsyntheticテストで確認したが、意味を理解した証拠ではない。

## 具体的な一組を見る

新暫定Microsoftの冷却記事Nは単語.723952/文字.730752、AI企業活用Rは単語.732793/文字.718870。文字側では、読める断片が増えてもこの組の順序が逆転する。同じsourceの文体や定型語も文字で共有されるので、単に表現を細かくするだけでは目的の内容の違いを学べない場合がある。この説明は一例の観測であり、どの文字断片が逆転を因果的に作ったかの検証はしていない。

新6件のbinaryは全R、precision.5/recall1/F1.6667、運用は全Uで確信R recall0・precision未定義。元goldもこの2条件で二値成績は変わらない。固定C1の文字方式を採用したり、結果を見てCやthresholdを追加調整したりする理由にはならない。

## 一巡の結論と次の判断

今回、**語の綴りの一部を再利用できること**と、**この条件で分類の目的を改善できること**を分けられた。実用的な改善は確認できず、運用方式は元のまま維持する。文字方式一般が無効だとは結論しない。

次に実験を広げるなら、暫定ラベルの意味的整合性と独立人間評価を先に整え、複数source内のR/Nと時間を確保する判断が必要。新6件への同意を即要求したり、既知4件を再質問したりはしない。未検証は新6件の独立人間精度、未来記事、全source、十分な標本でのcalibration、別表現/Cの最適化、近似template検出の取りこぼし。外部モデルや外部AI送信は今回必要なく行っていない。

## 再現・検証・保存境界

```bash
python experiments/issue13/representation_probe.py \
  --private-root <original-private-results> \
  --probe <private-new-probe.jsonl> \
  --output-root .diagnosis/issue13-representation/replay
python -m pytest -q
```

[数値・source別分布・記事別確率・fold](representation-results.json)、[検証記録](representation-validation.json)、[公式RSS取得観測](lineage/representation-feed-observations.json)を保存。計画とprobeのhashを照合し、教師・RSS・監査の既存manifestも照合。単語モデルと元保存joblibは教師59件上最大差2.22e-16。同じ条件で再実行して数値・分割が完全一致し、係数/切片/IDF差も0。word joblibのバイトhashは再保存間で異なり、charは一致したため、バイト同一と数値再現を区別して検証JSONに記録する。元保存artifactのhash検証とは別のチェックである。61テスト成功。新規4テストは文字部分の再利用、分類器条件共通、版違い群のfold分離と全行一度ずつ採点、評価語彙をfitへ混ぜないことを確認する。

各モデルのjoblibは `.diagnosis/issue13-representation/` にのみ保存。公開はコード・テスト・hash・URL・数値・説明だけで、新規第三者title/summary、生RSS、model、秘密情報、端末絶対pathを含めない。新6件を元教師やgoldへ加えず、promotion、Phase4C、本番接続、mergeはない。CIは公開コードと既存MLOpsの検証で、助手ラベルを人間goldへ変えるものではない。

今回の比較・再現検証・文書更新は2026-10-09 11:06:52–11:18:55 JST（約12分3秒、RSS取得待ちを含む）。同じ条件の再実行は再現確認であり、追加のparameter探索ではない。GitHub保存と最終CI確認は続く作業としてPRコメントに記録する。
