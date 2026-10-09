# 承認済み2組4件の初回採点

**今回の小評価は完了。全4件がuncertain、二値precision=.5/recall=1、順位AUC=1。採用やgateへは進まない。** 追加学習せず、保存済みの既定C1・教師59件モデルだけを使った。

## 答えの由来

2026-10-09 10:45 JST、プロジェクト本人が「生成AIの企業活用とRAGは拾う、AI用途の説明がない量子チップとセキュリティ会議案内は拾わない」という助手の要約・採否提案を承認した。私的な発言原文やメッセージ識別子は転載しない。

参照ラベルはMS01=N、MS02=R、MS03=N、MS04=R。これは **助手提案に対する本人の編集方針承認** であり、原RSS入力を本人が独立精読し、提案を伏せて答えたblind評価ではない。本人承認はあるが、`gold.jsonl`へ追加も置換もしない。`review_method`と`reference_origin`を [結果JSON](approved-small-evaluation.json) に明示した。候補作成時の未承認manifestは履歴として保持する。

## 固定したもの

- model version `20261008T231148972272Z-291d65aa`、C1、seed42。
- training hash `sha256:291d65aa8fac6aaf53812e4ac584b448fe8d2c258a9ef646c2f5b2acc76e2b18`。
- joblib SHA256 `44610aa8ab69d035fcd4da8ebff7ece0dbe5708409691a3fc9f9cab28ecae701` を保存metadataと照合。
- 入力は既に固定したcore4件のtitle+summary。全8候補JSONLのhashと各文章hashを照合し、拡張2件・政策例2件は採点から除外。
- R≥.80、N<.40、間はU。運用profile・原教師・元gold・holdoutのhash不変。

## 初回の結果

| ID | 内容の要点 | 方針承認の参照 | P(R) | 二値判定 | 運用判定 |
|---|---|---|---:|---|---|
| MS01 |量子チップ|N|.701709|R|U|
| MS02 |生成AIの企業活用|R|.750142|R|U|
| MS03 |セキュリティ会議案内|N|.696491|R|U|
| MS04 |RAG手法|R|.744684|R|U|

二値はTP2/FP2/TN0/FN0、precision2/4=.5、recall2/2=1、F1=.666667。運用はR0/U4/N0、確信R precisionは未定義、確信R recall0/2=0。Uをレビューへ残すなら採用対象2/2を保持（retained-for-review recall1）。OOVは.590038。sourceはMicrosoft Cloud一つなので、source別分布もU4のみ。

採用側のスコアは除外側より全て高く、順位AUC=1。組Aの差は.048432、組Bは.048193。ただし固定境界で内容どおりの採否はできず、二値の成績は「このsourceなら全部R」とする単純比較のprecision.5/recall1/F1.666667と同じ。sourceだけの定数スコアなら順位AUC=.5なので、今回のスコアには内容等に応じた差があるが、4件の結果からsource偏りの除去を証明できない。

**採点後に.70付近へ境界を移して成功扱いすることはしない。** 正負の間に境界を置けそうな小集合でも、既に見た4件へ合わせるだけになり、未知の記事での成功は示せない。今回は順序と境界の違いを確かめる教材として記録する。

## 何が分かり、何が残ったか

この同source・近い時刻の対比では、本人が拾いたいと承認した2件が上に来た。一方、二値でも運用でも正負をそのまま分ける結果にはならない。uncertainのキューへ回す仕組みは対象2件を落としていないが、4件全て人の確認を要するため、選別負担を減らせた証拠にはならない。

未検証なのは、独立blind人間精度、他source、未来の記事、十分な件数でのcalibration（スコアと正解頻度の対応）、採用可能な性能。今回の承認は原入力を本人が独立採点した事実へ置き換えられない。次の検証候補は [評価設計](independent-evaluation-plan.md) に残すが、新規大規模実験や採用を今回は実行しない。本人に同じ4件の判断を再度求める必要もない。

## 再現と検証

```bash
python experiments/issue13/score_approved_review.py \
  --private-root <original-private-results> \
  --review-input <private-review-candidates.jsonl> \
  --output .diagnosis/issue13-independent-eval/approved-small-evaluation.json
python -m pytest -q
```

保存joblibのロードと推論だけで、fitは呼ばない。結果JSONを再実行して完全一致、入力・モデル・保護対象のhash照合成功。57テスト成功。新規3テストはfitを呼ぶと失敗する凍結stubで二値/運用/順位の区別、方針承認がblindではない状態、改変入力の採点前拒否を検証する。CIは公開コードの検証であり、この参照ラベルを独立goldへ昇格させない。

新規公開物は本文なしの数値・来歴とコード。title/summary、生RSS、モデル、私的メッセージ原文は公開しない。学習者向けの説明は [learning-notes.md](learning-notes.md) の追記にある。

初回採点・再実行・検証・文書更新は2026-10-09 10:49:28–10:53:25 JST（約3分57秒）。GitHub保存と最終CI確認はその後の作業としてPRコメントに記録する。
