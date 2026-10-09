# URL欠落の修正と6件編集監査の準備

2026-10-10の再開では、Issue15のレビュー保存commit `cac6fc857a88256a22efb6c4b74d464d8763c5c7` を確認し、既存内容を保持して進めた。範囲はURL欠落の再現・修正・回帰検証、元59件の実影響監査、提案非提示の原入力票の準備。新モデルや編集方針の決定は行わない。

## 何を疑い、どう直したか

OOF（各記事を学習していないモデルで採点する内部評価）の群分けは、同じ記事をtrain/testへ分けないためにURLも検査する。しかし `representation_probe.py` はLabeledItemから5項目を作り直してURLを落としていた。元のJSONLにはURLがあっても、群関数へ届かない。

別source・別ID・違うtitle/summaryで同じURLを持つ合成2行を、元の5項目射影では2群、修正後では1群として確認した。修正は `grouping_rows()` で元metadataをID対応させ、source_urlを群分けへ保持する。元metadataのURLがない場合は黙って続けず停止する。モデルの入力は従来どおりtitle+summaryで、URLを学習させない。型付きの共通data schemaや本番コードは変更していない。

新規回帰検証は実際のOOF用射影経路を通り、同URLの再配信保持、URL欠落時の停止、URLがモデル文章へ混ざらないことを確認する。歴史的レビューの `recompute.py` は旧射影の抜けを示す検算として保持し、元保存JSONと一致することも確認した。

## 元59件では何が起きていたか

教師hashを再計算し、記録 `sha256:291d65aa8fac6aaf53812e4ac584b448fe8d2c258a9ef646c2f5b2acc76e2b18` と一致した。原59件を実際に読んで監査した結果:

| 項目 | 結果 |
|---|---:|
|教師件数 / URLあり|59 / 59|
|正規化URLの種類|59|
|同URLの2行組|0|
|旧/修正群数|49 / 49|
|群の同一関係が変わる2行組|0|
|3foldのURL交差|全fold 0|
|旧/修正/公開保存foldの割当|全fold一致|

したがって、**この元59件ではURL欠落に起因する実URL漏洩は確認されなかった**。潜在不具合の修正と、過去の測定値を無効にすることは別。新しくモデルを学習せず、既存fold割当との照合だけを行った。URLが異なる近似templateや同一イベントの意味的重複、source/timeの独立性まで証明したものではない。

[監査CLI](audit_training_urls.py)、[実監査結果](url-audit-results.json)で追検証できる。再現には元59件のhash一致の私有JSONLが必要。公開metadataだけから追加教師本文を復元したとは扱わない。

## 6件・3境界の準備

レビューの指定したGitHub UI/端末実装とcoding agent設定、AWS HyperPod管理とMoE強化学習、Microsoft物理冷却とAI企業活用の6件を、元入力hashに合わせて準備した。既存MS01–MS04の承認済み4件は除外。全文を取り直したり、新要約で置き換えたりしない。入力票には原title+summaryだけを置き、順序をshuffleして中立case IDを付けた。提案・理由・スコア・sourceフィールド・元役割・ペア対応を表示しない。文章中の企業名は残るのでsource-blindではない。

私有保存先（元作業環境のrepo rootから）:

- `.diagnosis/issue15-followup/editorial/private-input-sheet.md`: 本人が最初に読む6件の票。
- `private-input-sheet.json`: 同じ原入力の機械可読版。
- `private-response-template.json`: 初回判断、根拠/必要な背景、evidence_status、方針確認後の判断、日時を別に記録する空欄。
- `private-after-response-key.json`: 初回回答後に技術者が開くID/由来対応。ラベル・予測は鍵にも載せない。

原入力の判断後に [3境界の質問](editorial-policy-questions.md)へ進む。本人には最大6件と3問を残すが、助手は採否や方針を決めていない。人間レビューはまだ0件。既知teacher/隔離/probeを含む監査であり、今ラベルを隠しても過去の露出を消せず、未見の独立精度へは置き換えられない。

[本文なし準備manifest](editorial-preparation-manifest.json)と[準備CLI](prepare_editorial_audit.py)を公開保存した。原入力票・対応鍵は公開しない。原私有cacheがない場合は同じ6件の票を完全再現できたと扱わない。

## 検証・未実施・次の担当

65テスト成功。新規4検証はURLの射影保持3件と、提案/予測/ペアを除き原文章を保持する票1件。旧レビュー検算は保存2JSONと一致。URL監査と票準備は再実行JSON一致。保護された元教師・gold・profile等のhash不変。

```bash
python experiments/issue15/audit_training_urls.py \
  --training-data <private-original59-jsonl-directory> \
  --output .diagnosis/issue15-followup/url-audit.json
python experiments/issue15/prepare_editorial_audit.py \
  --output-root .diagnosis/issue15-followup/editorial
python experiments/issue15/recompute.py
python -m pytest -q
```

未実施: 本人の原入力採否・3境界の方針確定、新たなモデル学習/予測、全ての意味的重複監査、教師・gold・threshold変更、Phase4C、本番接続、merge、Issue close。

次は親担当が私有原入力票を本人へ適切に提示する。本人は初回6件の採否/不足/未決と根拠を記録し、続いて3境界を決める。技術担当は初回判断と方針後の変化を分けて整理する。情報不足や未決をNへ押し込まない。今回の承認範囲は準備までなので、未回答の欄は空白のまま保存する。

## 実作業時刻

開始: 2026-10-10 02:35:26 JST（2026-10-09 17:35:26 UTC）。終了: 2026-10-10 02:45:14 JST（2026-10-09 17:45:14 UTC）、実作業9分48秒。予定枠内の背景作業で、画面・音の操作は行っていない。

## remote保存の状態

pushは自動承認レビューにより拒否された。理由は、この会話で確認できる依頼がpush/公開を範囲外としており、外部送信の承認とdestinationの信頼性を確認できないこと。remoteへの保存は未実施。今回commitのremote CIも未確認で、ローカル65テストと区別する。公開を進める場合は親担当が明示的な範囲変更の承認を確認する。
