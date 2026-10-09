# 独立評価の小さな実作成と次の一歩

**学習データもモデルも増やさず、まず2組・4記事を人間の編集判断で確かめる。** 新しい評価候補8件を実作成した。4件は最小レビュー、2件は対比負例がまだない拡張候補、2件は政策境界だけの参考例。全て未確定で、学習・予測・精度計算をしていない。

## 最小レビュー: 同じ配信元の2組

次表は助手の内容説明であり、第三者の原見出し・概要の転載ではない。入力原文のtitle+summaryは私有 `private-human-review.md` に固定し、助手の提案ラベルやモデル予測を載せていない。本文には企業名があるため完全な配信元blindではない。公式リンクの本文を読んでRSS入力にない情報を足す採点はせず、情報不足なら保留する。

| 組 | 候補ID | RSSから分かる内容 | 記事日時UTC | 助手の仮提案 |
|---|---|---|---|---|
| A | MS01 | [量子計算チップの発表](https://news.microsoft.com/source/features/innovation/microsofts-majorana-1-chip-carves-new-path-for-quantum-computing)。AI用途の説明はない |2025-02-19 16:00|除外候補|
| A | MS02 | [生成AI・Copilotの企業活用事例](https://www.microsoft.com/en-us/microsoft-cloud/blog/2025/02/19/revolutionizing-work-customer-success-stories-with-gen-ai) |2025-02-19 15:30|採用候補|
| B | MS03 | [セキュリティ会議の参加案内](https://www.microsoft.com/en-us/security/blog/2025/02/03/hear-from-microsoft-security-experts-at-these-top-cybersecurity-events-in-2025)。AI用途の説明はない |2025-02-03 17:00|除外候補|
| B | MS04 | [言語モデルに資料を検索して渡すRAG手法の解説](https://www.microsoft.com/en-us/microsoft-cloud/blog/2025/02/04/common-retrieval-augmented-generation-rag-techniques-explained) |2025-02-04 16:00|採用候補|

**本人への最小確認は2つの組だけ:** 「AはMS02を採用・MS01を除外でよいですか」「BはMS04を採用・MS03を除外でよいですか」。違う場合は該当IDに採用/除外/情報不足を示せばよい。全教師の採点は求めない。確認は新しい正解を4件だけ作るためであり、助手の答えを人間の答えと混同しないために必要になる。提案通りなら同一source内R2/N2になるが、人間判断が変わったら実ラベルの分布を数え直し、足りない対比を別候補で補う。ラベルを均等化するために人間の答えを変えない。

日時差はAが30分、Bが23時間。同じsource・近い時期にも正負があるかを調べ、source名や古さだけで正解できない最初の小さな教材にする。n=4かつMicrosoftだけなので精度保証・全sourceへの一般化・採用判定はできない。

## sourceの拡張と境界例

GitHubはモデル学習によるコード補完改善（GH01）、AWSは言語モデル提供開始（AWS01）の明確なR候補を保存したが、独立で明確なNの対比が未確保。GitHub公式AI&ML RSSの過去4ページ40件も調べたが、今回の範囲で政策境界を使わず対比を完成できていない。これら2件を主評価に混ぜて「3sourceでbalanced」とは呼ばない。目標は各source R2/N2、合計12件だが、これは未達の設計目標。無理にNと決めず情報不足を隔離する。

次の2種類は**今回の4件評価から除外できるため、再開の必須質問ではない**。将来境界評価を追加する際だけ、次の短い政策質問を使う。

- [Copilot CLIの端末ASCII描画](https://github.blog/engineering/from-pixels-to-characters-the-engineering-behind-github-copilot-clis-animated-ascii-banner): RSSの焦点は描画、端末互換性、アクセシビリティで、AIモデルの改善ではない。「AIツール自体の一般UI実装も、AI外電の対象に含めますか？」
- [HyperPodの管理・統治](https://aws.amazon.com/blogs/machine-learning/best-practices-for-amazon-sagemaker-hyperpod-administration-and-governance): RSSの焦点は権限、共有容量、管理境界。「AI/ML専用基盤の運用管理も、具体的な学習・推論の説明がなくても含めますか？」これは以前隔離した既知例で、新しい独立精度に使わない。

既存方針の確認: [Issue13](https://github.com/yo4e/gaiden-sieve/issues/13) はAI系sourceにもNがあるとして、配信元だけの採否を否定する。[確定gold-reviewのGitHub UI例](../../profiles/ai-gaiden/gold-review.jsonl) は人間Rで、AI機構の詳細説明が薄くても一律Nではない。Googleファッション例とAWS分析例は人間N。これら既存判断は再質問も再ラベルもせず固定監査に残す。一方、AI関連UIの記事が全てRか、一般基盤管理がRかという普遍規則はその個別判断から確定できない。AI外電の掲載policyも公式RSS利用を規定しているが、この2種類の内容境界を明文化していない。

## 分離の具体的な手順

1. **保護集合を固定:** 元教師59、既存holdout10、人間gold10、既知LLM監査8、既に採点したMicrosoft4、隔離例5を別役割で保持。goldは学習にも候補選定・parameter調整にも混ぜない。今回のcore4件はこれらとのidentity/近似群重複0。
2. **記事単位の群を作る:** id、trackingを外したURL、正規化title、title+summary一致を検査。sourceが違っても同じ記事は一群にする。RSS定型末尾を外し、数字・版番号を共通化したtoken列と単語3個のまとまりで近似テンプレートを検出する。Jaccard（共通するまとまり÷全まとまり）≥.60、各側5まとまり以上を同じ群として推移的に結合。これは事前固定の保守的heuristicで、意味的独立の証明ではない。一般topicが同じ記事を近似群にしない場合もあるため、レビュー時に同じイベント・同じチュートリアル系列なら群を統合し、採点前に再固定する。
3. **群ごとに役割を固定:** 保護群と重なる新候補は主評価から除外する。coreは人間確認後も評価専用で教師に移さない。将来のtrain/dev/evaluationでは群を跨いで置かず、同じsourceのR/Nを各評価cohort内に確保する。source留置評価を作るならそのsourceの全群を学習側から外して新規fitする別実験とし、既存59モデルでsource留置をしたとは主張しない。
4. **時間を別軸で扱う:** 今回coreは過去記事の内容対比であり、future holdoutではない。将来の時間評価の設計はcutoff `2026-10-09T00:00:00Z`、7日間embargo（境目の近い例を除く期間）、評価開始 `2026-10-16T00:00:00Z`。publication時刻を使い、取得時刻で新記事扱いしない。同一群がcutoff/embargoを跨ぐなら全群purge（除外）。その未来記事はまだ準備できず、今回はスケジュールや自動監視を作らない。
5. **確認後の小さな再評価:** human_label、確認日時、理由、入力hashを別評価ファイルに保存し、同じ凍結モデルのままcore4を初回採点する。助手提案は履歴として別に残す。正誤件数、binary（2択）成績、固定境界R/U/N、正例をU含めレビューへ残せた数、OOVを記録。sourceだけで全Rにする比較なら人間R2/N2のときprecision.5/recall1になるため、この単純比較と同一source内順位を見る。n=4の率だけでgateや採用に進まない。

## 実作成物・検証

公開: [候補manifest](lineage/independent-review-candidates.json)、[公式RSS取得観測](lineage/evaluation-feed-observations.json)、[準備CLI](prepare_evaluation.py)、[群と時間の検査](evaluation_design.py)。本文なしmanifestはURL、日時、feed hash、文章hash、暫定提案、人間未確認状態、群IDと保護群一致を記録する。RSS取得6ページ成功、HTML取得0。全core/拡張候補で保護群一致0、政策基盤例だけ既存隔離例と一致する。

私有: `.diagnosis/issue13-independent-eval/review/private-review-candidates.jsonl`（8件）、`private-human-review.md`（原入力を読むための未採点worksheet）、取得XML/cache。公開repoから本文を逆算できるとは主張せず、再現にはhash一致の私有原入力が必要。

```bash
python experiments/issue13/prepare_evaluation.py \
  --private-root <original-private-results> \
  --new-pool-root <private-new-rss-pools> \
  --output .diagnosis/issue13-independent-eval/review
python -m pytest -q
```

2回の作成結果はJSON完全一致。54テスト成功。新規6テストはsource跨ぎ同URL、版番号だけ違うtemplate、近似/非類似、時間embargo境界、時刻を跨ぐ群のpurge、変更された私有pool hashの拒否を検証。protected hash不変。新規学習・採点・採用・gold編集・本番接続はない。
