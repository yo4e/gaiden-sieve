# GAIDEN SIEVE

GAIDEN SIEVE は、RSS/Atom の記事を媒体ごとの編集方針に照らして分類する、小さな機械学習システムです。

このリポジトリの主目的は、分類精度だけを追うことではありません。教師データ、学習コード、モデル、評価、promotion、監視までを一つの運用として扱う **MLOps の最小ループ**を、実物を追いながら学べる形で作ります。設計の正本は [`DESIGN.md`](DESIGN.md) です。

## 現在の実装範囲: Phase 4B.1

Phase 3 では、Phase 2 の **追跡・評価・明示昇格できる model lifecycle** を土台に、CI、再現性チェック、JSONL batch classification、簡単な drift report、uncertain queue までをつなぎます。production promotion は引き続き人間が明示的に行います。

```text
labeled data + code
        ↓
      train
        ↓
candidate .joblib
        ↓
metadata / lineage
        ↓
reload + evaluate
        ↓
quality gate
   ├─ fail → reject
   └─ pass → explicit promote
                    ↓
               production
                    ↓
                 classify
```

主な実装は次の通りです。

- `gaiden_sieve.train`: TF-IDF + Logistic Regression、random holdout / 明示external holdout、candidate 作成
- `gaiden_sieve.artifacts`: joblib artifact、JSON metadata、data/model hash、candidate / production の保存・ロード
- `gaiden_sieve.evaluate`: Precision / Recall / F1 と profile ごとの quality gate
- `gaiden_sieve.promote`: gate を通った candidate だけを明示的に production へ昇格
- `gaiden_sieve.classify`: production model の `predict_proba()` を profile のしきい値へ通し、`relevant / uncertain / not_relevant` を返す
- `gaiden_sieve.batch`: unlabeled JSONL の batch classification と uncertain queue
- `gaiden_sieve.drift`: 平均予測確率、uncertain率、source別分布、OOV feature率の観測
- `gaiden_sieve.reproducibility`: data / code / config / dependency / metrics の追跡確認
- `gaiden_sieve.shadow`: gate を通った candidate を promotion せず実入力で比較観測
- `gaiden_sieve.ai_gaiden_shadow`: AI外電の既存取得・admissionを read-only で再利用する接続アダプタ
- `python -m gaiden_sieve`: train / evaluate / verify / promote / classify / drift / shadow CLI

`uncertain` は第三の学習クラスではありません。`not_relevant_threshold <= P(relevant) < relevant_threshold` の中間帯を、運用上 `uncertain` と呼びます。

### candidate と production

candidate は「新しく学習できたモデル」、production は「現在採用されているモデル」です。この2つを同一視しません。

```text
artifacts/
└── ai-gaiden/
    ├── candidate/
    │   ├── <model_version>.joblib
    │   └── <model_version>.metadata.json
    └── production/
        ├── model.joblib
        └── metadata.json
```

candidate の metadata には最低限、次を記録します。

- profile / model version
- training data SHA-256
- model artifact SHA-256
- Git commit SHA（取得できる場合）
- random seed / test size
- train / holdout size
- 主要 hyperparameter
- Precision / Recall / F1
- quality gate の基準と pass / fail
- Python / scikit-learn / joblib version
- created_at

ここで大事なのは「成績が何点だったか」だけではなく、**そのモデルがどのデータ・コード・設定から作られたかを後から辿れること**です。

### quality gate

`profiles/ai-gaiden/config.yml` の初期基準は次です。

```yaml
precision_gate: 0.95
recall_gate: 0.80
```

新しい candidate が作れても、この基準を満たさなければ production には昇格できません。promotion は自動ではなく、candidate の `model_version` を指定する明示操作です。

### Phase 3 の MLOps loop

```text
code + fixture/data
       ↓
 GitHub Actions
       ↓
test → train → evaluate → reproducibility report
                         ↓
production model → batch classify
                         ↓
                    drift report
                         ↓
                  uncertain queue
```

GitHub Actions は pull request / push ごとに小さい fixture で `pytest → train → evaluate → verify → report` を実行します。workflow の権限は `contents: read` のみで、`promote` は呼びません。つまり **CI が production model を自動更新する経路はありません**。

再現性チェックでは「同じ条件なら必ず全バイト同一」とだけ考えず、training data hash、Git commit SHA、random seed、test split、hyperparameter、dependency version、metrics が結び付いていて、保存済み candidate の評価経路を後から再実行できることを確認します。

drift report は異常判定器ではありません。平均 `P(relevant)`、uncertain率、source別の分類分布、TF-IDF vocabulary にない feature の割合を、**最近の入力が以前と変わってきたか気づくための観測値**として保存します。

`uncertain.jsonl` は active learning の入口です。各行に `id / source / title / summary / probability / classified_at / model_version` を残し、将来 human review や LLM review へ渡せるようにします。Phase 3 ではレビュー自動化や scheduled retraining は行いません。

### Phase 4A: AI外電 shadow mode

Phase 4A では、AI外電の本番更新フローへ SIEVE を差し込まない。代わりに、GAIDEN SIEVE 側の手動 GitHub Actions workflow `.github/workflows/shadow-ai-gaiden.yml` が AI外電を read-only checkout し、現在の公式RSS / GitHub Releases と admission 設定を使って独立した比較実験を行う。

```text
AI外電 current feeds + admission
          ↓
read-only shadow export
          ├─ existing admission: all / ai_terms_v1
          ↓
gate-passing candidate (no promotion)
          ↓
relevant / uncertain / not_relevant
          ↓
comparison + drift + uncertain queue
```

ここでは candidate を production に promote しない。shadow 実行時に現在の quality gate を再確認し、gate を満たした candidate だけを観察に使う。結果は Actions artifact `ai-gaiden-sieve-shadow` に保存し、AI外電の記事、`seen.json`、daily-news workflow、公開判断には一切反映しない。

主な出力は次の通り。

- `source-observation.json`: AI外電の取得成功 / 失敗と既存 admission 件数
- `incoming.jsonl`: admission 前の実記事 + 既存 admission 判定
- `classified.jsonl`: SIEVE の分類結果
- `comparison.json`: ai_terms_v1 / all と SIEVE の一致・不一致、source別分布
- `drift.json`: 平均予測確率、uncertain率、OOV傾向など
- `uncertain.jsonl`: 後で人間が確認する候補

`comparison.json` の disagreement は、それだけで「どちらかが間違い」という判定ではない。human gold は10件の確認済み境界例だけなので、gold に含まれない current RSS については引き続き **差分を見つけるための観測値**として扱う。

手動実行は GitHub Actions の **AI外電 SIEVE shadow experiment** から行う。scheduled retraining、LLM review、自動掲載にはまだ接続しない。


### Phase 4B.1: 実RSS bootstrap と独立holdout

Phase 4A の最初の shadow artifact では、20件の手作り fixture だけで学習した candidate が実RSS 100件をすべて `uncertain` にし、OOV feature rate は約 0.924 だった。Phase 4B.1 では TF-IDF + Logistic Regression としきい値をそのままにし、まず教師データを実入力へ寄せる。

現在の実RSSデータは次の4層に分ける。

- `profiles/ai-gaiden/bootstrap/`: 36件（relevant 24 / not_relevant 12）。`label_source=llm` の bootstrap 教師データ。
- `profiles/ai-gaiden/real-holdout/`: 10件（relevant 6 / not_relevant 4）。bootstrap と id が重ならない明示holdout。
- `profiles/ai-gaiden/gold-review.jsonl`: 10件のレビュー履歴。assistant の `proposed_label` と人間の `human_label` を併記し、`review_status=confirmed_human` で確認済みを追跡する。\n- `profiles/ai-gaiden/gold.jsonl`: 上記10件を人間が確定した gold set（relevant 4 / not_relevant 6、`label_source=human`）。学習には使わず、candidate の編集判断への整合を見る監査評価に使う。

元データは AI外電 shadow run `36220686078` の実RSS artifact に由来する。長大な release note 等はモデル入力を一部データだけが支配しないよう summary を先頭500文字まで保存し、切り詰めた行には元文字数と `summary_truncated_at_chars` を残す。

Phase 4B.1 の candidate は bootstrap 全件で fit し、別ファイルの real holdout で既存 quality gate を測る。metadata には training / evaluation dataset の hash と `evaluation_mode=external_holdout` を記録する。human gold は小さく意図的に境界例へ寄せた10件なので gate にはせず、CI / shadow で独立した監査 metrics として記録する。既存20件の `labeled.jsonl` は lifecycle テスト用 fixture として残す。

保存済み historical 100件を同じ TF-IDF + Logistic Regression で再生した観測は `reports/ai-gaiden/phase4b1-historical-replay.json` にある。baseline の `uncertain=100/100` に対し historical replay は `relevant=80 / uncertain=0 / not_relevant=20`、OOV feature rate は約 0.777 だった。ただし bootstrap の not_relevant が Sourcegraph の汎用releaseへ偏っているため、このきれいな分離を「汎化性能が十分」と解釈しない。確定済み human gold と新しい live shadow の両方を見て判断する。

ここでも production promotion は行わず、Phase 4C / AI外電の公開判断への接続には進まない。

実RSSデータで candidate を学習・再評価する例:

```bash
python -m gaiden_sieve train \
  --profile ai-gaiden \
  --training-data profiles/ai-gaiden/bootstrap \
  --evaluation-data profiles/ai-gaiden/real-holdout

python -m gaiden_sieve evaluate \
  --profile ai-gaiden \
  --training-data profiles/ai-gaiden/bootstrap \
  --evaluation-data profiles/ai-gaiden/real-holdout \
  --candidate <model_version>
```

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

Windows PowerShell では仮想環境の有効化だけ次のようになります。

```powershell
.venv\Scripts\Activate.ps1
```

## CLI

### 1. candidate を学習・保存する

```bash
python -m gaiden_sieve train --profile ai-gaiden
```

学習済み Pipeline を joblib へ保存したあと、**保存した candidate を再ロードして holdout 評価**します。出力 JSON には `model_version`、metrics、quality gate、data lineage が含まれます。

### 2. 保存済み candidate を再評価する

```bash
python -m gaiden_sieve evaluate \
  --profile ai-gaiden \
  --candidate <model_version>
```

`--candidate` を省略すると、metadata の `created_at` が最も新しい candidate を使います。

再評価時には現在の教師データ SHA-256 と candidate metadata の `training_data_hash` を比較します。教師データが変わっていれば、別データを同じ candidate の成績として扱わないよう停止します。

### 3. gate を通った candidate を明示的に昇格する

```bash
python -m gaiden_sieve promote \
  --profile ai-gaiden \
  --candidate <model_version>
```

Precision / Recall が現在の profile gate を下回る candidate は拒否され、production は更新されません。

### 4. production model で1件分類する

```bash
python -m gaiden_sieve classify \
  --profile ai-gaiden \
  --title "OpenAI releases a new model" \
  --summary "The model improves reasoning and tool use."
```

Phase 1 と違い、`classify` はその場で再学習せず、**明示 promotion 済みの production model** をロードします。

### 5. candidate の再現性・lineage を確認する

```bash
python -m gaiden_sieve verify \
  --profile ai-gaiden \
  --candidate <model_version>
```

data hash、code commit、seed / split、hyperparameter、dependency version、recorded metrics を確認し、同じ candidate の holdout 評価を再実行します。チェックに失敗した場合は終了コード 2 を返すため、CI でも利用できます。

### 6. production model で JSONL を一括分類する

入力 JSONL は最低限 `id / source / title` を持ち、`summary` は省略できます。

```bash
python -m gaiden_sieve classify \
  --profile ai-gaiden \
  --input incoming.jsonl \
  --output classified.jsonl \
  --uncertain-output uncertain.jsonl
```

batch でも1件分類と同じ profile threshold を使います。uncertain queue には high / low confidence item は入りません。

### 7. drift report と uncertain queue を生成する

```bash
python -m gaiden_sieve drift \
  --profile ai-gaiden \
  --input incoming.jsonl
```

既定では `reports/ai-gaiden/drift-<timestamp>.json` と `reports/ai-gaiden/uncertain.jsonl` を生成します。`--output` と `--uncertain-output` で保存先を変更できます。

## Phase 3 で確認する MLOps の基本

**artifact** はコードとは別に管理すべき学習成果物です。`model.joblib` だけを残しても「何から作られたか」が分からなければ、後から評価や比較ができません。そのため metadata と data lineage を対にします。

**candidate と production** を分けることで、「新しいモデルを作れた」と「本番採用してよい」を別の判断にできます。quality gate は、その間に置く最低品質の防波堤です。

**promotion を明示操作にする**のは、v0 では自動化より観測可能性を優先するためです。Phase 3 の CI は candidate の学習・評価・再現性確認までを自動化しますが、production への昇格は自動化しません。

**drift と uncertain queue** は、モデルが間違ったと断定するためではなく「入力が変わってきたか」「どの例を次に見直すべきか」を残すための運用データです。コードが正常でも入力分布は変わる、という MLOps の別の故障モードを観測します。

また、TF-IDF と Logistic Regression は引き続き1本の `Pipeline` として保存するため、学習時と production 推論時で同じ前処理を使います。`label`、`label_reason`、`label_source` は特徴量へ入りません。

> [!NOTE]
> 既存20件の `labeled.jsonl` は lifecycle と MLOps loop の回帰確認用 fixture として残しています。実RSSの Phase 4B.1 データは `bootstrap/` と `real-holdout/` に分離し、人間レビュー履歴は `gold-review.jsonl`、確定済み human gold は `gold.jsonl` で別管理します。
