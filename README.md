# GAIDEN SIEVE

GAIDEN SIEVE は、RSS/Atom の記事を媒体ごとの編集方針に照らして分類する、小さな機械学習システムです。

このリポジトリの主目的は、分類精度だけを追うことではありません。教師データ、学習コード、モデル、評価、promotion、監視までを一つの運用として扱う **MLOps の最小ループ**を、実物を追いながら学べる形で作ります。設計の正本は [`DESIGN.md`](DESIGN.md) です。

## 現在の実装範囲: Phase 3

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

- `gaiden_sieve.train`: stratified 80/20 split、TF-IDF + Logistic Regression、candidate 作成
- `gaiden_sieve.artifacts`: joblib artifact、JSON metadata、data/model hash、candidate / production の保存・ロード
- `gaiden_sieve.evaluate`: Precision / Recall / F1 と profile ごとの quality gate
- `gaiden_sieve.promote`: gate を通った candidate だけを明示的に production へ昇格
- `gaiden_sieve.classify`: production model の `predict_proba()` を profile のしきい値へ通し、`relevant / uncertain / not_relevant` を返す
- `gaiden_sieve.batch`: unlabeled JSONL の batch classification と uncertain queue
- `gaiden_sieve.drift`: 平均予測確率、uncertain率、source別分布、OOV feature率の観測
- `gaiden_sieve.reproducibility`: data / code / config / dependency / metrics の追跡確認
- `python -m gaiden_sieve`: train / evaluate / verify / promote / classify / drift CLI

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
> 現在の20件は lifecycle と MLOps loop を確認するための手作りfixtureです。この小さなholdoutで高い Precision / Recall / F1 が出ても、実運用の性能を意味しません。実RSS、gold set、shadow mode、実運用での drift 解釈は後続Phaseで扱います。
