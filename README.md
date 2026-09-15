# GAIDEN SIEVE

GAIDEN SIEVE は、RSS/Atom の記事を媒体ごとの編集方針に照らして分類する、小さな機械学習システムです。

このリポジトリの主目的は、分類精度だけを追うことではありません。教師データ、学習コード、モデル、評価、promotion、監視までを一つの運用として扱う **MLOps の最小ループ**を、実物を追いながら学べる形で作ります。設計の正本は [`DESIGN.md`](DESIGN.md) です。

## 現在の実装範囲: Phase 2

Phase 2 では、Phase 1 の TF-IDF + Logistic Regression を「学習して終わり」から **追跡・評価・明示昇格できる model lifecycle** へ進めます。

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
- `python -m gaiden_sieve`: train / evaluate / promote / classify の最小 CLI

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

## Phase 2 CLI

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

## この Phase で確認する MLOps の基本

**artifact** はコードとは別に管理すべき学習成果物です。`model.joblib` だけを残しても「何から作られたか」が分からなければ、後から評価や比較ができません。そのため metadata と data lineage を対にします。

**candidate と production** を分けることで、「新しいモデルを作れた」と「本番採用してよい」を別の判断にできます。quality gate は、その間に置く最低品質の防波堤です。

**promotion を明示操作にする**のは、v0 では自動化より観測可能性を優先するためです。CI が勝手に production を書き換える仕組み、drift、uncertain queue は Phase 3 の範囲です。

また、TF-IDF と Logistic Regression は引き続き1本の `Pipeline` として保存するため、学習時と production 推論時で同じ前処理を使います。`label`、`label_reason`、`label_source` は特徴量へ入りません。

> [!NOTE]
> 現在の20件は lifecycle を確認するための手作りfixtureです。この小さなholdoutで高い Precision / Recall / F1 が出ても、実運用の性能を意味しません。実RSS、gold set、drift監視は後続Phaseで扱います。
