# GAIDEN SIEVE

GAIDEN SIEVE は、RSS/Atom の記事を媒体ごとの編集方針に照らして分類する、小さな機械学習システムです。

このリポジトリの主目的は、分類精度だけを追うことではありません。教師データ、学習コード、モデル、評価、promotion、監視までを一つの運用として扱う **MLOps の最小ループ**を、実物を追いながら学べる形で作ります。設計の正本は [`DESIGN.md`](DESIGN.md) です。

## 現在の実装範囲: Phase 1

Phase 1 では、Phase 0 の schema と profile の上に最初の分類器を載せます。

```text
labeled data
    ↓
train / test split
    ↓
TF-IDF
    ↓
Logistic Regression
    ↓
predict_proba()
    ↓
Precision / Recall / F1
```

主な実装は次の通りです。

- `gaiden_sieve.train`: `random_seed=42` の stratified 80/20 split と scikit-learn `Pipeline`
- `TfidfVectorizer`: `title + summary` の語・語句の出現パターンを数値特徴量へ変換
- `LogisticRegression`: TF-IDF 特徴量から `relevant / not_relevant` の境界を学習
- `gaiden_sieve.evaluate`: test set だけを使った Precision / Recall / F1
- `gaiden_sieve.classify`: `predict_proba()` の `P(relevant)` を profile のしきい値へ通して `relevant / uncertain / not_relevant` を返す
- `python -m gaiden_sieve`: 学習結果の確認と1件分類を行う最小CLI

`uncertain` は第三の学習クラスではありません。`not_relevant_threshold <= P(relevant) < relevant_threshold` の中間帯を、運用上 `uncertain` と呼びます。

### この Phase で確認する MLOps の基本

**TF-IDF** は文章の意味を人間のように理解するのではなく、どの語や語句が各文書で特徴的かを数値化します。**Logistic Regression** はその数値から二値分類の境界を学び、確率を返します。

教師データを `train` と `test` に分けるのは、学習に使っていない記事で成績を見るためです。Phase 1 では `stratify` で両ラベルの比率を保ち、random seed を固定して、同じデータとコードなら同じ分割を再現できるようにしています。

Precision は「拾った記事の純度」、Recall は「拾うべき記事をどれだけ拾えたか」です。AI外電では関係ない記事を混ぜないことと、重要記事を取りこぼさないことのトレードオフをこの2つで見ます。F1 はそのバランス指標です。

また、TF-IDF と Logistic Regression を `Pipeline` にまとめることで、学習時と推論時に必ず同じ特徴量化を通します。`label`、`label_reason`、`label_source` はモデル入力へ入らず、`title + summary` だけを特徴量にすることで data leakage を防ぎます。

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

## Phase 1 CLI

学習し、holdout test set の Precision / Recall / F1 を表示します。

```bash
python -m gaiden_sieve train --profile ai-gaiden
```

同じ Phase 1 学習ループで作った in-memory model を使い、1件だけ分類します。

```bash
python -m gaiden_sieve classify \
  --profile ai-gaiden \
  --title "OpenAI releases a new model" \
  --summary "The model improves reasoning and tool use."
```

出力は JSON で、`probability_relevant` と `classification` を含みます。

Phase 1 ではモデル artifact を保存せず、CLI 実行ごとに小さな fixture から学習します。candidate artifact、metadata、quality gate、production promotion は Phase 2 で実装します。

> [!NOTE]
> 現在の20件は学習ループを確認するための手作りfixtureです。この小さなholdoutで高い Precision / Recall / F1 が出ても、実運用の性能を意味しません。実RSSやgold setでの評価は後続Phaseで行います。
