# GAIDEN SIEVE

GAIDEN SIEVE は、RSS/Atom の記事を媒体ごとの編集方針に照らして分類する、小さな機械学習システムです。

このリポジトリの主目的は、分類精度だけを追うことではありません。教師データ、学習コード、モデル、評価、promotion、監視までを一つの運用として扱う **MLOps の最小ループ**を、実物を追いながら学べる形で作ります。設計の正本は [`DESIGN.md`](DESIGN.md) です。

## 現在の実装範囲: Phase 0

Phase 0 では、まだモデルを学習しません。後続の Phase で同じ前提を安全に使えるよう、次の土台だけを実装します。

- `src/gaiden_sieve/`: 共通エンジンの package skeleton
- `profiles/ai-gaiden/config.yml`: 媒体ごとの評価基準・判定しきい値
- `profiles/ai-gaiden/labeled.jsonl`: 小さな教師データ fixture
- `gaiden_sieve.data`: 教師データの schema と JSONL loader
- `gaiden_sieve.profile`: profile の schema と YAML loader
- `tests/`: schema と loader の最小テスト

### なぜモデルより先に schema を作るのか

機械学習ではコードだけでなく、**どんな形のデータを学習に使ったか**がモデルの挙動を左右します。Phase 0 でデータ形式と profile 形式を明示しておくと、Phase 1 以降の train / evaluate / classify が同じ契約に乗り、入力の混乱や data leakage を見つけやすくなります。

特に、モデルへ渡す本文は `title` と `summary` だけに限定します。`label` や `label_reason` は教師として必要ですが、特徴量に混ぜると答えを入力へ漏らす data leakage になるためです。

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

## Phase Plan

次の Phase 1 で TF-IDF + Logistic Regression、train/test split、Precision / Recall / F1、ローカル分類コマンドを追加します。
