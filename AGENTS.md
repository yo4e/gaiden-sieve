# GAIDEN SIEVE — 実装者向けガイド

このリポジトリを別セッション・別エージェントで継続するときは、**最初に `DESIGN.md` を通読すること**。

`DESIGN.md` が製品設計・学習目標・MLOps 方針の正本であり、このファイルは実装時の補助ルールと引き継ぎメモである。

## プロジェクトの目的

GAIDEN SIEVE は、RSS/Atom の記事を媒体ごとの編集方針に照らして分類する、小さな機械学習システムである。

最初の profile は `ai-gaiden`。

v0 の主目的は高性能な分類器を作ることではなく、次の MLOps の最小ループを、山田佳江が実物を追いながら理解できる形で一周すること。

```text
data + code
    ↓
  train
    ↓
 candidate
    ↓
 evaluate
    ↓
 quality gate
    ↓
 promote
    ↓
 production
    ↓
 monitor / drift
    ↓
 retrain
```

## v0 の技術方針

- Python
- scikit-learn
- TF-IDF
- Logistic Regression
- JSONL または CSV の教師データ
- joblib によるモデル保存
- pytest
- GitHub Actions
- profile ごとに教師データ・モデル・評価基準を分離

最初から LLM、GPU、Kubernetes、Kubeflow、SageMaker 等へ広げない。

まずローカルと GitHub Actions だけで最小ループを完成させる。

## 実装順序

`DESIGN.md` の Phase Plan に従う。特に最初は RSS 接続をしない。

1. package skeleton を作る
2. `ai-gaiden` profile と小さな fixture を用意する
3. fixture を読み込む
4. TF-IDF + Logistic Regression を学習する
5. Precision / Recall / F1 を計測する
6. candidate artifact と metadata を保存する
7. quality gate を実装する
8. pass した candidate だけ production に promote する
9. production model で新規記事を分類する
10. drift report と uncertain queue を追加する
11. pytest と GitHub Actions を整える
12. 最後に AI外電の実RSSを shadow mode で接続する

実装を進めるたびに、何を学ぶための工程なのかが `DESIGN.md` と対応していることを確認する。

## コードと言語のルール

- ファイル名、変数名、関数名、クラス名、CLI、JSON/YAML のキーは原則 **英語**。
- README、設計文書、学習向け説明は原則 **日本語**。
- **コード内コメントは日本語で書く。**
- ただし全行にコメントを付けない。山田佳江がコードを開いたときに「ここで何をしているか」「なぜこの処理が必要か」が追える程度にする。
- Python の一般常識を逐一説明するコメントより、機械学習・MLOps の意味が分かるコメントを優先する。

良い例:

```python
# 学習時と推論時で同じ前処理を使えるよう、TF-IDF と分類器を1本の Pipeline にまとめる。
model = Pipeline([...])

# test データは学習に使わず、最後の成績確認だけに使う。
y_pred = model.predict(x_test)

# quality gate を通っていないモデルは production に上書きしない。
if not gate.passed:
    raise PromotionRejected(...)
```

避けたい例:

```python
# 変数に1を足す
count += 1

# ファイルを開く
with open(path) as f:
    ...
```

コメントは「コードを日本語訳する」のではなく、**設計意図を説明する**ために使う。

## 学習者向けの実装方針

このプロジェクトでは、動けばよいだけの実装にしない。

山田佳江が MLOps を学ぶため、重要な箇所では以下を読み取れる構造にする。

- 何が training data なのか
- どこで feature 化しているのか
- どこで model が学習されるのか
- candidate と production の違い
- なぜ test set を分けるのか
- Precision / Recall がどこで計算されるのか
- どのデータ・コードから model artifact が作られたか
- quality gate がなぜ必要なのか
- drift が「コードの故障」と違うこと

必要なら docstring や README の短い補足も使う。ただし教材化しすぎて実装自体を読みにくくしない。

## 重要な設計制約

- v0 の学習モデルは二値分類: `relevant` / `not_relevant`。
- `uncertain` は独立クラスではなく予測確率の中間帯として扱う。
- 基本入力は `title + summary`。`label` や `label_reason` を特徴量へ混ぜない。
- data leakage を避ける。
- random seed、依存バージョン、データ hash、Git commit、主要 hyperparameter を可能な範囲で記録する。
- promotion は v0 では明示操作。CI が勝手に production model を更新しない。
- AI外電への接続は最初は shadow mode。自動掲載へ直結しない。
- 共通エンジンを媒体ごとに fork しない。profile で編集方針・教師データ・モデルを分離する。

## 迷ったときの優先順位

1. `DESIGN.md` の意図に忠実か
2. 山田佳江が読んで学べるか
3. 再現・評価・追跡ができるか
4. 小さく動くか
5. その後で拡張性や高度化を考える

企業級 MLOps の見た目を再現するために、不要なインフラや抽象化を増やさない。

## 完成条件

v0 の Definition of Done は `DESIGN.md` の「18. v0 の Definition of Done」を正本とする。

別セッションで実装を開始する場合は、まずリポジトリの現在状態を確認し、そのチェックリストの未完了項目から続けること。
