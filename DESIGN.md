# GAIDEN SIEVE 設計書

> RSS に入ってきた記事を、媒体ごとの編集方針に照らして「載せたい / 載せない / 判断保留」に仕分ける、小さな機械学習分類器。
>
> この文書は実装仕様であると同時に、MLOps を実際に作りながら理解するための学習ノートでもある。

## 1. 目的

GAIDEN SIEVE は、RSS/Atom から取得した記事の `title` と `summary` を主な入力として、ある媒体に掲載する価値があるかを分類する。

最初の対象は **AI外電** とする。

例:

```text
OpenAI releases a new reasoning model
→ relevant

Apple announces a new phone case
→ not_relevant

NVIDIA announces a new platform
→ uncertain
```

ただし、GAIDEN SIEVE 自体を AI外電専用にしない。

設計の基本は次の通り。

- **分類エンジンは共通**
- **教師データ・モデル・評価基準は媒体ごとに分離**
- 将来は `ai-gaiden`、`tech-gaiden`、`science-gaiden` など複数 profile を持てる

つまり、同じ記事でも媒体ごとに正解が違ってよい。

```text
GAIDEN SIEVE
├── ai-gaiden profile
├── tech-gaiden profile
└── future profiles ...
```

---

## 2. 今回の学習目標

このプロジェクトの第一目的は、巨大なAIモデルを作ることではない。

**「機械学習モデルを作って終わり」ではなく、学習・評価・公開・監視・更新までを一つの運用として扱う MLOps の基本形を、小さいシステムで一周すること**を目標にする。

MLOps を一言で書くと:

```text
data + code
    ↓
  train
    ↓
  model
    ↓
 evaluate
    ↓
 promote
    ↓
 production
    ↓
 monitor
    ↓
 retrain
```

普通のソフトウェアでは、主に `code → test → deploy` を管理する。

機械学習では、それに加えて **data と model もバージョン管理・評価の対象になる**。

同じコードでも、学習データが変われば別のモデルになる。モデルが正常に動いていても、世の中の言葉や記事傾向が変われば精度が落ちる。そのため「コードが壊れていない」だけでは十分ではない。

---

## 3. v0 の方針

最初は意図的に小さくする。

### 採用するもの

- Python
- scikit-learn
- TF-IDF
- Logistic Regression
- JSONL または CSV の教師データ
- joblib によるモデル保存
- pytest
- GitHub Actions
- Git によるコード・設定・小規模データの履歴管理

### 最初は採用しないもの

- 独自LLMの事前学習
- GPU
- Kubernetes
- Kubeflow
- SageMaker などの大規模クラウドMLOps基盤
- 常時稼働するモデルサーバ
- 大規模な Feature Store
- 大規模な Model Registry サービス

理由は単純で、**MLOps の構造を学ぶ前にインフラ学習が主役になってしまうのを避けるため**。

v0 はローカルPCと GitHub Actions だけで完結させる。

---

## 4. 「小さい分類器」とは何か

v0 では LLM を分類器として使わない。

使うのは次の2段構成。

```text
text
 ↓
TF-IDF
 ↓
Logistic Regression
 ↓
probability
```

### 4.1 TF-IDF

コンピュータは文章そのものを Logistic Regression に渡せないので、まず文章を数値に変える必要がある。

TF-IDF は、文章中の単語や語句を「どれくらいその文書らしい特徴か」という数値に変換する方法。

大まかには:

- その記事によく出る語は重要度が上がる
- どの記事にも出る語は重要度が下がる

たとえば学習データの中で、

```text
LLM
inference
model
machine learning
```

が `relevant` 側によく現れ、

```text
earnings
store opening
smartphone case
```

が `not_relevant` 側によく現れれば、その違いを数値として表現できる。

TF-IDF 自体は「意味」を人間のように理解しているわけではない。**語の出現パターンを特徴量に変換している**。

### 4.2 Logistic Regression

Logistic Regression は、その特徴量から「どちらのクラスらしいか」を学習する小さな分類モデル。

名前に `Regression` とあるが、二値分類にも非常によく使われる。

出力はたとえば:

```text
P(relevant) = 0.94
```

のような確率として扱える。

これにしきい値を設ける。

```text
>= 0.80       → relevant
0.40 - 0.80   → uncertain
< 0.40        → not_relevant
```

実際のしきい値は、評価結果を見て profile ごとに決める。

---

## 5. 教師データをどう作るか

### 5.1 Bootstrap labeling

最初の教師データは LLM によって作ってよい。

RSS記事ごとに、少なくとも次を保存する。

```json
{
  "id": "example-001",
  "source": "openai-news",
  "title": "OpenAI releases ...",
  "summary": "...",
  "label": "relevant",
  "label_source": "llm",
  "label_reason": "AI model release",
  "labeled_at": "2026-09-10T00:00:00Z"
}
```

重要なのは、**正解ラベルだけでなく「誰が / 何が付けたラベルか」を残すこと**。

`label_source` を残す理由は、将来:

```text
llm
human
rule
imported
```

のように教師データの由来を区別できるようにするため。

### 5.2 LLM ラベルは絶対の正解ではない

LLM が教師になれば、人間の手作業は大きく減る。

ただし、そのモデルを LLM のラベルだけで評価すると、

> 「そのLLMの判断をどれだけ上手にコピーできたか」

だけを測る危険がある。

そのため将来的には、小さくてもよいので `gold` データセットを持つ。

`gold` は「AI外電として本当に載せたい / 載せたくない」という編集判断を固定した評価用データ。

v0 では LLM ラベルだけで始めてよい。gold set は Phase 2 以降で追加する。

---

## 6. ラベル設計

v0 では学習モデル自体は二値分類とする。

```text
1 = relevant
0 = not_relevant
```

`uncertain` は独立した学習クラスではなく、**予測確率の中間帯**として扱う。

例:

```text
0.91 → relevant
0.57 → uncertain
0.11 → not_relevant
```

この設計にする理由:

- 最初のモデルを単純に保てる
- `uncertain` の意味を profile ごとのしきい値で調整できる
- 将来 active learning に接続しやすい

---

## 7. データ分割

教師データを全部学習に使ってはいけない。

最低限、次に分ける。

```text
train
validation
 test
```

### train

モデルが実際に学習するデータ。

### validation

しきい値・特徴量・ハイパーパラメータを調整するときに使う。

### test

最後の成績表。

**test を見ながらモデルを調整すると test も実質的に学習に使ったことになる**ので、基本的には最後まで触りすぎない。

v0 ではデータ量が少ない可能性があるため、まず:

```text
train: 80%
test: 20%
```

でもよい。

データが増えたら train / validation / test に分離する。

さらに実運用に近づけるなら、ランダム分割だけでなく **時間で分ける評価**も重要になる。

例:

```text
Jan-Jun → train
Jul     → validation
Aug     → test
```

RSSニュースは時代によって語彙が変化するため、未来の記事を過去の記事でどれだけ判定できるかを見る方が実運用に近い。

---

## 8. 評価指標

Accuracy だけを見ない。

AI外電では、特に次を使う。

### Precision

`relevant` と判定した記事のうち、本当に relevant だった割合。

```text
Precision = 正しく relevant とした数
            ----------------------
            relevant と判定した総数
```

Precision が低いと、AIではない記事が大量に混ざる。

### Recall

本当に relevant な記事のうち、どれだけ拾えたか。

```text
Recall = 正しく relevant とした数
         ----------------------
         本当に relevant な総数
```

Recall が低いと、重要なAIニュースを取りこぼす。

### F1

Precision と Recall のバランスを見る指標。

### AI外電での初期方針

v0 の仮の promotion gate:

```text
precision >= 0.95
recall    >= 0.80
```

これは固定の真理ではない。

「多少取りこぼしても、関係ない記事を載せない」ことを重視するなら Precision を高める。

逆に「候補を広く拾って人間が後で見る」運用なら Recall を高めてもよい。

**評価指標はモデルの数学ではなく、製品の目的から決める。**

---

## 9. Candidate と Production

MLOps では「新しいモデルができた = 即本番」にはしない。

```text
train
 ↓
candidate
 ↓
evaluate
 ↓
quality gate
 ├─ fail → reject
 └─ pass → promote
              ↓
          production
```

モデルファイルには最低限、次のメタデータを対応づける。

```json
{
  "profile": "ai-gaiden",
  "model_version": "2026-09-10.1",
  "training_data_version": "sha256:...",
  "code_commit": "git-sha",
  "precision": 0.96,
  "recall": 0.84,
  "created_at": "2026-09-10T00:00:00Z"
}
```

ここが MLOps の重要な点。

**「どのモデルが良かったか」だけでなく、「そのモデルが何から作られたか」を再現できるようにする。**

---

## 10. Drift

モデルはコードが壊れなくても劣化する。

たとえば、過去のAI記事では:

```text
machine learning
neural network
LLM
```

が頻出していたが、新しい時代には:

```text
agentic workflow
reasoning system
synthetic media
AI PC
```

など別の言葉が増えるかもしれない。

このように **入力データの分布が変化すること**を data drift と呼ぶ。

さらに、本当に知りたいのは **モデル性能そのものが落ちたか**であり、これは concept drift と関連する。

v0 では複雑な統計監視をしない。

最初は次のような簡単な指標でよい。

- 平均予測確率
- uncertain の割合
- 未知語 / 語彙外傾向
- source ごとの予測分布
- 後から正解が得られた記事の Precision / Recall

例:

```text
normally:
uncertain_rate = 0.08

this week:
uncertain_rate = 0.31
```

こうなったら:

> 最近の記事、以前と何か違うのでは？

という再学習候補になる。

---

## 11. Active Learning 的な運用

全部の記事を LLM に毎回判定させる必要はない。

小さい分類器を production に置き、判断が難しいものだけ外部の先生へ戻す。

```text
incoming RSS
    ↓
production model
    ↓
probability
 ┌─────────────┐
 │ high / low  │ → 自動判定
 └─────────────┘
       │
   uncertain
       ↓
  LLM review
       ↓
new labeled data
       ↓
  retraining
```

これは active learning に近い考え方。

モデル自身が苦手な例を集め、その例を教師データへ追加することで効率よく改善する。

v0 では「LLMを自動呼び出す仕組み」まで必須にしない。

まずは `uncertain.jsonl` を生成できれば十分。

---

## 12. 複数媒体への拡張

GAIDEN SIEVE のコードは共通にし、profile を切り替える。

```text
profiles/
├── ai-gaiden/
│   ├── config.yml
│   ├── labeled.jsonl
│   └── gold.jsonl
│
└── tech-gaiden/
    ├── config.yml
    ├── labeled.jsonl
    └── gold.jsonl
```

`config.yml` の例:

```yaml
profile: ai-gaiden
positive_label: relevant
precision_gate: 0.95
recall_gate: 0.80
relevant_threshold: 0.80
not_relevant_threshold: 0.40
text_fields:
  - title
  - summary
```

同じ共通コードを使って:

```bash
python -m gaiden_sieve train --profile ai-gaiden
python -m gaiden_sieve train --profile tech-gaiden
```

とする。

こうすれば、媒体ごとに:

```text
ai-v3
tech-v2
science-v1
```

のように別々の production model を管理できる。

---

## 13. 想定ディレクトリ構成

```text
gaiden-sieve/
├── README.md
├── DESIGN.md
├── pyproject.toml
│
├── src/
│   └── gaiden_sieve/
│       ├── __init__.py
│       ├── train.py
│       ├── evaluate.py
│       ├── classify.py
│       ├── promote.py
│       ├── drift.py
│       ├── data.py
│       └── artifacts.py
│
├── profiles/
│   └── ai-gaiden/
│       ├── config.yml
│       ├── labeled.jsonl
│       └── gold.jsonl
│
├── artifacts/
│   └── ai-gaiden/
│       ├── candidate/
│       └── production/
│
├── reports/
│   └── ai-gaiden/
│
├── tests/
│   ├── test_data.py
│   ├── test_train.py
│   ├── test_classify.py
│   └── test_promotion.py
│
└── .github/
    └── workflows/
        └── mlops.yml
```

注意:

学習済みモデルは将来大きくなる可能性がある。

v0 の小さい `joblib` artifact は Git 管理してもよいが、サイズが増えたら GitHub Releases / artifact storage / object storage などへ移す。

---

## 14. CLI の想定

```bash
# 学習
python -m gaiden_sieve train --profile ai-gaiden

# candidate 評価
python -m gaiden_sieve evaluate --profile ai-gaiden

# quality gate を通った candidate を production へ昇格
python -m gaiden_sieve promote --profile ai-gaiden

# 1件分類
python -m gaiden_sieve classify \
  --profile ai-gaiden \
  --title "OpenAI releases a new model" \
  --summary "..."

# JSONL を一括分類
python -m gaiden_sieve classify \
  --profile ai-gaiden \
  --input incoming.jsonl \
  --output classified.jsonl

# drift report
python -m gaiden_sieve drift --profile ai-gaiden --input incoming.jsonl
```

---

## 15. GitHub Actions

GitHub Actions は「GitHub 上の別のコンピュータで自動実行される作業手順」と考えるとよい。

v0 では pull request / push ごとに:

```text
install
  ↓
test
  ↓
train (small fixture / test dataset)
  ↓
evaluate
  ↓
report
```

を実行する。

重要:

CI の目的は production model を毎回勝手に更新することではない。

まずは:

- コードが壊れていないか
- 学習処理が再現できるか
- 評価処理が実行できるか
- quality gate のロジックが正しいか

を自動確認する。

production promotion は v0 では明示的な操作にする。

将来、十分に安全になれば:

```text
scheduled retraining
→ evaluate
→ pass
→ approval
→ promote
```

へ発展できる。

---

## 16. 再現性

MLOps では「昨日は精度96%だったのに、今日やったら93%」が理由不明だと困る。

そのため、可能な範囲で:

- random seed を固定
- Python / dependency version を固定
- train/test split を記録
- training data hash を記録
- Git commit SHA を記録
- hyperparameters を記録

する。

たとえば:

```json
{
  "random_seed": 42,
  "test_size": 0.2,
  "ngram_range": [1, 2],
  "max_features": 20000,
  "classifier": "LogisticRegression",
  "C": 1.0
}
```

**再現性とは、完全に同じ数字を必ず出すことだけではなく、「なぜこのモデルがこの結果になったか追跡できる」ことも含む。**

---

## 17. データリーク

初心者が特に注意したい問題。

### 例1: 同じ記事が train と test の両方に入る

RSSの重複取得などで起こる。

モデルは内容を覚えているので、異常に高い精度になる。

### 例2: 未来の情報を過去モデルの学習に使う

実運用では知らないはずの情報が混ざる。

### 例3: LLM のラベル理由文を特徴量に入れる

```text
label_reason: "This is clearly an AI article"
```

まで学習入力に含めれば、答えを教えているのと同じ。

したがってモデル入力は原則:

```text
title + summary
```

に限定する。

`label`、`label_reason`、評価結果などは入力特徴量に混ぜない。

---

## 18. v0 の Definition of Done

最初の完成条件は、次をすべて満たすこと。

- [ ] `ai-gaiden` profile がある
- [ ] 教師データを読み込める
- [ ] TF-IDF + Logistic Regression を学習できる
- [ ] candidate model を保存できる
- [ ] Precision / Recall / F1 を出せる
- [ ] promotion gate を評価できる
- [ ] pass した model だけ production へ昇格できる
- [ ] production model で新しいRSS itemを分類できる
- [ ] `relevant / uncertain / not_relevant` を返せる
- [ ] model metadata を保存する
- [ ]簡単な drift report を出せる
- [ ] pytest が通る
- [ ] GitHub Actions でテスト・評価パイプラインが動く
- [ ] README に「何を学べるプロジェクトか」を説明する

ここまでできれば、MLOps の最小ループを一周したと言ってよい。

---

## 19. Phase Plan

### Phase 0 — Skeleton

- package structure
- profile format
- data schema
- minimal tests

### Phase 1 — First classifier

- TF-IDF
- Logistic Regression
- train / test split
- metrics
- local classify command

### Phase 2 — Model lifecycle

- candidate artifact
- metadata
- quality gate
- promotion
- production loading

### Phase 3 — MLOps loop

- GitHub Actions
- reproducibility checks
- drift report
- uncertain queue

### Phase 4 — AI外電との接続実験

- 実RSS item を入力
- offline batch classification
- AI外電の既存フローを壊さず比較運用
- 誤判定分析

この段階でも、いきなり自動掲載には接続しない。

まずは **shadow mode** で動かす。

shadow mode とは、分類結果を実際の公開判断には使わず、裏側で予測だけ出して現在の運用と比較する方法。

### Phase 5 — Model upgrade experiments

必要になった場合のみ:

- character n-gram
- Sentence Transformers embeddings + classifier
- compact Transformer classifier
- fine-tuning

へ進む。

重要なのは、新しいモデルを使うこと自体を目的にしないこと。

**v0 より評価指標・運用コスト・保守性のどこが改善するか**を比較して採用する。

---

## 20. 将来のモデル交換

GAIDEN SIEVE の設計は、分類器を交換できるようにする。

```text
v1: TF-IDF + Logistic Regression

v2: Sentence embedding + classifier

v3: fine-tuned Transformer
```

同じ test / gold set を使って比較すれば:

```text
model     precision   recall   latency   size
v1        0.95        0.82     5ms       1MB
v2        0.97        0.88     40ms      90MB
v3        0.98        0.90     120ms     300MB
```

のように、単なる「賢さ」だけでなく運用コストまで比較できる。

この比較こそ MLOps 的に重要。

最高精度のモデルが、必ずしも最良の production model ではない。

---

## 21. このプロジェクトで覚えたい用語

| 用語 | このプロジェクトでは何を意味するか |
| --- | --- |
| feature | title / summary をTF-IDFで数値化したもの |
| label | relevant / not_relevant の正解 |
| training | 教師データから分類器の重みを学ぶこと |
| inference | 学習済みモデルで新しい記事を分類すること |
| artifact | 学習して保存したモデルや評価レポート |
| metric | Precision / Recall / F1 などの成績 |
| candidate | 本番候補の新モデル |
| production | 現在採用されているモデル |
| promotion | candidate を production に昇格すること |
| pipeline | 学習→評価→昇格などの一連の処理 |
| drift | 時間経過で入力や正解傾向が変わること |
| retraining | 新しい教師データを追加して学習し直すこと |
| data lineage | どのデータからどのモデルが作られたかの履歴 |
| reproducibility | 同じ条件を追跡・再現できること |
| shadow mode | 本番判断に使わず裏で予測して比較する運用 |

---

## 22. 設計上の原則

1. **まず小さく一周する。**
   - 最初から企業級MLOps基盤を再現しない。

2. **モデルより評価を先に大事にする。**
   - 高性能そうなモデルを入れる前に、「何を良いとするか」を決める。

3. **媒体の編集方針をラベルとして学ぶ。**
   - 普遍的な「AIニュース判定器」を目指しすぎない。

4. **コード・データ・モデルを別物として管理する。**
   - MLOps の中心的な考え方。

5. **自動化は、観測できてから増やす。**
   - まず shadow mode、次に半自動、最後に必要なら自動化。

6. **不確実さを捨てない。**
   - 無理に二択にせず `uncertain` を運用上の第一級データとして扱う。

7. **AI外電に統合する前に、GAIDEN SIEVE 単体で検証可能にする。**
   - 学習用プロジェクトとしても、将来の再利用可能コンポーネントとしても重要。

---

## 23. 最初の一歩

実装の最初は、いきなり RSS 接続から始めない。

10〜30件程度の小さな手作りfixtureで:

```text
load data
→ train
→ evaluate
→ save model
→ classify one new item
```

まで通す。

その後、実RSSデータへ広げる。

この順番にする理由は、問題が起きたときに:

> データ取得が悪いのか
> 学習処理が悪いのか
> 評価処理が悪いのか

を切り分けやすくするため。

**MLOps はモデルを賢くする技術だけではなく、「どこで何が起きたか分かるシステムを作る技術」でもある。**

---

## 24. この設計の到達点

GAIDEN SIEVE v0 が完成したとき、理解できていてほしいことは次の一文にまとまる。

> 機械学習モデルは単独のファイルではなく、データ、コード、評価、履歴、監視、更新を含む運用システムの一部である。

その感覚を、自分で動かした小さな分類器を通して理解することが、このプロジェクトの本当の完成条件である。
