# Issue #15 レビュー・検算資料

対象の固定コミット: [`59166ad4c2a70a54355e0c34dcfd3ea9f91b5e24`](https://github.com/yo4e/gaiden-sieve/commit/59166ad4c2a70a54355e0c34dcfd3ea9f91b5e24)

[Issue #15](https://github.com/yo4e/gaiden-sieve/issues/15) に対する独立レビューを、チャット添付ZIPから**履歴管理しやすい個別ファイル**として保存した。最初に [independent-review.md](independent-review.md) を読む。

| ファイル | 内容 |
| --- | --- |
| `independent-review.md` | レビュー全文。結論・根拠・次の最小検証・未検証範囲 |
| `recompute.py` | 公開済みの固定コミットの保存スコアを使った限定的な独立検算と合成例の重複群検査 |
| `review-checks.json` | 検算済み数値と制約。元ZIPに収録した出力の固定コピー |
| `score-extract.json` | 上記検算のため転記した最小限のスコアとラベル。元ZIPの固定コピー |

## 検算

リポジトリのルートから実行:

```sh
python experiments/issue15/recompute.py
```

Python標準ライブラリだけで動き、再学習・外部通信・本番変更はしない。出力はgitignore対象の `.diagnosis/issue15-review-checks/` に置く。そこに生成された `review-checks.json` と `score-extract.json` を、このディレクトリの保存版と比較できる。

元ZIP内の `verification/_source/support.py` と `verification/_source/evaluation_design.py` は、**すでにこのリポジトリの `experiments/issue13/` にあるため複写しない**。代わりに `recompute.py` は元ファイルのGit blob SHAを照合してから読み込む。元ZIPの `recompute-output.txt` も `review-checks.json` の標準出力と同内容なので重複保存しない。

この検算の入力は公開スコアの限定抜粋であり、元59件の再学習・全foldの再実行・元RSS本文の精読・全pytestを実施したことを意味しない。URL欠落を示す試験データは合成例であり、実際の元59件で漏洩が起きた証拠ではない。原記事title/summary、生RSS、モデルファイル、個人情報や認証情報は含めない。

この保存操作では元の分類モデル・設定・教師・goldを変更していない。レビュー後の独立評価や本番採用の可否は別判断。

## レビュー後の再開作業

[URL欠落の修正・元59件監査・6件入力票準備](followup.md)を追記した。歴史的レビュー5ファイルは保持し、確認済みと未実施を区別する。原入力の採否と3境界は本人判断として残している。
