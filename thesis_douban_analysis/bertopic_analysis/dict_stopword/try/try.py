# -*- coding: utf-8 -*-
"""
候補停用詞の文書頻度(document frequency)診断スクリプト

目的:
    トピックのtop wordに頻出する語について、それが「多数のトピックに
    またがって出現する汎用語(=停用詞候補)」なのか「特定の意味領域に
    紐づく分析対象語」なのかを、直感ではなく全コーパスでの文書頻度で
    定量的に判定する。

設計方針:
    - run_bertopic_pipeline_v3.py 本体は一切変更・importしない。
      CSVパスと自定义词典パスのみを本体設定から手動で複製して使用する
      (値がズレた場合は下記 CONFIG を本体の CORPUS_CONFIGS / CUSTOM_DICT_PATH
       と突き合わせて修正すること)。
    - トークン化はjieba+自定义词典のみを適用し、停用詞除去は行わない
      (停用詞候補そのものが除去されてしまっては頻度を測れないため)。
    - 「文書」= 元のCSVの1行(1件の評論)。BERTopicのunit(chunk/sentence)
      ではなく、元の評論単位で頻度を測る。これはstopword表がトークナイザ
      レベル(≒評論全体からの語除去)に効くため、評論単位で見るのが
      判定目的に最も合致するという判断による。
    - ある語が評論本文中に「部分文字列として出現」した評論の件数を数える
      (トークン境界のズレによる過小カウントを避けるため、まず部分文字列
       一致で計測する。トークン単位での一致率は別途注記する)。

出力:
    ./stopword_candidate_doc_freq.csv
    列: corpus, word, doc_count, total_docs, doc_freq_ratio
"""

import re
from pathlib import Path

import jieba
import pandas as pd

# ============ CONFIG (本体 run_bertopic_pipeline_v3.py の設定値を手動複製) ============
CUSTOM_DICT_PATH = "../dict_stopword/output/custom_dict_template.txt"

CORPORA = [
    {
        "name": "短評",
        "csv_path": "../../data_cleaning2/text_cleaning/duanping_clean_base.csv",
        "content_col": "content",
    },
    {
        "name": "書評",
        "csv_path": "../../data_cleaning2/text_cleaning/shuping_clean_base.csv",
        "content_col": "content",
    },
]

# 判定対象の候補語(ユーザーとの議論で挙がった第一類・第二類をまとめて計測し、
# 閾値判断は出力後に人手で行う)
CANDIDATE_WORDS = [
    # 第一類: 高確信度停用詞候補
    "人", "看", "看到", "说", "不是", "本书",
    "上", "很多", "一种", "一人", "一位", "几本", "只", "这种", "那种",
    # 第二類: 要検討
    "问题", "好", "成为", "相信", "感觉", "过去", "产生", "认为",
]

OUTPUT_PATH = Path("./stopword_candidate_doc_freq.csv")
# =======================================================================


def load_custom_dict(path):
    p = Path(path)
    if p.exists():
        jieba.load_userdict(str(p))
        print(f"自定义词典を読み込みました: {path}")
    else:
        print(f"[警告] 自定义词典が見つかりません: {path}(未反映のまま続行)")


def tokenize(text):
    return [t.strip() for t in jieba.cut(str(text)) if t.strip()]


def compute_doc_freq(df, content_col, words, corpus_name):
    """
    各候補語について:
      - substring_doc_count: 評論本文に部分文字列として出現した評論数
      - token_doc_count: jieba分かち書き後のトークン列に完全一致で
                          出現した評論数(参考値。分詞のされ方次第で
                          substringより少なくなり得る)
    の両方を出す。判定は基本 substring 基準(語除去は元テキストレベルの
    処理であるため)、token基準は分詞の妥当性チェックの参考情報とする。
    """
    texts = df[content_col].dropna().astype(str).tolist()
    total_docs = len(texts)
    print(f"[{corpus_name}] 対象評論数: {total_docs}")

    print(f"[{corpus_name}] 分かち書き中(トークン基準の計測のため)...")
    tokenized_texts = [set(tokenize(t)) for t in texts]

    records = []
    for w in words:
        substring_hits = sum(1 for t in texts if w in t)
        token_hits = sum(1 for toks in tokenized_texts if w in toks)
        records.append({
            "corpus": corpus_name,
            "word": w,
            "doc_count_substring": substring_hits,
            "doc_freq_ratio_substring": round(substring_hits / total_docs, 4) if total_docs else 0.0,
            "doc_count_token_exact": token_hits,
            "doc_freq_ratio_token_exact": round(token_hits / total_docs, 4) if total_docs else 0.0,
            "total_docs": total_docs,
        })
    return pd.DataFrame(records)


def main():
    load_custom_dict(CUSTOM_DICT_PATH)

    all_results = []
    for corpus_cfg in CORPORA:
        csv_path = Path(corpus_cfg["csv_path"])
        if not csv_path.exists():
            print(f"[エラー] CSVが見つかりません: {csv_path} をスキップします")
            continue
        df = pd.read_csv(csv_path)
        df_result = compute_doc_freq(df, corpus_cfg["content_col"], CANDIDATE_WORDS, corpus_cfg["name"])
        all_results.append(df_result)

    if not all_results:
        print("[エラー] 有効なコーパスが1件もありませんでした。CONFIGのパスを確認してください。")
        return

    df_all = pd.concat(all_results, ignore_index=True)

    # 語ごとに見やすいよう、substring基準の頻度でソート(降順)し、
    # コーパス間で語を揃えて比較しやすい形にpivotしたビューも別途出力する
    df_all_sorted = df_all.sort_values(["word", "corpus"])
    df_all_sorted.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n結果を保存: {OUTPUT_PATH}")

    pivot = df_all.pivot(index="word", columns="corpus", values="doc_freq_ratio_substring")
    pivot_path = OUTPUT_PATH.with_name("stopword_candidate_doc_freq_pivot.csv")
    pivot.to_csv(pivot_path, encoding="utf-8-sig")
    print(f"コーパス間比較用ピボット表を保存: {pivot_path}")

    print("\n=== doc_freq_ratio_substring 上位(参考表示) ===")
    print(df_all.sort_values("doc_freq_ratio_substring", ascending=False)
          [["corpus", "word", "doc_freq_ratio_substring", "doc_freq_ratio_token_exact"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()