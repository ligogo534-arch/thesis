# -*- coding: utf-8 -*-
"""
构建短评(短评)+书评(長評)共用的自定义词典 与 停用词表候选清单
运行方式: python build_dict_stopwords.py

产出文件(均在 ./output/ 目录下):
1. custom_dict_test_result.csv   —— 专有名词分词测试结果,人工确认后整理成自定义词典
2. word_freq_for_review.csv      —— 高频词统计(含来源:短评/书评),人工标注后整理成停用词表
3. custom_dict_template.txt      —— 自定义词典模板(需人工补充/确认)
4. stopwords_candidate.txt       —— 停用词候选清单模板(需人工确认)

使用前请修改下方 CONFIG 中的文件路径。
"""

import re
import jieba
import pandas as pd
from collections import Counter
from pathlib import Path

SHORT_CSV_PATH = "../../data_cleaning2/text_cleaning/duanping_clean_base.csv"
LONG_CSV_PATH = "../../data_cleaning2/text_cleaning/shuping_clean_base.csv"
OUTPUT_DIR = Path("./output")
TOP_N_WORDS = 300                # 高频词统计中导出前N个供人工审查
MIN_WORD_LEN = 1                 # 统计时保留的最小词长(设为1可以看到所有,包括语气词单字)

# 已知需要重点检查的专有名词
CANDIDATE_TERMS = ["上野千鶴子","上野千鹤子","厌女","始于极限","从零开始的女性主义","父权制与资本主义","女性主义","父权制",
    "东大","东京大学","入学式","铃木凉美","再生产劳动","男性同性社会性欲望","典型女性气质","女性气质","政治抑郁","波伏娃","男性气概","林真理子"]


def load_and_merge_content(short_path, long_path):
    """读取短评与书评的 content 列,合并为一个语料列表,并标记来源"""
    texts = []

    short_df = pd.read_csv(short_path)
    for t in short_df["content"].dropna():
        texts.append({"content": str(t), "source": "短評"})

    long_df = pd.read_csv(long_path)
    for t in long_df["content"].dropna():
        texts.append({"content": str(t), "source": "書評"})

    print(f"短評: {len(short_df)} 件, 書評: {len(long_df)} 件, 合計テキスト: {len(texts)} 件")
    return texts


def test_proper_nouns(candidate_terms):
    """在未加载自定义词典的情况下,测试候选专有名词是否被正确切分为一个整体"""
    results = []
    for term in candidate_terms:
        seg_result = list(jieba.cut(term))
        is_split_correctly = (len(seg_result) == 1)
        results.append({
            "候補語": term,
            "デフォルト分詞結果": " / ".join(seg_result),
            "分割数": len(seg_result),
            "単一トークンか": is_split_correctly,
        })
    return pd.DataFrame(results)

def word_frequency_analysis(texts, top_n):
    """对短评+书评合并语料做分词与词频统计,分别记录在短评/书评中的出现次数"""
    counter_total = Counter()
    counter_short = Counter()
    counter_long = Counter()

    for item in texts:
        words = jieba.cut(item["content"])
        for w in words:
            w = w.strip()
            if len(w) < MIN_WORD_LEN:
                continue
            if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", w):
                # 纯标点符号跳过,不进入频次统计(标点会单独处理,不计入停用词候选审查)
                continue
            counter_total[w] += 1
            if item["source"] == "短評":
                counter_short[w] += 1
            else:
                counter_long[w] += 1

    most_common = counter_total.most_common(top_n)
    rows = []
    for word, freq in most_common:
        rows.append({
            "単語": word,
            "総頻度": freq,
            "短評での頻度": counter_short.get(word, 0),
            "書評での頻度": counter_long.get(word, 0),
            "文字数": len(word),
            "判定_ストップワード候補": "",   # 人工記入欄: Y/N
            "判定_固有名詞候補": "",         # 人工記入欄: Y/N
            "備考": "",
        })
    return pd.DataFrame(rows)


def write_template_files(output_dir):
    """输出自定义词典与停用词表的空白模板,附使用说明"""
    dict_template = output_dir / "custom_dict_template.txt"
    with open(dict_template, "w", encoding="utf-8") as f:
        f.write("# jieba 自定义词典模板\n")
        f.write("# 格式: 词语 [词频(可选)] [词性(可选)]\n")
        f.write("# 例: 上野千鶴子 1000 nr\n")
        f.write("# 请参考 custom_dict_test_result.csv 的测试结果, 将「単一トークンか=False」的词语补充到下方\n\n")
        for term in CANDIDATE_TERMS:
            f.write(f"# {term}\n")

    stopwords_template = output_dir / "stopwords_candidate.txt"
    with open(stopwords_template, "w", encoding="utf-8") as f:
        f.write("# 停用词候选清单模板\n")
        f.write("# 请从 word_freq_for_review.csv 中筛选出「判定_ストップワード候補=Y」的词语, 整理到此文件, 每行一个词\n")
        f.write("# 建议先合并通用停用词表(如哈工大停用词表), 再追加本清单中的网络用语/语气词\n\n")

    print(f"模板文件已生成: {dict_template}, {stopwords_template}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Part A: 专有名词分词测试 ---
    print("\n=== Part A: 専有名詞の分詞テスト(自定义词典构建用) ===")
    df_terms = test_proper_nouns(CANDIDATE_TERMS)
    df_terms.to_csv(OUTPUT_DIR / "custom_dict_test_result.csv", index=False, encoding="utf-8-sig")
    print(df_terms.to_string(index=False))
    print(f"\n結果を保存: {OUTPUT_DIR / 'custom_dict_test_result.csv'}")
    print("「単一トークンか」が False の語は、自定义词典に追加することを推奨します。")

    # --- Part B: 高频词统计 ---
    print("\n=== Part B: 高頻度語統計(停用词表构建用) ===")
    texts = load_and_merge_content(SHORT_CSV_PATH, LONG_CSV_PATH)
    df_freq = word_frequency_analysis(texts, TOP_N_WORDS)
    df_freq.to_csv(OUTPUT_DIR / "word_freq_for_review.csv", index=False, encoding="utf-8-sig")
    print(f"上位{TOP_N_WORDS}語を保存: {OUTPUT_DIR / 'word_freq_for_review.csv'}")
    print("CSVを開き、「判定_ストップワード候補」列に Y/N を人手で記入してください。")

    # --- 模板文件 ---
    write_template_files(OUTPUT_DIR)

    print("\n=== 完了 ===")
    print("次のステップ:")
    print("1. custom_dict_test_result.csv を確認し、正しく分割されなかった固有名詞を custom_dict_template.txt に追記")
    print("2. word_freq_for_review.csv を確認し、意味を持たないネットスラング・語気詞を stopwords_candidate.txt に追記")
    print("3. 完成した自定义词典・停用词表を、本パイプラインの前処理スクリプトで読み込む")


if __name__ == "__main__":
    main()