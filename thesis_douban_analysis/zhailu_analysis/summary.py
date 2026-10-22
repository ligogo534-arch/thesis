import pandas as pd
import re
import unicodedata
from pathlib import Path


INPUT_FILE = "../data_cleaning1/data_cleaned/zhailu_clean.csv"
OUTPUT_FILE = "sentence_frequency.csv"

df = pd.read_csv(INPUT_FILE,encoding="utf-8-sig")

print("原始投稿数:", len(df))

def split_sentences(text):
    if pd.isna(text):
        return []
    text = str(text)
    # 按 。！？！？ 拆分
    sentences = re.split(r"[。！？!?]+",text)
    # 去除空句
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences

def normalize_sentence(sentence):
    # Unicode统一
    sentence = unicodedata.normalize("NFKC",sentence)
    # 去空白
    sentence = re.sub(r"\s+","",sentence)
    sentence = re.sub(r"[「」『』\"“”]","",sentence)
    sentence = re.sub(r"[。、，,．.・；;：:！？!?（）()\[\]【】]","",sentence)
    return sentence.strip()


sentence_records = []

for _, row in df.iterrows():
    sentences = split_sentences(row["文本内容"])
    for sentence in sentences:
        normalized = normalize_sentence(sentence)
        if len(normalized) == 0:
            continue
        sentence_records.append({"sentence_normalized": normalized, "sentence_original": sentence, "用户名": row["用户名"], "引用出处": row["引用出处"]})


sentence_df = pd.DataFrame(sentence_records)

print("拆分后句子数:",len(sentence_df))

result = (sentence_df.groupby("sentence_normalized").agg(
        出现次数=("sentence_normalized", "count"),
        引用用户数=("用户名", "nunique"),
        原文示例=("sentence_original", "first"),
        引用出处=("引用出处", lambda x: " | ".join(sorted(set(str(i) for i in x if pd.notna(i) and str(i).strip() != ""))))
    ).reset_index())

result = result.sort_values(["引用用户数", "出现次数"],ascending=False)

result.to_csv(OUTPUT_FILE,index=False,encoding="utf-8-sig")

print("\n完成")
print("输出句子数:", len(result))
print("\nTop20:")
print(result.head(20))