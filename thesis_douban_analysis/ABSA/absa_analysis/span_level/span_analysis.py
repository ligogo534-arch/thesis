from pathlib import Path
import pandas as pd
import json

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR.parent.parent / "absa_overall" / "absa_full_output" / "ABSA_Qwen3_最终全量结果.xlsx"
OUTPUT_FILE = BASE_DIR / "absa_span_level.csv"
'''
#1.json展开
df = pd.read_excel(INPUT_FILE)
rows = []

for _, row in df.iterrows():
    if row["LLM_JSON解析状态"] != "成功":
        continue
    try:
        data = json.loads(row["LLM_JSON解析"]) if isinstance(row["LLM_JSON解析"], str) else row["LLM_JSON解析"]
    except Exception:
        continue

    if not isinstance(data, list):
        continue

    for item in data:
        if not isinstance(item, dict):
            continue
        rows.append({
            "ABSA_ID": row["ABSA_ID"],
            "语料来源": row["语料来源"],
            "book": row["book"],
            "time": row["time"],
            "content": row["content"],
            "text_span": item.get("text_span"),
            "aspect": item.get("aspect"),
            "polarity": item.get("polarity"),
            "target": item.get("target")
        })

span_df = pd.DataFrame(rows)
#span_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"原始评论数：{len(df)}")
print(f"展开后的text span数：{len(span_df)}")
print(f"输出文件：{OUTPUT_FILE}")


#2.检查字段是否完整

check = span_df.groupby("ABSA_ID").agg({"语料来源": "nunique", "book": "nunique", "time": "nunique", "content": "nunique"})
print("语料来源不一致：", (check["语料来源"] > 1).sum())
print("book不一致：", (check["book"] > 1).sum())
print("time不一致：", (check["time"] > 1).sum())
print("content不一致：", (check["content"] > 1).sum())
'''

#3.计算原文件展开应该有多少行
'''
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR.parent.parent / "absa_overall" / "absa_full_output" / "ABSA_Qwen3_最终全量结果.xlsx"

df = pd.read_excel(INPUT_FILE)

def count_json_items(x):
    try:
        data = json.loads(x) if isinstance(x, str) else x
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0

df["理论text_span数"] = df["LLM_JSON解析"].apply(count_json_items)

print("原始评论数：", len(df))
print("有JSON对象的评论数：", (df["理论text_span数"] > 0).sum())
print("理论展开后的text span总数：", df["理论text_span数"].sum())
'''
result = pd.read_excel(INPUT_FILE)
#print(f"distinct ABSA_ID数量：{result['ABSA_ID'].nunique()}")

valid = result["LLM_JSON解析"].astype(str).str.strip() != "[]"
count = result.loc[valid, "ABSA_ID"].nunique()

print(f"JSON输出不为空的distinct ABSA_ID数量：{count}")