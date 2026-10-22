from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR.parent / "span_level" / "absa_span_level.csv"
OUTPUT_FILE = BASE_DIR / "review_aspect_ratio.csv"
'''
df = pd.read_csv(INPUT_FILE)

group_cols = ["ABSA_ID", "语料来源", "book", "time", "content", "aspect"]

count_df = df.groupby(group_cols + ["polarity"], dropna=False).size().unstack(fill_value=0)

for p in ["正", "负", "中性"]:
    if p not in count_df.columns:
        count_df[p] = 0

count_df["span_total"] = count_df[["正", "负", "中性"]].sum(axis=1)
count_df["Positive"] = count_df["正"] / count_df["span_total"]
count_df["Negative"] = count_df["负"] / count_df["span_total"]
count_df["Neutral"] = count_df["中性"] / count_df["span_total"]

result = count_df.reset_index()[group_cols + ["正", "负", "中性", "span_total", "Positive", "Negative", "Neutral"]]
result = result.rename(columns={"正": "Positive_span", "负": "Negative_span", "中性": "Neutral_span"})

result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"原始text span数：{len(df)}")
print(f"评论×aspect数量：{len(result)}")
print(f"输出文件：{OUTPUT_FILE}")
'''
result = pd.read_csv(OUTPUT_FILE)
print(f"distinct ABSA_ID数量：{result['ABSA_ID'].nunique()}")