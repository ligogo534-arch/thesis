from pathlib import Path
import pandas as pd

'''
# 原始 Excel 路径
INPUT_FILE = Path("情感词汇本体.xlsx")
df = pd.read_excel(INPUT_FILE)

print("主情感分类：")
print(df["情感分类"].value_counts(dropna=False))

print("\n辅助情感分类：")
print(df["辅助情感分类"].value_counts(dropna=False))
'''

REVIEW_RESULT_FILE = Path("../../overall_output/output_1/result.csv")
df = pd.read_csv(REVIEW_RESULT_FILE, encoding="utf-8-sig")
'''
score_cols = ["乐", "好", "怒", "哀", "惧", "恶", "惊", "整体极性"]

bad = df[df[score_cols].isna().all(axis=1)]
print("8个分数全部为空的记录数：", len(bad))

for _, row in bad.iterrows():
    print("\n" + "=" * 70)
    print("sample_id:", row["sample_id"])
    print("data_type:", row["data_type"])
    print("book_id:", row["book_id"])
    print("time:", row["time"])
    print("文字数:", row["文字数"])
    print("DUTIR情感词数:", row["DUTIR情感词数"])
    print("content长度:", len(str(row["content"])))
    print("content前200字:", str(row["content"])[:200])
    print("content后200字:", str(row["content"])[-200:])
    print("\n分数：")
    print(row[score_cols])
'''
result_df = pd.read_csv(REVIEW_RESULT_FILE,encoding="utf-8-sig")
row = result_df.loc[result_df["文字数"].idxmax()]

print("sample_id:", row["sample_id"])
print("文字数:", row["文字数"])
print("乐:", row["乐"])
print("好:", row["好"])
print("怒:", row["怒"])
print("哀:", row["哀"])
print("惧:", row["惧"])
print("恶:", row["恶"])
print("惊:", row["惊"])
print("整体极性:", row["整体极性"])