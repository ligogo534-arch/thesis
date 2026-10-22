from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR.parent /  "review_level" / "review_aspect_ratio.csv"
OUTPUT_FILE = BASE_DIR / "annual_ratio.csv"

df = pd.read_csv(INPUT_FILE)

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df["year"] = df["time"].dt.year

group_cols = ["语料来源", "year", "aspect"]

annual = df.groupby(group_cols).agg(
    Positive=("Positive", "mean"),
    Negative=("Negative", "mean"),
    Neutral=("Neutral", "mean"),
    N=("ABSA_ID", "nunique")
).reset_index()

annual["Total"] = annual["Positive"] + annual["Negative"] + annual["Neutral"]

check = annual[~annual["Total"].round(10).eq(1)]
print(f"构成比不等于1的记录数：{len(check)}")
if len(check) > 0:
    print(check)

annual.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"评论×aspect数据量：{len(df)}")
print(f"年度×aspect数据量：{len(annual)}")
print(f"输出文件：{OUTPUT_FILE}")
print("\n各年度各aspect的平均构成比检查：")
print(annual["Total"].describe())