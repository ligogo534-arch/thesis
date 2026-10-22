#程度词清洗

from pathlib import Path
from openpyxl import Workbook
import pandas as pd

'''
INPUT_FILE = Path("degree_words_raw.txt")
OUTPUT_FILE = Path("degree_cleaning.xlsx")

# HowNet 的原始等级名称
LEVEL_NAMES = {
    1: "极其/最",
    2: "很",
    3: "较",
    4: "稍",
    5: "欠",
    6: "超",
}

rows = []
current_level = None

with INPUT_FILE.open("r", encoding="gb18030") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 判断六个等级
        for level in range(1, 7):
            if line.startswith(f"{level}."):
                current_level = level
                break
        else:
            # 当前行是词语
            if current_level is not None:
                rows.append([line,LEVEL_NAMES[current_level],"",""])

# 创建 Excel
wb = Workbook()
ws = wb.active
ws.title = "程度词清洗"
# 表头
ws.append(["词语","HowNet等级","清洗结果", "备注"])
# 写入数据
for row in rows:
    ws.append(row)
# 调整列宽
ws.column_dimensions["A"].width = 20
ws.column_dimensions["B"].width = 15
ws.column_dimensions["C"].width = 15
ws.column_dimensions["D"].width = 40

wb.save(OUTPUT_FILE)

print(f"完成：{OUTPUT_FILE}")
print(f"共提取 {len(rows)} 个词条")
'''

#写回程度词txt
INPUT_FILE = Path("degree_cleaning.xlsx")
OUTPUT_FILE = Path("degree_words.txt")

LEVEL_WEIGHTS = {
    "极其/最": 2.0,
    "超": 1.5,
    "很": 1.25,
    "较": 1.2,
    "稍": 0.8,
    "欠": 0.5,
}

df = pd.read_excel(INPUT_FILE)
df = df[df["清洗结果"].isna() | (df["清洗结果"].astype(str).str.strip() == "")].copy()
df["权重"] = df["HowNet等级"].map(LEVEL_WEIGHTS)

# 检查是否存在无法匹配的等级
invalid = df[df["权重"].isna()]

if not invalid.empty:
    print("以下词语的 HowNet 等级无法匹配：")
    print(invalid[["词语", "HowNet等级"]].to_string(index=False))
    raise ValueError("存在无法匹配的 HowNet 等级，请检查 Excel。")

# 检查是否存在重复词
duplicates = df[df["词语"].duplicated(keep=False)]
if not duplicates.empty:
    print("警告：存在重复词语：")
    print(duplicates[["词语", "HowNet等级", "权重"]].sort_values("词语").to_string(index=False))
    raise ValueError("存在一个词对应多个等级的情况，请先处理重复词后再生成 TXT。")

# 写入 TXT
with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(f"{row['词语']}\t{row['权重']}\n")

print(f"完成：{OUTPUT_FILE}")
print(f"最终程度词数量：{len(df)}")