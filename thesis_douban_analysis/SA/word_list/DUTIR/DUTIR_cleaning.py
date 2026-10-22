from pathlib import Path
import pandas as pd

# 原始 Excel 路径
INPUT_FILE = Path("情感词汇本体.xlsx")

# 输出目录
OUTPUT_DIR = INPUT_FILE.parent / "cleaned"/ "cleaned_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORD_COL = "词语"
SENSE_COL = "词义序号"
PRIMARY_EMOTION_COL = "情感分类"
PRIMARY_INTENSITY_COL = "强度"
PRIMARY_POLARITY_COL = "极性"
# 辅助情感列不参与计算
SECONDARY_EMOTION_COL = "辅助情感分类"
SECONDARY_INTENSITY_COL = "强度.1"
SECONDARY_POLARITY_COL = "极性.1"

print("=" * 70)
print("DUTIR 词典清洗")
print("=" * 70)
print(f"\n输入文件：{INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"\n找不到文件：\n{INPUT_FILE}\n" f"请修改 INPUT_FILE。")

# 注意：不能把 "NA" 当成缺失值！
df = pd.read_excel(INPUT_FILE, keep_default_na=False)

print(f"原始行数：{len(df):,}")
required_columns = [
    WORD_COL,
    PRIMARY_EMOTION_COL,
    PRIMARY_INTENSITY_COL,
    PRIMARY_POLARITY_COL,
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"\n缺少必要列：{missing_columns}")

# 5. 基础清理
# 词语转字符串并去除首尾空格
df[WORD_COL] = (df[WORD_COL].astype(str).str.strip())

# 把字符串形式的空值处理成 NaN
df[WORD_COL] = df[WORD_COL].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
df[PRIMARY_EMOTION_COL] = (df[PRIMARY_EMOTION_COL].astype("string").str.strip())

# 强度和极性转为数值
df[PRIMARY_INTENSITY_COL] = pd.to_numeric(df[PRIMARY_INTENSITY_COL], errors="coerce")
df[PRIMARY_POLARITY_COL] = pd.to_numeric(df[PRIMARY_POLARITY_COL], errors="coerce")

# 6. 删除关键字段缺失的行
before_missing = len(df)
missing_required = df[
    df[WORD_COL].isna()
    | df[PRIMARY_EMOTION_COL].isna()
    | df[PRIMARY_INTENSITY_COL].isna()
    | df[PRIMARY_POLARITY_COL].isna()
].copy()

df = df.dropna(
    subset=[WORD_COL,PRIMARY_EMOTION_COL,PRIMARY_INTENSITY_COL,PRIMARY_POLARITY_COL,]).copy()
print(f"\n关键字段缺失而删除的行：" f"{len(missing_required):,}")

# 7. 按词语分组进行清洗
print("\n" + "=" * 70)
print("开始按词语进行冲突处理")
print("=" * 70)

clean_records = []
excluded_records = []
total_words = df[WORD_COL].nunique()
print(f"\n待处理唯一词语数：{total_words:,}")

for word, group in df.groupby(WORD_COL, sort=False):
    # A. 主情感分类
    emotions = (
        group[PRIMARY_EMOTION_COL]
        .dropna()
        .astype(str)
        .str.strip()
    )

    unique_emotions = sorted(
        emotions.unique().tolist()
    )
    # 如果主情感分类不唯一 → 剔除
    if len(unique_emotions) > 1:
        excluded_records.append({
            "词语": word,
            "剔除原因": "主情感分类冲突",
            "主情感分类": " / ".join(unique_emotions),
            "主强度": " / ".join(
                str(x)
                for x in sorted(
                    group[PRIMARY_INTENSITY_COL]
                    .dropna()
                    .unique()
                )
            ),
            "主极性": " / ".join(
                str(int(x)) if float(x).is_integer()
                else str(x)
                for x in sorted(
                    group[PRIMARY_POLARITY_COL]
                    .dropna()
                    .unique()
                )
            ),
            "词义数量": len(group),
        })

        continue

    emotion = unique_emotions[0]

    # B. 主极性
    polarities = (
        group[PRIMARY_POLARITY_COL]
        .dropna()
        .tolist()
    )

    unique_polarities = sorted(set(polarities))

    if len(unique_polarities) == 1:
        final_polarity = unique_polarities[0]
    else:
        # 计算众数
        polarity_counts = (pd.Series(polarities).value_counts())
        max_count = polarity_counts.max()
        modes = sorted(polarity_counts[polarity_counts == max_count].index.tolist())
        # 存在唯一众数 → 使用众数
        if len(modes) == 1:
            final_polarity = modes[0]
        # 没有唯一众数 → 剔除
        else:
            excluded_records.append({
                "词语": word,
                "剔除原因": "主极性冲突且不存在唯一众数",
                "主情感分类": emotion,
                "主强度": " / ".join(
                    str(x)
                    for x in sorted(
                        group[PRIMARY_INTENSITY_COL]
                        .dropna()
                        .unique()
                    )
                ),
                "主极性": " / ".join(
                    str(int(x)) if float(x).is_integer()
                    else str(x)
                    for x in unique_polarities
                ),
                "词义数量": len(group),
            })
            continue

    # C. 主强度
    intensities = (group[PRIMARY_INTENSITY_COL].dropna().astype(float))
    final_intensity = intensities.mean()

    # D. 保存清洗后的词
    clean_records.append({
        "词语": word,
        "情感分类": emotion,
        "强度": final_intensity,
        "极性": final_polarity,
    })

# 8. 转换为 DataFrame
clean_df = pd.DataFrame(
    clean_records,
    columns=[
        "词语",
        "情感分类",
        "强度",
        "极性",
    ]
)

excluded_df = pd.DataFrame(
    excluded_records,
    columns=[
        "词语",
        "剔除原因",
        "主情感分类",
        "主强度",
        "主极性",
        "词义数量",
    ]
)

# 9. 格式化强度
# 如果均值为整数，例如 5.0，则显示为 5
# 如果为非整数，例如 4.5，则保留小数
clean_df["强度"] = clean_df["强度"].round(4)

# 10. 统计清洗结果
original_rows = len(df)
original_unique_words = df[WORD_COL].nunique()

clean_words = len(clean_df)
excluded_words = len(excluded_df)

print("\n" + "=" * 70)
print("清洗结果")
print("=" * 70)

print(f"""
原始有效行数：              {original_rows:,}
原始唯一词语数：            {original_unique_words:,}

清洗后唯一词语数：          {clean_words:,}
剔除词语数：                {excluded_words:,}
""")

# 11. 输出剔除原因统计
if not excluded_df.empty:
    reason_counts = (excluded_df["剔除原因"].value_counts().reset_index())
    reason_counts.columns = ["剔除原因","词语数量",]

else:
    reason_counts = pd.DataFrame(columns=["剔除原因","词语数量",])

# 12. 输出清洗后的 Excel
clean_excel = OUTPUT_DIR / "DUTIR_clean.xlsx"
clean_df.to_excel(clean_excel,index=False)
print(f"\n✓ 清洗后的 Excel：")
print(clean_excel)

# 13. 输出剔除词 Excel
excluded_excel = OUTPUT_DIR / "DUTIR_excluded.xlsx"
excluded_df.to_excel(excluded_excel,index=False)
print(f"\n✓ 被剔除词语：")
print(excluded_excel)

# 14. 输出清洗统计 Excel
summary_excel = OUTPUT_DIR / "DUTIR_cleaning_summary.xlsx"
summary_df = pd.DataFrame({
    "项目": [
        "原始行数",
        "关键字段缺失行数",
        "原始有效行数",
        "原始唯一词语数",
        "清洗后唯一词语数",
        "剔除词语数",
    ],
    "数量": [
        len(df) + len(missing_required),
        len(missing_required),
        original_rows,
        original_unique_words,
        clean_words,
        excluded_words,
    ],
})

with pd.ExcelWriter(summary_excel,engine="openpyxl") as writer:
    summary_df.to_excel(writer,sheet_name="总体统计",index=False)
    reason_counts.to_excel(writer,sheet_name="剔除原因",index=False)
print(f"\n✓ 清洗统计：")
print(summary_excel)

# 15. 输出 TXT
txt_file = OUTPUT_DIR / "DUTIR_clean.txt"
with open(txt_file,"w",encoding="utf-8") as f:
    for _, row in clean_df.iterrows():
        word = str(row["词语"])
        emotion = str(row["情感分类"])
        intensity = row["强度"]
        polarity = row["极性"]

        # 强度整数化
        if float(intensity).is_integer():
            intensity_str = str(int(intensity))
        else:
            intensity_str = str(intensity)

        # 极性整数化
        if float(polarity).is_integer():
            polarity_str = str(int(polarity))
        else:
            polarity_str = str(polarity)

        f.write(
            f"{word}\t"
            f"{emotion}\t"
            f"{intensity_str}\t"
            f"{polarity_str}\n"
        )

print(f"\n✓ 清洗后的 TXT：")
print(txt_file)

# 16. 输出前20个被剔除词
if not excluded_df.empty:
    print("\n" + "=" * 70)
    print("前20个被剔除词")
    print("=" * 70)
    print(excluded_df.head(20).to_string(index=False))

# 17. 最终总结
print("\n" + "=" * 70)
print("DUTIR 清洗完成")
print("=" * 70)

print(f"""
最终可用于情感分析的词语：{clean_words:,}
剔除词语：                {excluded_words:,}

输出目录：
{OUTPUT_DIR}
""")

print("清洗规则：")
print("1. 主情感分类冲突 → 剔除")
print("2. 主极性唯一 → 直接使用")
print("3. 主极性冲突 → 取众数")
print("4. 主极性无唯一众数 → 剔除")
print("5. 主强度冲突 → 取均值")
print("6. 辅助情感不参与计算")
print("7. 每个词最终只保留一条记录")