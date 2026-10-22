from pathlib import Path
import pandas as pd

# 修改成你的 DUTIR Excel 文件路径
INPUT_FILE = Path("情感词汇本体.xlsx")

# 检查结果输出目录
OUTPUT_DIR = Path("dutir_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMNS = [
    "词语",
    "词性种类",
    "词义数",
    "词义序号",
    "情感分类",
    "强度",
    "极性",
    "辅助情感分类",
    "强度.1",
    "极性.1",
]


# ============================================================
# 3. 读取 Excel
# ============================================================

print("=" * 70)
print("DUTIR 词典结构检查")
print("=" * 70)

print(f"\n正在读取：{INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\n找不到文件：{INPUT_FILE}\n"
        f"请修改 INPUT_FILE 为实际的 DUTIR Excel 路径。"
    )

df = pd.read_excel(INPUT_FILE)

print(f"读取完成。")


# ============================================================
# 4. 检查列名
# ============================================================

print("\n" + "=" * 70)
print("一、列名检查")
print("=" * 70)

print("\n实际列名：")
for col in df.columns:
    print(f"  {col!r}")

missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

extra_columns = [
    col for col in df.columns
    if col not in EXPECTED_COLUMNS
]

if missing_columns:
    print("\n❌ 缺少以下列：")
    for col in missing_columns:
        print(f"  - {col}")

    raise ValueError(
        "\nDUTIR 文件列名与预期不一致，请检查 Excel。"
    )

else:
    print("\n✓ 所有预期列均存在。")

if extra_columns:
    print("\n提示：存在额外列：")
    for col in extra_columns:
        print(f"  - {col}")


# ============================================================
# 5. 基本统计
# ============================================================

print("\n" + "=" * 70)
print("二、基本统计")
print("=" * 70)

print(f"\n总行数：{len(df):,}")

unique_words = df["词语"].dropna().astype(str).nunique()

print(f"唯一词语数：{unique_words:,}")

duplicate_word_rows = df["词语"].duplicated(keep=False).sum()

print(f"涉及重复词语的行数：{duplicate_word_rows:,}")


# ============================================================
# 6. 缺失值检查
# ============================================================

print("\n" + "=" * 70)
print("三、缺失值检查")
print("=" * 70)

missing_stats = df[EXPECTED_COLUMNS].isna().sum()

print()

for col, count in missing_stats.items():
    if count > 0:
        print(f"⚠ {col}: {count:,} 个缺失")
    else:
        print(f"✓ {col}: 无缺失")


# ============================================================
# 7. 找出同一词语对应多个词义
# ============================================================

print("\n" + "=" * 70)
print("四、同一词语对应多个词义")
print("=" * 70)

# 对每个词语统计“词义序号”的不同值
word_sense_count = (
    df.groupby("词语")["词义序号"]
    .nunique(dropna=True)
)

multi_sense_words = word_sense_count[
    word_sense_count > 1
].index.tolist()

print(
    f"\n存在多个词义序号的词语数量："
    f"{len(multi_sense_words):,}"
)

if multi_sense_words:
    multi_sense_df = (
        df[df["词语"].isin(multi_sense_words)]
        .sort_values(["词语", "词义序号"])
    )

    print("\n前20个示例：")
    print(
        multi_sense_df[
            ["词语", "词义数", "词义序号"]
        ]
        .drop_duplicates()
        .head(20)
        .to_string(index=False)
    )
else:
    multi_sense_df = pd.DataFrame()


# ============================================================
# 8. 同一词语对应多个主情感分类
# ============================================================

print("\n" + "=" * 70)
print("五、同一词语对应多个主情感分类")
print("=" * 70)

primary_emotion_count = (
    df.groupby("词语")["情感分类"]
    .nunique(dropna=True)
)

multi_primary_emotion_words = primary_emotion_count[
    primary_emotion_count > 1
].index.tolist()

print(
    f"\n存在多个主情感分类的词语数量："
    f"{len(multi_primary_emotion_words):,}"
)

if multi_primary_emotion_words:
    multi_primary_emotion_df = (
        df[df["词语"].isin(multi_primary_emotion_words)]
        .sort_values(["词语", "词义序号"])
    )

    print("\n前20个示例：")
    print(
        multi_primary_emotion_df[
            ["词语", "词义序号", "情感分类", "强度", "极性"]
        ]
        .drop_duplicates()
        .head(20)
        .to_string(index=False)
    )
else:
    multi_primary_emotion_df = pd.DataFrame()


# ============================================================
# 9. 同一词语对应多个主强度
# ============================================================

print("\n" + "=" * 70)
print("六、同一词语对应多个主情感强度")
print("=" * 70)

primary_intensity_count = (
    df.groupby("词语")["强度"]
    .nunique(dropna=True)
)

multi_primary_intensity_words = primary_intensity_count[
    primary_intensity_count > 1
].index.tolist()

print(
    f"\n存在多个主情感强度的词语数量："
    f"{len(multi_primary_intensity_words):,}"
)

if multi_primary_intensity_words:
    multi_primary_intensity_df = (
        df[df["词语"].isin(multi_primary_intensity_words)]
        .sort_values(["词语", "词义序号"])
    )

    print("\n前20个示例：")
    print(
        multi_primary_intensity_df[
            ["词语", "词义序号", "情感分类", "强度"]
        ]
        .drop_duplicates()
        .head(20)
        .to_string(index=False)
    )
else:
    multi_primary_intensity_df = pd.DataFrame()


# ============================================================
# 10. 同一词语对应多个主极性
# ============================================================

print("\n" + "=" * 70)
print("七、同一词语对应多个主极性")
print("=" * 70)

primary_polarity_count = (
    df.groupby("词语")["极性"]
    .nunique(dropna=True)
)

multi_primary_polarity_words = primary_polarity_count[
    primary_polarity_count > 1
].index.tolist()

print(
    f"\n存在多个主极性的词语数量："
    f"{len(multi_primary_polarity_words):,}"
)

if multi_primary_polarity_words:
    multi_primary_polarity_df = (
        df[df["词语"].isin(multi_primary_polarity_words)]
        .sort_values(["词语", "词义序号"])
    )

    print("\n前20个示例：")
    print(
        multi_primary_polarity_df[
            ["词语", "词义序号", "情感分类", "强度", "极性"]
        ]
        .drop_duplicates()
        .head(20)
        .to_string(index=False)
    )
else:
    multi_primary_polarity_df = pd.DataFrame()


# ============================================================
# 11. 辅助情感分类检查
# ============================================================

print("\n" + "=" * 70)
print("八、辅助情感分类检查")
print("=" * 70)

secondary_emotion_count = (
    df.groupby("词语")["辅助情感分类"]
    .nunique(dropna=True)
)

multi_secondary_emotion_words = secondary_emotion_count[
    secondary_emotion_count > 1
].index.tolist()

print(
    f"\n存在多个不同辅助情感分类的词语数量："
    f"{len(multi_secondary_emotion_words):,}"
)

# 有辅助情感的词
has_secondary = df[
    df["辅助情感分类"].notna()
    & (
        df["辅助情感分类"]
        .astype(str)
        .str.strip()
        != ""
    )
]

print(
    f"具有辅助情感分类的词条数："
    f"{len(has_secondary):,}"
)

if not has_secondary.empty:
    print("\n前20个辅助情感示例：")
    print(
        has_secondary[
            [
                "词语",
                "词义序号",
                "情感分类",
                "强度",
                "极性",
                "辅助情感分类",
                "强度.1",
                "极性.1",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 12. 检查主情感字段是否合法
# ============================================================

print("\n" + "=" * 70)
print("九、主情感字段取值检查")
print("=" * 70)

emotion_values = (
    df["情感分类"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

print("\n主情感分类：")
print(sorted(emotion_values))


# ============================================================
# 13. 检查强度取值
# ============================================================

print("\n" + "=" * 70)
print("十、主强度取值检查")
print("=" * 70)

intensity_values = (
    df["强度"]
    .dropna()
    .unique()
)

print("\n主强度取值：")
print(sorted(intensity_values))


# ============================================================
# 14. 检查极性取值
# ============================================================

print("\n" + "=" * 70)
print("十一、主极性取值检查")
print("=" * 70)

polarity_values = (
    df["极性"]
    .dropna()
    .unique()
)

print("\n主极性取值：")
print(sorted(polarity_values))


# ============================================================
# 15. 输出详细 Excel
# ============================================================

print("\n" + "=" * 70)
print("十二、输出检查结果")
print("=" * 70)

output_file = OUTPUT_DIR / "DUTIR_check_results.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    # 基本统计
    summary = pd.DataFrame({
        "项目": [
            "总行数",
            "唯一词语数",
            "涉及重复词语的行数",
            "多个词义词语数",
            "多个主情感词语数",
            "多个主强度词语数",
            "多个主极性词语数",
            "具有辅助情感词条数",
            "多个辅助情感词语数",
        ],
        "数量": [
            len(df),
            unique_words,
            duplicate_word_rows,
            len(multi_sense_words),
            len(multi_primary_emotion_words),
            len(multi_primary_intensity_words),
            len(multi_primary_polarity_words),
            len(has_secondary),
            len(multi_secondary_emotion_words),
        ]
    })

    summary.to_excel(
        writer,
        sheet_name="summary",
        index=False
    )

    # 多词义
    if not multi_sense_df.empty:
        multi_sense_df.to_excel(
            writer,
            sheet_name="多词义",
            index=False
        )

    # 多主情感
    if not multi_primary_emotion_df.empty:
        multi_primary_emotion_df.to_excel(
            writer,
            sheet_name="多主情感",
            index=False
        )

    # 多强度
    if not multi_primary_intensity_df.empty:
        multi_primary_intensity_df.to_excel(
            writer,
            sheet_name="多主强度",
            index=False
        )

    # 多极性
    if not multi_primary_polarity_df.empty:
        multi_primary_polarity_df.to_excel(
            writer,
            sheet_name="多主极性",
            index=False
        )

    # 辅助情感
    if not has_secondary.empty:
        has_secondary.to_excel(
            writer,
            sheet_name="辅助情感",
            index=False
        )


print(f"\n✓ 检查结果已保存：")
print(output_file)


# ============================================================
# 16. 最终总结
# ============================================================

print("\n" + "=" * 70)
print("检查完成")
print("=" * 70)

print(f"""
DUTIR 总行数：                  {len(df):,}
唯一词语数：                    {unique_words:,}
多个词义的词语：                {len(multi_sense_words):,}
多个主情感分类的词语：          {len(multi_primary_emotion_words):,}
多个主强度的词语：              {len(multi_primary_intensity_words):,}
多个主极性的词语：              {len(multi_primary_polarity_words):,}
具有辅助情感的词条：            {len(has_secondary):,}
多个辅助情感分类的词语：        {len(multi_secondary_emotion_words):,}
""")

print("下一步请重点查看：")
print("  1. dutir_check/DUTIR_check_results.xlsx")
print("  2. 多词义")
print("  3. 多主情感")
print("  4. 多主强度")
print("  5. 多主极性")

print("\n暂时不要进行情感分数计算。")