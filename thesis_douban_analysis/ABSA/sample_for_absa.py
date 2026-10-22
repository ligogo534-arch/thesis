from pathlib import Path
import pandas as pd


# ============================================================
# 1. 路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SHORT_FILE = (BASE_DIR/ "data_cleaning2"/ "text_cleaning"/ "duanping_clean_base.csv")
LONG_FILE = (BASE_DIR/ "data_cleaning2"/ "text_cleaning"/ "shuping_clean_base.csv")

OUTPUT_DIR = BASE_DIR / "ABSA"/"absa_sampling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. 抽样参数
# 每个「书本 × 评论类型」再抽15条
N_PER_STRATUM = 15
RANDOM_STATE = 123

# 3. 读取原始数据
short_df = pd.read_csv(
    SHORT_FILE,
    encoding="utf-8-sig"
)

long_df = pd.read_csv(
    LONG_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# 4. 添加评论类型
# ============================================================

short_df["评论类型"] = "短评"
long_df["评论类型"] = "书评"


# ============================================================
# 5. 利用长评建立 book_id → book_name
# ============================================================

book_name_map = (
    long_df[
        ["book_id", "book_name"]
    ]
    .drop_duplicates("book_id")
    .set_index("book_id")["book_name"]
    .to_dict()
)

short_df["book_name"] = short_df["book_id"].map(
    book_name_map
)


# ============================================================
# 6. 合并
# ============================================================

df = pd.concat(
    [short_df, long_df],
    ignore_index=True,
    sort=False
)


# ============================================================
# 7. 清理必要字段
# ============================================================

df = df.dropna(
    subset=["book_id", "content"]
).copy()

df["book_id"] = (
    df["book_id"]
    .astype(str)
    .str.strip()
)

df["content"] = (
    df["content"]
    .astype(str)
    .str.strip()
)

df["book_name"] = (
    df["book_name"]
    .fillna("")
    .astype(str)
    .str.strip()
)


df = df[
    (df["book_id"] != "") &
    (df["content"] != "")
].copy()


# ============================================================
# 8. 创建唯一评论ID
# ============================================================

# 书评优先使用 id
# 短评没有 id，则使用 book_id + user_name + time + content

df["唯一ID"] = ""


# ---------- 书评 ----------

long_mask = df["评论类型"] == "书评"

df.loc[long_mask, "唯一ID"] = (
    "L_"
    + df.loc[long_mask, "id"]
    .fillna("")
    .astype(str)
)


# ---------- 短评 ----------

short_mask = df["评论类型"] == "短评"

df.loc[short_mask, "唯一ID"] = (
    "S_"
    + df.loc[short_mask, "book_id"].astype(str)
    + "_"
    + df.loc[short_mask, "user_name"]
    .fillna("")
    .astype(str)
    + "_"
    + df.loc[short_mask, "time"]
    .fillna("")
    .astype(str)
    + "_"
    + df.loc[short_mask, "content"]
    .astype(str)
)


# ============================================================
# 9. 读取已经人工看过的160条
# ============================================================

OLD_SAMPLE_FILES = (OUTPUT_DIR / "ABSA_分层抽样40条.xlsx", OUTPUT_DIR / "ABSA_分层抽样新增120条.xlsx")

old_dfs = []
for file in OLD_SAMPLE_FILES:
    if not file.exists():
        raise FileNotFoundError(f"找不到之前的抽样文件：{file}")
    old_df = pd.read_excel(file,sheet_name="人工阅读")
    old_dfs.append(old_df)

old_df = pd.concat(old_dfs,ignore_index=True)
print(f"\n之前已经人工阅读：{len(old_df)} 条")

# ============================================================
# 10. 建立旧样本的唯一ID
# ============================================================

old_df["唯一ID"] = ""

# ---------- 原来的书评 ----------
old_long_mask = old_df["评论类型"] == "书评"

old_df.loc[old_long_mask, "唯一ID"] = (
    "L_"
    + old_df.loc[old_long_mask, "id"]
    .fillna("")
    .astype(str)
)


# ---------- 原来的短评 ----------

old_short_mask = old_df["评论类型"] == "短评"

old_df.loc[old_short_mask, "唯一ID"] = (
    "S_"
    + old_df.loc[old_short_mask, "book_id"]
    .astype(str)
    + "_"
    + old_df.loc[old_short_mask, "user_name"]
    .fillna("")
    .astype(str)
    + "_"
    + old_df.loc[old_short_mask, "time"]
    .fillna("")
    .astype(str)
    + "_"
    + old_df.loc[old_short_mask, "content"]
    .astype(str)
)

old_ids = set(old_df["唯一ID"])
print("已识别旧样本：", len(old_ids), "条")


# ============================================================
# 11. 排除已经抽过的40条
# ============================================================
before_n = len(df)

df_new = df[
    ~df["唯一ID"].isin(old_ids)
].copy()

after_n = len(df_new)

print("\n原始数据：", before_n)
print("排除旧样本后：", after_n)
print("排除数量：", before_n - after_n)


# ============================================================
# 12. 查看剩余各层数量
# ============================================================

remaining_summary = (
    df_new
    .groupby(
        ["book_id", "book_name", "评论类型"],
        dropna=False
    )
    .size()
    .reset_index(name="剩余数量")
)

print("\n========== 排除旧样本后的各层数量 ==========")

print(
    remaining_summary.to_string(index=False)
)


# ============================================================
# 13. 分层抽取新的120条
# ============================================================

sampled_parts = []

for (
    book_id,
    book_name,
    review_type
), group in df_new.groupby(
    ["book_id", "book_name", "评论类型"],
    dropna=False,
    sort=True
):

    available_n = len(group)

    sample_n = min(
        N_PER_STRATUM,
        available_n
    )

    sampled = group.sample(
        n=sample_n,
        replace=False,
        random_state=RANDOM_STATE
    ).copy()

    sampled["该层剩余数量"] = available_n
    sampled["本次抽样数量"] = sample_n

    sampled_parts.append(sampled)


# ============================================================
# 14. 合并
# ============================================================

sampled_df = pd.concat(
    sampled_parts,
    ignore_index=True
)


# ============================================================
# 15. 添加人工编码字段
# ============================================================

sampled_df["Aspect"] = ""
sampled_df["Sentiment"] = ""
sampled_df["Aspect_备注"] = ""


# ============================================================
# 16. 样本编号
# ============================================================

sampled_df.insert(
    0,
    "样本ID",
    range(41, 41 + len(sampled_df))
)


# ============================================================
# 17. 排序
# ============================================================

sampled_df = sampled_df.sort_values(
    [
        "book_id",
        "评论类型",
        "样本ID"
    ]
).reset_index(drop=True)


# ============================================================
# 18. 输出
# ============================================================

OUTPUT_FILE = (
    OUTPUT_DIR / "ABSA_分层抽样第三批120条.xlsx"
)

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    sampled_df.to_excel(
        writer,
        sheet_name="人工阅读",
        index=False
    )

    summary = (
        sampled_df
        .groupby(
            ["book_id", "book_name", "评论类型"],
            dropna=False
        )
        .size()
        .reset_index(name="本次抽样数量")
    )

    summary.to_excel(
        writer,
        sheet_name="抽样统计",
        index=False
    )


# ============================================================
# 19. 输出检查
# ============================================================

print("\n========================================")
print("新的分层抽样完成")
print("========================================")

print(
    f"本次抽样：{len(sampled_df)} 条"
)

print(
    "\n各层抽样数量："
)

print(
    sampled_df
    .groupby(
        ["book_id", "book_name", "评论类型"],
        dropna=False
    )
    .size()
    .to_string()
)

print(
    f"\n输出文件：{OUTPUT_FILE}"
)