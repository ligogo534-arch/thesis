import pandas as pd
import numpy as np
from pathlib import Path

# 1. 文件与参数设置
INPUT_FILE = "../absa_overall/absa_full_output/ABSA_Qwen3_最终全量结果.xlsx"
OUTPUT_FILE = "absa_stability_sample_50.csv"

RANDOM_SEED = 42
SAMPLE_SIZE = 50

ID_COL = "ABSA_ID"
TYPE_COL = "语料来源"
BOOK_COL = "book"
TEXT_COL = "content"

# =========================
# 2. 读取数据
# =========================
df = pd.read_excel(INPUT_FILE)

required_cols = [ID_COL, TYPE_COL, BOOK_COL, TEXT_COL]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(
        f"文件缺少以下列：{missing_cols}\n"
        f"当前文件列名：{df.columns.tolist()}"
    )

# 去除关键字段缺失的记录
df = df.dropna(subset=[ID_COL, TYPE_COL, BOOK_COL, TEXT_COL]).copy()

# 每条评论应对应唯一ID
if df[ID_COL].duplicated().any():
    raise ValueError(
        f"{ID_COL} 存在重复值，请先确认每行是否对应一条独立评论。"
    )

# =========================
# 3. 检查分层结构
# =========================
strata = (
    df.groupby([BOOK_COL, TYPE_COL], dropna=False)
      .size()
      .reset_index(name="原始评论数")
)

print("各分层的原始评论数：")
print(strata.to_string(index=False))

n_strata = len(strata)

if n_strata != 8:
    raise ValueError(
        f"预期为8个分层（4本书×2种投稿类型），"
        f"实际检测到{n_strata}个。请检查书名和投稿类型字段。"
    )

# 每层先分配6条，剩余2条分配给样本量较大的层
base_n = SAMPLE_SIZE // n_strata  # 6
remainder = SAMPLE_SIZE % n_strata  # 2

# =========================
# 4. 分层随机抽样
# =========================
rng = np.random.default_rng(RANDOM_SEED)

# 为保证余数分配可复现，按每层原始样本量从大到小排序
strata = strata.sort_values(
    "原始评论数", ascending=False
).reset_index(drop=True)

selected_rows = []

for i, row in strata.iterrows():
    book = row[BOOK_COL]
    data_type = row[TYPE_COL]

    # 每层抽6条，原始样本量最大的两层各抽7条
    n_take = base_n + (1 if i < remainder else 0)

    group = df[
        (df[BOOK_COL] == book) &
        (df[TYPE_COL] == data_type)
    ]

    if len(group) < n_take:
        raise ValueError(
            f"分层 {book} / {data_type} 只有 {len(group)} 条，"
            f"不足以抽取 {n_take} 条。"
        )

    sampled_indices = rng.choice(
        group.index.to_numpy(),
        size=n_take,
        replace=False
    )

    selected_rows.append(df.loc[sampled_indices])

sample = pd.concat(selected_rows, ignore_index=True)

# 打乱最终样本的行顺序
sample = sample.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# 保存抽样序号，方便后续匹配两次模型输出
sample.insert(0, "stability_sample_id", range(1, len(sample) + 1))

# =========================
# 5. 输出与检查
# =========================
sample.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n抽样完成：{len(sample)} 条")
print(f"随机种子：{RANDOM_SEED}")
print(f"输出文件：{Path(OUTPUT_FILE).resolve()}")

print("\n各分层实际抽样数：")
print(
    sample.groupby([BOOK_COL, TYPE_COL])
          .size()
          .to_string()
)

print("\n样本预览：")
print(sample[[ID_COL, BOOK_COL, TYPE_COL, TEXT_COL]].head())