from pathlib import Path
import pandas as pd


# 1. 文件路径
INPUT_FILE = Path("../../ABSA/absa_overall/absa_full_output/ABSA_Qwen3_最终全量结果.xlsx")
OUTPUT_FILE = Path("absa_validation_sample_40.csv")

# 2. 参数
N_PER_STRATUM = 5
RANDOM_SEED = 42

# 3. 读取数据
df = pd.read_excel(INPUT_FILE)
print(f"原始数据量：{len(df)}")

# 4. 检查分层变量
# =========================
print("\n投稿种别：")
print(df["语料来源"].value_counts())

print("\n书籍：")
print(df["book"].value_counts())

# 5. 按「书籍 × 投稿种别」分层抽样
sample_df = (
    df.groupby(["book", "语料来源"], group_keys=False)
      .apply(
          lambda x: x.sample(
              n=N_PER_STRATUM,
              random_state=RANDOM_SEED
          )
      )
      .reset_index(drop=True)
)

# 6. 随机打乱顺序
# =========================
sample_df = sample_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# 7. 保存
sample_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# 8. 检查结果
# =========================
print(f"\n抽样结果：{len(sample_df)} 条")

print("\n各层样本数：")
print(
    sample_df
    .groupby(["book", "语料来源"])
    .size()
)

print(f"\n已保存：{OUTPUT_FILE}")