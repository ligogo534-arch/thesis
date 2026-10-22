import pandas as pd
from pathlib import Path

# 1. 参数设置
INPUT_FILE = Path("../../SA/overall_output/output_1/result.csv")
OUTPUT_FILE = Path("sa_sample_40_new.csv")

#第一次抽样40个
# RANDOM_STATE = 42
#第二次抽样新增40个
RANDOM_STATE = 2026
N_PER_STRATUM = 5

# 2. 读取数据
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

print(f"原始数据量：{len(df)}")
print(f"book_id：{df['book_id'].unique()}")
print(f"data_type：{df['data_type'].unique()}")

# 3. 检查分层组合

strata = (df.groupby(["book_id", "data_type"]).size().reset_index(name="count"))

print("\n各层样本数：")
print(strata.to_string(index=False))

# 检查是否每个层都有至少5条
insufficient = strata[strata["count"] < N_PER_STRATUM]

if len(insufficient) > 0:
    print("\n以下层不足5条，无法进行均等抽样：")
    print(insufficient.to_string(index=False))
    raise ValueError("存在样本数不足5条的分层，请检查数据。")

# 4. 分层随机抽样
sampled_df = (df.groupby(["book_id", "data_type"], group_keys=False).
              apply(lambda x: x.sample(n=N_PER_STRATUM,random_state=RANDOM_STATE)).reset_index(drop=True))

# 5. 打乱最终40条样本的顺序
sampled_df = ( sampled_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True))

# 6. 保存
sampled_df.to_csv(OUTPUT_FILE,index=False, encoding="utf-8-sig")

# 7. 检查抽样结果
print("\n最终抽样结果：")
print(f"样本数：{len(sampled_df)}")
print(f"唯一 sample_id：{sampled_df['sample_id'].nunique()}")
print("\n各书籍 × 投稿种别的抽样数量：")
check = (sampled_df.groupby(["book_id", "data_type"]).size().reset_index(name="sample_count"))
print(check.to_string(index=False))
print(f"\n已保存至：{OUTPUT_FILE}")