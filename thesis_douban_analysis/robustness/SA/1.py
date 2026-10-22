import pandas as pd

old = pd.read_csv("sa_sample_40.csv", encoding="utf-8-sig")
new = pd.read_csv("sa_sample_40_new.csv", encoding="utf-8-sig")

old_ids = set(old["sample_id"])
new_ids = set(new["sample_id"])

overlap = old_ids & new_ids

print(f"第一次样本数：{len(old_ids)}")
print(f"第二次样本数：{len(new_ids)}")
print(f"重复样本数：{len(overlap)}")

if len(overlap) == 0:
    print("✅ 两次抽样完全没有重复样本")
else:
    print("⚠️ 两次抽样存在重复样本：")
    print(sorted(overlap))