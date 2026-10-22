import pandas as pd

#检查重复text行
'''
input_file = "../../data/zhailu6.csv"
output_file = "zhailu_duplicates.csv"

df = pd.read_csv(input_file)
dup_df = df[df.duplicated(subset=["文本内容"], keep=False)].copy()

group_count = (dup_df.groupby("文本内容").size().reset_index(name="重复次数"))
dup_df = dup_df.merge(group_count, on="文本内容")

dup_df = dup_df.sort_values(["重复次数", "文本内容"],ascending=[False, True])

print("=" * 60)
print(f"总记录数：{len(df)}")
print(f"重复正文组数：{group_count.shape[0]}")
print(f"涉及记录数：{len(dup_df)}")
print("=" * 60)

print("\n重复次数最多的前20组：")
print(group_count.sort_values("重复次数", ascending=False).head(20))

dup_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n已导出：{output_file}")
'''

#检查重复行产生原因
'''
df = pd.read_csv("../../data/zhailu6.csv")
def check_duplicate(df, subset, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    dup = df[df.duplicated(subset=subset, keep=False)].copy()
    if dup.empty:
        print("没有发现重复。")
        return
    group_count = (
        dup.groupby(subset)
           .size()
           .reset_index(name="重复次数")
           .sort_values("重复次数", ascending=False)
    )

    print(f"重复组数：{len(group_count)}")
    print(f"涉及记录数：{len(dup)}")

    print("\n重复次数最多的前20组：")
    print(group_count.head(20))

    dup = dup.merge(group_count, on=subset)
    dup = dup.sort_values(
        by=["重复次数"] + subset,
        ascending=[False] + [True] * len(subset)
    )

    print("\n前50条重复记录：")
    print(dup[["书名","用户名","时间","点赞数","回复数","重复次数","文本内容",]].head(50))

# 1. 同一用户 + 相同正文
check_duplicate(df,["用户名", "文本内容"],"① 同一用户 + 相同正文")
# 2. 同一本书 + 相同正文
check_duplicate(df,["书名", "文本内容"],"② 同一本书 + 相同正文")
# 3. 同一本书 + 同一用户 + 相同正文
check_duplicate(df,["书名", "用户名", "文本内容"],"③ 同一本书 + 同一用户 + 相同正文")
'''

#删除zhailu中同一本书 + 同一用户 + 相同正文的数据行（1行）
'''
input_file = "../../data/zhailu6.csv"
output_file = "../data_cleaned/zhailu_clean.csv"
df = pd.read_csv(input_file)
print(f"原始记录数：{len(df)}")

mask = df.duplicated(subset=["书名", "用户名", "文本内容"],keep="first")

deleted = df.loc[mask].copy()
print(f"将删除的重复记录数：{len(deleted)}")
if len(deleted) > 0:
    print("\n将删除的记录：")
    print(
        deleted[
            [
                "书名",
                "用户名",
                "时间",
                "点赞数",
                "回复数",
                "文本内容",
            ]
        ]
    )

df_clean = df.drop_duplicates(subset=["书名", "用户名", "文本内容"],keep="first")
print(f"\n删除后记录数：{len(df_clean)}")
df_clean.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n已保存：{output_file}")
'''

#检查缺失值
input_file = "../data_cleaned/zhailu_clean.csv"
df = pd.read_csv(input_file)
# ① 无figure、无JSON、无citation
mask1 = (df["figure数量"].isna() &df["正文结构JSON"].isna() &df["引用出处"].isna())
print("①", mask1.sum())
# ② 有figure、无JSON、有citation
mask2 = (
    df["figure数量"].notna() &df["正文结构JSON"].isna() & df["引用出处"].notna())
print("②", mask2.sum())
# ③ 有figure、有JSON、无citation
mask3 = (df["figure数量"].notna() & df["正文结构JSON"].notna() & df["引用出处"].isna())
print("③", mask3.sum())
# ④ 全部存在
mask4 = (df["figure数量"].notna() & df["正文结构JSON"].notna() & df["引用出处"].notna())
print("④", mask4.sum())
