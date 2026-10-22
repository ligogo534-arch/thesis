import pandas as pd
#duanping中的32行重复行
'''
df = pd.read_csv("../data/duanping_4books.csv")
dup_df = df[df.duplicated(keep=False)]
dup_df = dup_df.sort_values(by=list(df.columns))

print(f"原始数据行数: {len(df)}")
print(f"重复行数: {len(dup_df)}")
print(f"重复组数: {dup_df.drop_duplicates().shape[0]}")

dup_df.to_csv("duanping_duplicate_rows.csv", index=False, encoding="utf-8-sig")
print("重复行已保存到：duanping_duplicate_rows.csv")
'''
#导出带原始index的重复行
'''
df = pd.read_csv("../data/duanping_4books.csv")
dup_df = (
    df[df.duplicated(keep=False)]
    .sort_values(list(df.columns))
    .reset_index(drop=False)   # 保留原来的行号
)
print(dup_df)
dup_df.to_csv("duanping_duplicate_rows_with_index.csv",index=False,encoding="utf-8-sig")
'''
#分析重复产生原因
'''
df = pd.read_csv("../data/duanping_4books.csv")
# 保留原始行号
df = df.reset_index().rename(columns={"index": "原始行号"})
dup_groups = []
group_id = 1
# 所有列（除了原始行号）
cols = [c for c in df.columns if c != "原始行号"]
for _, group in df.groupby(cols, dropna=False):
    if len(group) > 1:
        indexes = sorted(group["原始行号"].tolist())
        dup_groups.append({
            "重复组": group_id,
            "重复次数": len(group),
            "index1": indexes[0],
            "index2": indexes[1] if len(indexes) > 1 else None,
            "distance": indexes[1] - indexes[0] if len(indexes) > 1 else None,
            "是否相邻": "是" if len(indexes) > 1 and indexes[1]-indexes[0]==1 else "否",
            "书名": group.iloc[0]["book_id"],
            "用户名": group.iloc[0]["user_name"],
            "时间": group.iloc[0]["time"],
        })

        group_id += 1
report = pd.DataFrame(dup_groups)
report.to_csv(
    "duanping_duplicate_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)
print(report)

print("="*60)
print(f"重复组数：{len(report)}")
print(f"总重复记录数：{report['重复次数'].sum()}")
print("\n===== 是否相邻 =====")
print(report["是否相邻"].value_counts())
print("\n===== 相距行数统计 =====")
print(report["distance"].describe())
print("\n===== 每本书重复组数 =====")
print(report["book_id"].value_counts())
print("\n===== 距离最大的10组 =====")
print(report.sort_values("distance", ascending=False).head(10))
'''
#看duanping合并前两个文件内是否有重复值
'''
files = ["../../ueno/douban/crawler/duanping/duanping_3books.csv","../../ueno/douban/crawler/duanping/duanping_yannv.csv"]
for f in files:
    df = pd.read_csv(f)
    print("=" * 50)
    print(f)
    print("行数：", len(df))
    print("完全重复：", df.duplicated().sum())
'''
#duanping去除重复行
'''
input_file = "../data/duanping_4books.csv"
output_file = "duanping_dedup.csv"

df = pd.read_csv(input_file)
original_rows = len(df)
df_clean = df.drop_duplicates()
removed_rows = original_rows - len(df_clean)
clean_rows = len(df_clean)

print("=" * 50)
print("重复行删除结果")
print("=" * 50)
print(f"原始数据行数: {original_rows}")
print(f"删除重复行数: {removed_rows}")
print(f"处理后数据行数: {clean_rows}")
print(f"数据减少比例: {removed_rows / original_rows:.2%}")

df_clean.to_csv(output_file,index=False,encoding="utf-8-sig")
print("=" * 50)
print(f"清洗后文件已保存: {output_file}")
'''

#检查重复text
'''
df = pd.read_csv("duanping_dedup.csv")
df = df.reset_index().rename(columns={"index": "原始行号"})
text_dup = df[df["content"].notna() & df["content"].duplicated(keep=False)]
text_count = (text_dup.groupby("content").size().reset_index(name="出现次数").sort_values("出现次数", ascending=False))
print("="*50)
print("重复文本统计")
print("="*50)
print(f"重复文本种类数: {len(text_count)}")
print(f"涉及记录数: {len(text_dup)}")
print("\n出现次数最高的10个重复文本:")
print(text_count.head(10))
text_dup = text_dup.merge(text_count,on="content",how="left")
text_dup.to_csv("duanping_duplicate_text.csv",index=False,encoding="utf-8-sig")
print("\n重复文本详细记录已保存:")
print("duanping_duplicate_text.csv")
'''

#检查缺失值（用户名和文本各2）
'''
df = pd.read_csv("duanping_dedup.csv")
df_index = df.reset_index()
print("=" * 50)
print("缺失值统计")
print("=" * 50)
print(df[["user_name", "content"]].isna().sum())
missing_rows = df[df["user_name"].isna() |df["content"].isna()]
print("\n" + "=" * 50)
print("含缺失值的行")
print("=" * 50)
print(missing_rows)
missing_rows.to_csv("duanping_missing.csv",index=False,encoding="utf-8-sig")
print("\n缺失值记录已保存：duanping_missing_user_content.csv")
print(f"缺失记录总数：{len(missing_rows)}")
'''

#duanping删除没有文本内容的2行
input_file = "duanping_dedup.csv"  # 已去除重复行的数据
output_file = "../data_cleaned/duanping_clean.csv"  # 最终清洗数据

df = pd.read_csv(input_file)
original_rows = len(df)
missing_content = df["content"].isna()
deleted_rows = missing_content.sum()
df_clean = df[~missing_content].copy()

print("=" * 50)
print("缺失文本删除结果")
print("=" * 50)
print(f"处理前行数: {original_rows}")
print(f"删除缺失content行数: {deleted_rows}")
print(f"处理后行数: {len(df_clean)}")
print(f"数据减少比例: {deleted_rows / original_rows:.2%}")

df_clean.to_csv(output_file,index=False,encoding="utf-8-sig")
print("=" * 50)
print(f"清洗后文件已保存: {output_file}")