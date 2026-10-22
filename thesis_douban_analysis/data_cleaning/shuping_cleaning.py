import pandas as pd

df = pd.read_excel("../../data/shuping_final_2.xlsx")

#统计重复content数量
'''
df = pd.read_excel("../../data/shuping_final_2.xlsx")
text_count = (df[df["content"].notna()].groupby("content").size().reset_index(name="n"))
dup_groups = text_count[text_count["n"] > 1]
print("重复文本种类数：", len(dup_groups))
print("duplicated().sum()：", (dup_groups["n"] - 1).sum())
print("涉及记录数：", dup_groups["n"].sum())
'''
#shuping中的63个重复content
'''
df = pd.read_excel("../../data/shuping_final_2.xlsx")
df = df.reset_index().rename(columns={"index": "原始行号"})
CONTENT_COL = "content"
TIME_COL = "time"
dup_df = df[df[CONTENT_COL].notna() & df[CONTENT_COL].duplicated(keep=False)].copy()

dup_count = (
    dup_df.groupby(CONTENT_COL)
          .size()
          .reset_index(name="重复组大小")
)

dup_df = dup_df.merge(dup_count, on=CONTENT_COL, how="left")
sort_cols = ["重复组大小", CONTENT_COL]
if TIME_COL in dup_df.columns:
    sort_cols.append(TIME_COL)

dup_df = dup_df.sort_values(by=sort_cols,ascending=[False, True, True])
dup_df["Excel行号"] = dup_df["原始行号"] + 2

dup_df.to_csv("shuping_duplicate_content.csv",index=False,encoding="utf-8-sig")
print("=" * 50)
print(f"重复文本种类数：{len(dup_count)}")
print(f"涉及记录数：{len(dup_df)}")
print("重复文本详情已导出：shuping_duplicate_content_full.csv")
'''

#shuping中content-user_name-book_id都重复的行(62行)
'''
dup = df.duplicated(subset=["book_id", "user_name", "content"],keep="first")
print("将删除记录数：", dup.sum())

# 找出content重复但user不同的情况
g = (df.groupby("content")["user_name"].nunique().reset_index(name="用户数"))
print(g[g["用户数"] > 1])

#2本书下同一用户发布相同评论
g = (df.groupby(["user_name", "content"])["book_id"].nunique().reset_index(name="book_num"))
print(g[g["book_num"] > 1])
'''

#删除shuping中content-user_name-book_id都重复的行(62行)
# 保留原始顺序中的第一条记录
df_clean = df.drop_duplicates(subset=["book_id", "user_name", "content"],keep="first")

print(f"原始记录数：{len(df)}")
print(f"删除重复记录：{len(df) - len(df_clean)}")
print(f"清洗后记录数：{len(df_clean)}")

df_clean.to_csv("shuping_clean.csv",index=False,encoding="utf-8-sig")