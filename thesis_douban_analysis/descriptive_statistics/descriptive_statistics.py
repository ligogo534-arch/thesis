import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

#时间分布

OUTPUT_TABLE = "tables"
OUTPUT_FIGURE = "figures"

os.makedirs(OUTPUT_TABLE, exist_ok=True)
os.makedirs(OUTPUT_FIGURE, exist_ok=True)

'''
FILES = {
    "短評": ("../data_cleaning2/text_cleaning/duanping_clean_base.csv","time"),
    "書評": ("../data_cleaning2/text_cleaning/shuping_clean_base.csv", "time"),
    "読書ノート": ( "../data/note_4books.csv", "time"),
    "原文抜粋": ("../data_cleaning1/data_cleaned/zhailu_clean.csv", "时间")
}

def yearly_count(path, time_col):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df[time_col] = pd.to_datetime(df[time_col],errors="coerce")
    df["year"] = df[time_col].dt.year
    return df.groupby("year").size()

result = pd.DataFrame()
for name, (path, time_col) in FILES.items():
    result[name] = yearly_count(path, time_col)

result = (result.fillna(0).astype(int).sort_index())

result["合計"] = result.sum(axis=1)
table_path = os.path.join(OUTPUT_TABLE,"table_time_distribution.csv")

result.to_csv(table_path,encoding="utf-8-sig")
print(result)
plt.rcParams["font.family"] = "Yu Gothic"
plt.figure(figsize=(9,5))

for column in ["短評", "書評","読書ノート","原文抜粋"]:
    plt.plot( result.index,result[column], marker="o",linewidth=2,label=column)

plt.xlabel("年")
plt.ylabel("投稿数")

positions = result.index
labels = [str(y) for y in result.index]
# 最后一年改为"2025\n(~8月)"
labels[-1] = "2025\n(~8月)"
plt.xticks(positions,labels,rotation=45)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
figure_path = os.path.join(OUTPUT_FIGURE,"figure_time_distribution.png")

plt.savefig(figure_path, dpi=300,bbox_inches="tight")
plt.show()
print(f"\nCSV已保存：{table_path}")
print(f"图片已保存：{figure_path}")

#最早和最晚投稿时间
print("=" * 70)
print("各数据集投稿时间范围")
print("=" * 70)

for name, (path, time_col) in FILES.items():
    df = pd.read_csv(path, encoding="utf-8-sig")
    dt = pd.to_datetime(df[time_col], errors="coerce")
    print(f"\n{name}")
    print(f"记录数：{len(df)}")
    print(f"最早投稿：{dt.min()}")
    print(f"最晚投稿：{dt.max()}")
    print(f"无法解析日期：{dt.isna().sum()} 条")
print("\n" + "=" * 70)


#扩大图

zoom = result.loc[2015:2019]
plt.figure(figsize=(8, 5))

for column in ["短評", "書評", "読書ノート", "原文抜粋"]:
    plt.plot(zoom.index, zoom[column],marker="o",linewidth=2,label=column)

plt.xlabel("年")
plt.ylabel("投稿数")
plt.xticks(zoom.index)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
figure_path2 = os.path.join(OUTPUT_FIGURE,"figure_time_distribution_zoom.png")

plt.savefig(figure_path2,dpi=300,bbox_inches="tight")
plt.show()
print(f"局部放大图已保存：{figure_path2}")
'''

#检查可用字体
'''
from matplotlib import font_manager
fonts = sorted(set(f.name for f in font_manager.fontManager.ttflist))

for f in fonts:
    if any(x in f for x in ["Yu", "Meiryo", "MS", "Noto", "IPA"]):
        print(f)
'''

#文本长度统计
'''
FILES = {
    "短評": ("../data_cleaning2/text_cleaning/duanping_clean_base.csv","content"),
    "書評": ("../data_cleaning2/text_cleaning/shuping_clean_base.csv", "content"),
    "原文抜粋": ("../data_cleaning1/data_cleaned/zhailu_clean.csv", "文本内容")
}

datasets = {}
summary = []
for post_type, (path, text_col) in FILES.items():
    df = pd.read_csv(path)
    # 计算字符数
    df["text_length"] = (df[text_col].fillna("").astype(str).str.len())
    datasets[post_type] = df
    summary.append({
        "投稿タイプ": post_type,
        "件数": len(df),
        "平均": df["text_length"].mean(),
        "中央値": df["text_length"].median(),
        "標準偏差": df["text_length"].std(),
        "最小値": df["text_length"].min(),
        "第1四分位数": df["text_length"].quantile(0.25),
        "第3四分位数": df["text_length"].quantile(0.75),
        "最大値": df["text_length"].max()
    })
'''
'''
summary = pd.DataFrame(summary).round(1)
print(summary)
table_path = os.path.join(OUTPUT_TABLE,"text_length_summary.csv")
summary.to_csv(table_path,index=False, encoding="utf-8-sig")

#直方图
plt.rcParams["font.family"] = "Yu Gothic"
plt.figure(figsize=(9,6))

for post_type, df in (datasets.items()):
    plt.hist(
        df["text_length"],
        bins=50,
        alpha=0.5,
        label=post_type
    )

plt.xlabel("テキスト長（文字数）", fontsize=12)
plt.ylabel("投稿数", fontsize=12)
plt.legend(title="投稿タイプ")
plt.tight_layout()

figure_path = os.path.join(OUTPUT_FIGURE,"text_length_histogram.png")
plt.savefig(figure_path,dpi=300,bbox_inches="tight")
plt.show()
'''
#箱线图
'''
plt.rcParams["font.family"] = "Yu Gothic"
plt.figure(figsize=(6,6))
plt.boxplot(
    [datasets[name]["text_length"] for name in datasets],
    tick_labels=list(datasets.keys()),
    showfliers=False
)

plt.ylabel("文字数", fontsize=12)
plt.tight_layout()

figure_path = os.path.join(OUTPUT_FIGURE,"text_length_boxplot.png")
plt.savefig(figure_path,dpi=300,bbox_inches="tight")
plt.show()
'''

#检查文本最大值
'''
for name, df in datasets.items():
    print(
        name,
        "最大:", df["text_length"].max(),
        "95%:", df["text_length"].quantile(0.95),
        "99%:", df["text_length"].quantile(0.99)
    )
'''

#检查书评文本最大值
'''
datasets["書評"].sort_values("text_length", ascending=False).head(10).to_csv("others/longest_shuping.csv",
        index=False, encoding="utf-8-sig")
datasets["原文抜粋"].sort_values("text_length", ascending=False).head(10).to_csv("others/longest_zhailu.csv",
        index=False, encoding="utf-8-sig")


for name, df in datasets.items():
    q1 = df["text_length"].quantile(0.25)
    q2 = df["text_length"].median()
    q3 = df["text_length"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr

    print(f"\n{name}")
    print(f"Q1     : {q1:.1f}")
    print(f"Median : {q2:.1f}")
    print(f"Q3     : {q3:.1f}")
    print(f"IQR    : {iqr:.1f}")
    print(f"Upper  : {upper:.1f}")
    print(f"Max    : {df['text_length'].max()}")
'''

#按评分分类的描述统计（短评、长评）
'''
FILES = {
    "短評": (
        "../data_cleaning2/text_cleaning/duanping_clean_base.csv",
        "content",
        "rating",
        "user_name"
    ),
    "書評": (
        "../data_cleaning2/text_cleaning/shuping_clean_base.csv",
        "content",
        "rating",
        "user_name"
    )
}

summary = []
for post_type, (path, text_col, rating_col, user_col) in FILES.items():
    df = pd.read_csv(path)
    # 文字数
    df["文字数"] = (df[text_col].fillna("").astype(str).str.len())

    # 去除缺失评分
    df = df[df[rating_col].notna()].copy()
    total_posts = len(df)

    grouped = (df.groupby(rating_col).agg(
            件数=(rating_col, "size"),
            ユーザー数=(user_col, "nunique"),
            平均文字数=("文字数", "mean")
        ).reset_index()
    )

    grouped["構成比(%)"] = (grouped["件数"] / total_posts * 100)
    grouped.insert(0, "投稿タイプ", post_type)
    grouped = grouped.rename(columns={rating_col: "評価"})
    summary.append(grouped)

summary = pd.concat(summary, ignore_index=True)

summary["平均文字数"] = summary["平均文字数"].round(1)
summary["構成比(%)"] = summary["構成比(%)"].round(1)

summary = summary[["投稿タイプ", "評価","件数","構成比(%)","ユーザー数", "平均文字数"]]
print(summary)
table_path = os.path.join(OUTPUT_TABLE,"rating_summary.csv")
summary.to_csv(table_path,index=False,encoding="utf-8-sig")
'''

#统计多少评论有评分，多少没有评分
'''
FILES = {
    "短評": (
        "../data_cleaning2/text_cleaning/duanping_clean_base.csv",
        "rating"
    ),
    "書評": (
        "../data_cleaning2/text_cleaning/shuping_clean_base.csv",
        "rating"
    )
}

summary = []
for post_type, (path, rating_col) in FILES.items():
    df = pd.read_csv(path)
    total = len(df)
    rated = df[rating_col].notna().sum()
    unrated = df[rating_col].isna().sum()

    summary.append({
        "投稿タイプ": post_type,
        "総件数": total,
        "評価あり": rated,
        "評価あり割合(%)": rated / total * 100,
        "評価なし": unrated,
        "評価なし割合(%)": unrated / total * 100
    })

summary = pd.DataFrame(summary)
summary[["評価あり割合(%)", "評価なし割合(%)"]] = summary[["評価あり割合(%)", "評価なし割合(%)"]].round(1)
print(summary)
table_path = os.path.join(OUTPUT_TABLE,"rating_availability_summary.csv")
summary.to_csv(table_path,index=False,encoding="utf-8-sig")
'''

#按书名统计有无评分/星级占比
FILES = {
    "短評": (
        "../data_cleaning2/text_cleaning/duanping_clean_base.csv",
        "rating"
    ),
    "書評": (
        "../data_cleaning2/text_cleaning/shuping_clean_base.csv",
        "rating"
    )
}

results = []
for data_type, (file_path, rating_col) in FILES.items():
    print("処理中:", data_type)
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    for book, group in df.groupby("book_id"):
        total = len(group)
        # 有评分数据
        rating_df = group[group[rating_col].notna()]
        rating_count = len(rating_df)
        no_rating_count = (total - rating_count)

        rating_pct = (rating_count / total * 100 if total > 0 else 0)
        no_rating_pct = (no_rating_count / total * 100 if total > 0 else 0)
        mean_rating = (rating_df[rating_col].mean() if rating_count > 0 else None)
        std_rating = (rating_df[rating_col].std() if rating_count > 1 else None)

        star_values = {}
        for star in range(1, 6):
            star_count = (rating_df[rating_col].eq(star).sum())
            star_pct = (star_count / rating_count * 100  if rating_count > 0  else 0 )
            star_values[f"★{star}(%)"] = round(star_pct,2)

        results.append({
            "書籍名": book,
            "データ種別": data_type,
            "総評論数": total,
            "評価あり件数（%）":
                f"{rating_count} ({rating_pct:.2f})",
            "評価なし件数（%）":
                f"{no_rating_count} ({no_rating_pct:.2f})",
            "平均評価":
                round(mean_rating, 2)
                if pd.notna(mean_rating)
                else None,
            "標準偏差":
                round(std_rating, 2)
                if pd.notna(std_rating)
                else None,
            **star_values
        })


result_df = pd.DataFrame(results)
result_df = result_df[
    [
        "書籍名",
        "データ種別",
        "総評論数",
        "評価あり件数（%）",
        "評価なし件数（%）",
        "平均評価",
        "標準偏差",
        "★1(%)",
        "★2(%)",
        "★3(%)",
        "★4(%)",
        "★5(%)"
    ]
]

result_df = result_df.sort_values(["データ種別","書籍名"])
table_path = os.path.join(OUTPUT_TABLE,"作品別評価傾向.csv")
result_df.to_csv(table_path, index=False,encoding="utf-8-sig")

print("\n===== 完成 =====")
print(result_df)
print("\n保存:", table_path)