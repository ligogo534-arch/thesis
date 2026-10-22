import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 1. 参数设置：按实际文件修改
FILE_3 = "../overall_output/output_1/result.csv"  # 否定词窗口=3
FILE_2 = "../validation_output/output_1/result2.csv"  # 否定词窗口=2

ID_COL = "sample_id"
TIME_COL = "time"
SCORE_COL = "整体极性"

df3 = pd.read_csv(FILE_3)
df2 = pd.read_csv(FILE_2)

# 检查必要列
for name, df in [("窗口=3", df3), ("窗口=2", df2)]:
    required = {ID_COL, TIME_COL, SCORE_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}缺少列：{missing}")

# =========================
# 2. 数据清理与对齐
# =========================
a = df3[[ID_COL, TIME_COL, SCORE_COL]].copy()
b = df2[[ID_COL, TIME_COL, SCORE_COL]].copy()

a = a.rename(columns={
    TIME_COL: "time_3",
    SCORE_COL: "score_3"
})
b = b.rename(columns={
    TIME_COL: "time_2",
    SCORE_COL: "score_2"
})

for df, col in [(a, "score_3"), (b, "score_2")]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

a["time_3"] = pd.to_datetime(a["time_3"], errors="coerce")
b["time_2"] = pd.to_datetime(b["time_2"], errors="coerce")

if a[ID_COL].duplicated().any() or b[ID_COL].duplicated().any():
    raise ValueError("评论ID有重复，请先确认每条评论只有一行。")

if set(a[ID_COL]) != set(b[ID_COL]):
    raise ValueError("两个文件的评论ID不一致，请先检查样本是否相同。")

comp = a.merge(b, on=ID_COL, validate="one_to_one")

# 检查时间是否一致
both_dates = comp["time_3"].notna() & comp["time_2"].notna()
if not (comp.loc[both_dates, "time_3"].dt.normalize().to_numpy()
        == comp.loc[both_dates, "time_2"].dt.normalize().to_numpy()).all():
    raise ValueError("相同评论的时间不一致，请先核查。")

comp["time"] = comp["time_3"].combine_first(comp["time_2"])
comp["year"] = comp["time"].dt.year

# 两种设定均有有效分数的评论才参与数值比较
valid = comp.dropna(subset=["score_3", "score_2"]).copy()
if valid.empty:
    raise ValueError("没有两种设定下均有有效整体极性的评论。")

valid["差值_窗口2减窗口3"] = valid["score_2"] - valid["score_3"]
valid["绝对差值"] = valid["差值_窗口2减窗口3"].abs()

# =========================
# 3. 整体分数对比
# =========================
def describe_score(s):
    return pd.Series({
        "有效样本数": s.count(),
        "均值": s.mean(),
        "标准差": s.std(ddof=1),
        "中位数": s.median(),
        "最小值": s.min(),
        "最大值": s.max()
    })

summary = pd.DataFrame({
    "窗口=3": describe_score(valid["score_3"]),
    "窗口=2": describe_score(valid["score_2"])
})

summary["窗口2减窗口3"] = (
    summary["窗口=2"] - summary["窗口=3"]
)

correlation = valid["score_3"].corr(valid["score_2"])
mae = valid["绝对差值"].mean()
mean_diff = valid["差值_窗口2减窗口3"].mean()
median_abs_diff = valid["绝对差值"].median()

# 相对变化幅度：只有原始分数不接近0时才有意义；
# 若分数可正可负或包含0，建议主要报告绝对差值。
print("\n【整体极性描述统计】")
print(summary.round(4))

print("\n【两种设定的分数一致性】")
print(f"Pearson相关系数：{correlation:.4f}")
print(f"平均差值（窗口2 - 窗口3）：{mean_diff:.4f}")
print(f"平均绝对差值（MAE）：{mae:.4f}")
print(f"绝对差值中位数：{median_abs_diff:.4f}")

# =========================
# 4. 每条评论的分数变化
# =========================
change_summary = valid["绝对差值"].describe(
    percentiles=[0.5, 0.75, 0.9, 0.95]
).to_frame("绝对差值")

print("\n【评论级绝对差值分布】")
print(change_summary.round(4))

# =========================
# 5. 年度平均整体极性
# =========================
annual = (
    valid.dropna(subset=["year"])
    .groupby("year")
    .agg(
        样本数=("score_3", "size"),
        窗口3均值=("score_3", "mean"),
        窗口2均值=("score_2", "mean"),
        窗口3标准差=("score_3", "std"),
        窗口2标准差=("score_2", "std"),
        平均绝对差值=("绝对差值", "mean"),
        平均差值=("差值_窗口2减窗口3", "mean")
    )
)

annual["均值差异_窗口2减窗口3"] = (
    annual["窗口2均值"] - annual["窗口3均值"]
)

print("\n【年度平均整体极性对比】")
print(annual.round(4))

# =========================
# 6. 年度別の全体感情極性の推移（日本語図）
# =========================
mpl.rcParams["font.family"] = "Yu Gothic"
mpl.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    annual.index,
    annual["窗口3均值"],
    marker="o",
    linewidth=1.8,
    markersize=5,
    label="否定語ウィンドウ＝3"
)

ax.plot(
    annual.index,
    annual["窗口2均值"],
    marker="s",
    linewidth=1.8,
    markersize=5,
    label="否定語ウィンドウ＝2"
)

# 感情極性スコアの基準線
ax.axhline(
    y=0,
    color="gray",
    linestyle="--",
    linewidth=1
)

# 軸ラベル・タイトル
ax.set_xlabel("年", fontsize=12)
ax.set_ylabel("全体感情スコアの平均値", fontsize=12)
ax.set_title(
    "否定語ウィンドウの設定による年度別感情スコアの比較",
    fontsize=14,
    pad=12
)

# X軸：年
ax.set_xticks(annual.index)
ax.set_xticklabels(
    [str(int(year)) for year in annual.index],
    rotation=45
)

# グリッド・凡例
ax.grid(
    True,
    linestyle=":",
    linewidth=0.7,
    alpha=0.6
)
ax.legend(fontsize=10, frameon=False)

fig.tight_layout()

# 高解像度で保存
fig.savefig(
    "否定語ウィンドウ別_年度感情スコア比較.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================
# 7. 保存结果
# =========================
summary.to_csv("robustness_score_summary.csv", encoding="utf-8-sig")
annual.to_csv("robustness_annual_comparison.csv", encoding="utf-8-sig")
valid.to_csv("robustness_review_level_comparison.csv",
             index=False, encoding="utf-8-sig")

print("\n已保存整体统计、年度对比和评论级对比文件。")