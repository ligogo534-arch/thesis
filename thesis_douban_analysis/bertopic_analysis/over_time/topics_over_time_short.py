# -*- coding: utf-8 -*-

"""
短評コーパス用：dominant_topic をそのまま使って月次絶対件数の推移を可視化する。

短評は1件=1unitのため、topic_distributionではなくdominant_topicを
そのまま集計すればよい。
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# 入力CSV
INPUT_PATH = Path("../pipeline/output_short/run5&re/document_topic_aggregation.csv")

# 出力画像
OUTPUT_PATH = Path("duanping/topics_over_time_short.png")
PROPORTION_OUTPUT_PATH = Path("duanping/topics_over_time_short_proportion.png")
SMALL_MULTIPLES_OUTPUT_PATH = Path("duanping/topics_small_multiples_short.png")

# 是否排除 outlier（dominant_topic == -1）
# True  = 排除
# False = 不排除
EXCLUDE_OUTLIER = True

# 要绘制的Topic
# None = 绘制全部Topic
# 例如：[0, 2, 5] = 只绘制Topic 0、2、5
TOPICS_TO_PLOT = None

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
})

EVENTS = [
    ("2019-04-12", "東大入学式祝辞"),
    ("2023-02-17", "bilibili対談（全嘻嘻）"),
]

def load_data(
    path: str | Path,
    exclude_outlier: bool = True
) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 时间转换
    df["time"] = pd.to_datetime(df["time"])
    # dominant_topic 转换为整数
    df["dominant_topic"] = df["dominant_topic"].astype(int)
    # 排除离群点 Topic -1和8
    if exclude_outlier:
        df = df[~df["dominant_topic"].isin([-1, 8])].copy()
    # 转换为年份
    df["year"] = df["time"].dt.year
    return df

def build_yearly_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (df.groupby(["year", "dominant_topic"]).size().unstack(fill_value=0).sort_index())
    full_index = pd.RangeIndex(counts.index.min(),counts.index.max() + 1)
    counts = counts.reindex(full_index, fill_value=0)
    counts.index.name = "year"
    return counts

def plot_counts(
    counts: pd.DataFrame,
    output: str | Path,
    title: str = "短評：トピック別件数の年次推移",
    topics_to_plot: list[int] | None = None,):
    fig, ax = plt.subplots(figsize=(12, 6))

    # 如果没有指定Topic，则绘制全部
    cols = (topics_to_plot if topics_to_plot is not None else sorted(counts.columns))

    for topic_id in cols:
        if topic_id not in counts.columns:
            continue

        # ラベルは番号のみ
        ax.plot(counts.index, counts[topic_id], marker="o", markersize=3, linewidth=1.3, label=str(topic_id))


    ymax = (counts.to_numpy().max() if counts.size else 1)

    for date_str, event_label in EVENTS:
        event_date = pd.Timestamp(date_str)
        event_year = event_date.year
        if (counts.index.min() <= event_year <= counts.index.max()):
            ax.axvline(event_year,color="gray",linestyle="--",linewidth=1,alpha=0.7)
            ax.text(event_year,ymax * 1.02,event_label,rotation=90,va="bottom",
                ha="right",fontsize=11,color="dimgray")

    ax.set_xlabel("年")
    ax.set_ylabel("件数（絶対数）")
    #ax.set_title(title)

    # X轴：每年显示一个主刻度
    ax.set_xticks(counts.index)
    year_labels = ["2025（～8月）" if year == 2025 else str(year) for year in counts.index]
    ax.set_xticklabels(year_labels, rotation=45)

    ax.legend(title="トピック番号",bbox_to_anchor=(1.02, 1),loc="upper left",fontsize=11,frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output,dpi=300,bbox_inches="tight")

    plt.close(fig)
    print(f"saved: {output}")

def plot_yearly_proportions(
    counts: pd.DataFrame,
    output: str | Path,
    title: str = "短評：トピック別構成比の年次推移",
    topics_to_plot: list[int] | None = None,
):
    TOPIC_COLORS = [
        "#4E79A7",
        "#76A5AF",
        "#9FBAD0",
        "#59A14F",
        "#86B97F",
        "#B5C99A",
        "#F28E72",
        "#E6A07E",
        "#B07AA1",
        "#9C9C9C",
        "#7F8C8D",
        "#C5B0D5",
        "#D4A373",
        "#8C96C6",
        "#A6A6A6",
    ]

    proportions = counts.div(
        counts.sum(axis=1),
        axis=0
    ) * 100

    # 绘制Topic
    cols = (
        topics_to_plot
        if topics_to_plot is not None
        else sorted(proportions.columns)
    )

    # 只保留实际存在的Topic
    cols = [topic_id for topic_id in cols if topic_id in proportions.columns]
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = pd.Series(0.0, index=proportions.index)

    for i, topic_id in enumerate(cols):
        ax.bar(
            proportions.index,
            proportions[topic_id],
            bottom=bottom,
            label=str(topic_id),
            width=0.7,
            color=TOPIC_COLORS[i % len(TOPIC_COLORS)]
        )
        bottom += proportions[topic_id]

    ax.set_xlabel("年")
    ax.set_xticks(proportions.index)
    year_labels = [
        "2025（～8月）" if year == 2025 else str(year)
        for year in proportions.index
    ]

    ax.set_xticklabels(year_labels,rotation=45)

    ax.set_ylabel("構成比（%）")
    ax.set_ylim(0, 100)

    # 0、20、40、60、80、100%
    ax.set_yticks(range(0, 101, 20))
    #ax.set_title(title)

    ax.legend(
        title="トピック番号",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=11,
        frameon=False
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)
    print(f"saved: {output}")

def plot_topic_small_multiples(
    counts: pd.DataFrame,
    output: str | Path,
    topics_to_plot: list[int] | None = None,
):
    """
    短評：Topic別 Small Multiples

    ・3×3レイアウト
    ・Y軸を全パネルで統一
    ・2019年、2023年に社会的イベントの縦破線
    ・イベントの文字・番号は表示しない
    ・図内タイトルなし
    """

    # ========================================================
    # Topic設定
    # ========================================================

    cols = (
        topics_to_plot
        if topics_to_plot is not None
        else sorted(counts.columns)
    )

    # 実際に存在するTopicのみ
    cols = [
        topic_id
        for topic_id in cols
        if topic_id in counts.columns
    ]

    # 短評は9 Topic（0～7、9）
    if len(cols) != 9:
        raise ValueError(
            f"Small Multiplesには9 Topicが必要です。"
            f"現在のTopic数: {len(cols)}"
        )

    # ========================================================
    # 3 × 3
    # ========================================================

    nrows = 3
    ncols = 3

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 10),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    # ========================================================
    # Y軸を全パネルで統一
    # ========================================================

    ymax = counts[cols].to_numpy().max()

    if ymax == 0:
        ymax = 1

    y_limit = ymax * 1.15

    # ========================================================
    # X軸ラベル
    # ========================================================

    year_labels = [
        "2025（～8月）" if year == 2025 else str(year)
        for year in counts.index
    ]

    # ========================================================
    # 社会的イベント
    # ========================================================

    event_years = [2019, 2023]

    # ========================================================
    # 各Topicを描画
    # ========================================================

    for i, topic_id in enumerate(cols):

        ax = axes[i]

        # ----------------------------------------------------
        # Topicの年次件数
        # ----------------------------------------------------

        ax.plot(
            counts.index,
            counts[topic_id],
            marker="o",
            markersize=4,
            linewidth=1.5
        )

        # ----------------------------------------------------
        # Topic番号
        # ----------------------------------------------------

        ax.text(
            0.04,
            0.93,
            f"Topic {topic_id}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top"
        )

        # ----------------------------------------------------
        # 社会的イベントの縦線
        # ----------------------------------------------------

        for event_year in event_years:

            if (
                counts.index.min()
                <= event_year
                <= counts.index.max()
            ):

                ax.axvline(
                    event_year,
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7
                )

        # ----------------------------------------------------
        # Y軸
        # ----------------------------------------------------

        ax.set_ylim(
            0,
            y_limit
        )

        ax.tick_params(
            axis="y",
            labelsize=10
        )

        # ----------------------------------------------------
        # X軸
        # ----------------------------------------------------

        ax.set_xticks(counts.index)

        ax.set_xticklabels(
            year_labels,
            rotation=45,
            fontsize=10
        )

        # ----------------------------------------------------
        # 网格线
        # ----------------------------------------------------

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.5,
            alpha=0.4
        )

        # ----------------------------------------------------
        # 边框
        # ----------------------------------------------------

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ========================================================
    # 全体共通轴标签
    # ========================================================

    fig.supxlabel(
        "年",
        fontsize=14
    )

    fig.supylabel(
        "件数（絶対数）",
        fontsize=14
    )

    # ========================================================
    # 布局
    # ========================================================

    fig.tight_layout(
        rect=[0.04, 0.04, 1, 1]
    )

    # ========================================================
    # 保存
    # ========================================================

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"saved: {output}")

def main():
    print("=" * 60)
    print("短評：トピック別月次件数の可視化")
    print("=" * 60)

    print(f"入力ファイル: {INPUT_PATH}")
    df = load_data(INPUT_PATH,exclude_outlier=EXCLUDE_OUTLIER)
    print(f"分析対象件数: {len(df):,}")

    counts = build_yearly_counts(df)
    print(f"トピック数: {len(counts.columns)}")
    print(f"期間: "f"{counts.index.min()} ～ "f"{counts.index.max()}")

    # 保存月次件数CSV
    counts_output = OUTPUT_PATH.with_suffix(".csv")
    counts.to_csv(counts_output,encoding="utf-8-sig")
    print(f"saved counts table: {counts_output}")

    plot_counts(counts,OUTPUT_PATH,topics_to_plot=TOPICS_TO_PLOT)
    plot_yearly_proportions(counts,PROPORTION_OUTPUT_PATH, topics_to_plot=TOPICS_TO_PLOT)
    plot_topic_small_multiples(counts,SMALL_MULTIPLES_OUTPUT_PATH,topics_to_plot=TOPICS_TO_PLOT)
    print("=" * 60)
    print("完了")
    print("=" * 60)

if __name__ == "__main__":
    main()