# -*- coding: utf-8 -*-

"""
書評コーパス用

・topic_distribution を文書内で正規化
・外れ値 Topic -1 を除外
・各文書の Topic weight を年次で加重集計
・年次の絶対数を折れ線グラフで表示
・年次の構成比を100%積み上げ棒グラフで表示
・Topic別 Small Multiples を表示

書評は1件の文書に複数Topicが含まれるため、
dominant_topicではなくtopic_distributionを使用する。
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 基本設定
# ============================================================

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

INPUT_PATH = Path("../pipeline/output_long/run5&re/document_topic_aggregation.csv")
OUTPUT_DIR = Path("shuping")
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

# 年次绝对数量折线图
LINE_OUTPUT_PATH = (OUTPUT_DIR / "topics_over_time_long.png")
# 100%積み上げ棒グラフ
PROPORTION_OUTPUT_PATH = (OUTPUT_DIR / "topics_proportion_long.png")
# Small Multiples
SMALL_MULTIPLES_OUTPUT_PATH = (OUTPUT_DIR / "topics_small_multiples_long.png")

# 年次集計CSV
COUNTS_OUTPUT_PATH = (OUTPUT_DIR / "topics_yearly_weighted_counts_review.csv")

# 長評は14 Topic
# 実際のTopic番号が0～13の場合
TOPICS_TO_PLOT = list(range(14))

# 社会的イベント
EVENT_YEARS = [2019, 2023]

# topic_distribution を解析
def parse_topic_distribution(cell: str) -> dict:
    d = json.loads(cell)
    return {
        int(k): v
        for k, v in d.items()
        if k != "-1"
    }

def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    # 年次集計に変更
    df["year"] = df["time"].dt.year
    records = []
    n_excluded = 0

    # 文書ごとに Topic distribution を正規化
    for _, row in df.iterrows():
        dist = parse_topic_distribution(row["topic_distribution"])

        total = sum(dist.values())
        # 非外れ値Topicが存在しない文書を除外
        if total == 0:
            n_excluded += 1
            continue
        for topic_id, count in dist.items():
            records.append({
                "year": row["year"],
                "topic": topic_id,
                "weight": count / total,
            })

    if n_excluded:
        print(
            f"note: excluded "
            f"{n_excluded} documents "
            f"with no non-outlier units"
        )

    return pd.DataFrame.from_records(records)

# 年次 weighted counts
def build_yearly_weighted_counts(
    long_df: pd.DataFrame
) -> pd.DataFrame:
    counts = (long_df.groupby(["year", "topic"])["weight"].sum().unstack(fill_value=0.0).sort_index())
    # 全年度を補完
    full_index = pd.RangeIndex(counts.index.min(),counts.index.max() + 1)
    counts = counts.reindex(full_index,fill_value=0.0)
    counts.index.name = "year"
    return counts

# 年次絶対数：折れ線グラフ
def plot_counts(
    counts: pd.DataFrame,
    output: str | Path,
    topics_to_plot: list[int] | None = None,
):
    cols = (topics_to_plot if topics_to_plot is not None else sorted(counts.columns))
    cols = [topic_id for topic_id in cols if topic_id in counts.columns]
    fig, ax = plt.subplots(figsize=(12, 6))

    # Topic別折れ線
    for topic_id in cols:
        ax.plot(counts.index, counts[topic_id], marker="o", markersize=4, linewidth=1.5, label=str(topic_id))

    # 社会的イベント
    for event_year in EVENT_YEARS:
        if (
            counts.index.min()
            <= event_year
            <= counts.index.max()
        ):

            ax.axvline(event_year,color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("年")
    ax.set_xticks(counts.index)

    year_labels = [
        "2025（～8月）"
        if year == 2025
        else str(year)
        for year in counts.index
    ]

    ax.set_xticklabels(year_labels,rotation=45)

    ax.set_ylabel("トピック別加重出現量", labelpad=15)
    # 图例
    ax.legend(title="トピック番号",bbox_to_anchor=(1.02, 1),loc="upper left",frameon=False)
    # 边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output,dpi=300,bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {output}")

# 100%積み上げ棒グラフ
def plot_yearly_proportions(
    counts: pd.DataFrame,
    output: str | Path,
    topics_to_plot: list[int] | None = None,
):
    TOPIC_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC", "#2F6B9A", "#D37295",
"#6B9E78", "#A67C52",]

    proportions = (counts.div(counts.sum(axis=1),axis=0) * 100)
    cols = (topics_to_plot if topics_to_plot is not None else sorted(proportions.columns))

    cols = [topic_id for topic_id in cols if topic_id in proportions.columns]
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = pd.Series(0.0, index=proportions.index)

    # 100%積み上げ
    for i, topic_id in enumerate(cols):
        ax.bar(
            proportions.index,
            proportions[topic_id],
            bottom=bottom,
            label=str(topic_id),
            width=0.7,
            color=TOPIC_COLORS[topic_id]
        )
        bottom += proportions[topic_id]

    ax.set_xlabel("年")
    ax.set_xticks(proportions.index)

    year_labels = ["2025（～8月）" if year == 2025 else str(year)  for year in proportions.index]
    ax.set_xticklabels(year_labels,rotation=45)

    # Y軸
    ax.set_ylabel("構成比（%）")
    ax.set_ylim(0,100)
    ax.set_yticks(range(0, 101, 20))
    # 图例
    ax.legend(title="トピック番号", bbox_to_anchor=(1.02, 1), loc="upper left",frameon=False)

    # 边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output,dpi=300,bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {output}")


# ============================================================
# Small Multiples
# ============================================================

def plot_topic_small_multiples(
    counts: pd.DataFrame,
    output: str | Path,
    topics_to_plot: list[int] | None = None,
):

    """
    14 Topic の Small Multiples

    ・4×4レイアウト
    ・Y軸を全パネルで統一
    ・2019年、2023年に縦破線
    ・イベント文字なし
    ・イベント番号なし
    ・図内タイトルなし
    """

    cols = (
        topics_to_plot
        if topics_to_plot is not None
        else sorted(counts.columns)
    )

    cols = [
        topic_id
        for topic_id in cols
        if topic_id in counts.columns
    ]

    if len(cols) != 14:

        raise ValueError(
            f"Small Multiplesには14 Topicが必要です。"
            f"現在のTopic数: {len(cols)}"
        )

    # --------------------------------------------------------
    # 4 × 4
    # --------------------------------------------------------

    nrows = 4
    ncols = 4

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 11),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    # --------------------------------------------------------
    # 全Topic共通のY軸
    # --------------------------------------------------------

    ymax = counts[
        cols
    ].to_numpy().max()

    if ymax == 0:

        ymax = 1

    y_limit = ymax * 1.15

    # --------------------------------------------------------
    # X軸ラベル
    # --------------------------------------------------------

    year_labels = [
        "2025（～8月）"
        if year == 2025
        else str(year)
        for year in counts.index
    ]

    # --------------------------------------------------------
    # 各Topic
    # --------------------------------------------------------

    for i, topic_id in enumerate(cols):

        ax = axes[i]

        # ----------------------------------------------------
        # Topic折线
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
        # 社会的イベント
        # ----------------------------------------------------

        for event_year in EVENT_YEARS:

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

        ax.set_xticks(
            counts.index
        )

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

    # --------------------------------------------------------
    # 删除4×4中剩余的2个空白面板
    # --------------------------------------------------------

    for j in range(
        len(cols),
        len(axes)
    ):

        fig.delaxes(
            axes[j]
        )

    # --------------------------------------------------------
    # 全体共通轴标签
    # --------------------------------------------------------

    fig.supxlabel(
        "年",
        fontsize=14
    )

    fig.supylabel(
        "トピック別加重出現量",
        fontsize=14
    )

    # --------------------------------------------------------
    # 布局
    # --------------------------------------------------------

    fig.tight_layout(
        rect=[0.04, 0.04, 1, 1]
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"saved: {output}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("書評コーパス：年次Topic分析")
    print("=" * 60)

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------

    print("\n[1] データ読み込み")

    long_df = load_data(
        INPUT_PATH
    )

    print(
        f"分析対象unit数: "
        f"{len(long_df):,}"
    )

    # 年次集計
    print("\n[2] 年次weighted集計")

    counts = build_yearly_weighted_counts(
        long_df
    )

    print(
        f"期間: "
        f"{counts.index.min()} ～ "
        f"{counts.index.max()}"
    )

    counts.to_csv(
        COUNTS_OUTPUT_PATH,
        encoding="utf-8-sig"
    )

    print(
        f"saved counts table: "
        f"{COUNTS_OUTPUT_PATH}"
    )

    print("\n[3] 年次絶対数：折れ線グラフ")

    plot_counts(
        counts,
        LINE_OUTPUT_PATH,
        topics_to_plot=TOPICS_TO_PLOT
    )

    print("\n[4] 年次構成比：100%積み上げ棒グラフ")

    plot_yearly_proportions(
        counts,
        PROPORTION_OUTPUT_PATH,
        topics_to_plot=TOPICS_TO_PLOT
    )

    print("\n[5] Topic別 Small Multiples")

    plot_topic_small_multiples(
        counts,
        SMALL_MULTIPLES_OUTPUT_PATH,
        topics_to_plot=TOPICS_TO_PLOT
    )

    print("\n" + "=" * 60)
    print("完了")
    print("=" * 60)


if __name__ == "__main__":
    main()