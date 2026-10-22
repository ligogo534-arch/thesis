from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic"]
# ============================================================
# 1. 路径
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "overall_output" / "output_1" / "result.csv"
OUTPUT_DIR = BASE_DIR / "overall_output" / "output_1" / "figures_yearly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. 参数
EMOTION_COLUMNS = ["乐", "好", "怒", "哀", "惧", "恶", "惊"]

EMOTION_LABELS = {
    "乐": "楽",
    "好": "好",
    "怒": "怒",
    "哀": "哀",
    "惧": "恐",
    "恶": "嫌",
    "惊": "驚"
}

# ============================================================
# 3. 读取数据
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"]).copy()
df["year"] = df["time"].dt.year
print("=" * 70)
print("情感时序分析")
print("=" * 70)

print(f"总评论数：{len(df):,}")
print(f"短評：{(df['data_type'] == '短評').sum():,}")
print(f"書評：{(df['data_type'] == '書評').sum():,}")

# 4. 按年份计算平均情感分数
yearly_emotion = (
    df.groupby(["data_type", "year"])[EMOTION_COLUMNS]
    .mean()
    .reset_index()
)

yearly_polarity = (
    df.groupby(["data_type", "year"])["整体极性"]
    .mean()
    .reset_index()
)

# ============================================================
# 5. 绘制七类情感时序图
def plot_emotion_trend(data_type, output_file):
    data = yearly_emotion[yearly_emotion["data_type"] == data_type].copy()
    data = data.sort_values("year")

    fig, ax = plt.subplots(figsize=(10, 6))

    for emotion in EMOTION_COLUMNS:
        ax.plot(
            data["year"],
            data[emotion],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=EMOTION_LABELS[emotion]
        )

    ax.set_xlabel("年")
    ax.set_ylabel("正規化感情スコア")
    ax.set_title(f"{data_type}の感情スコアの時系列変化")

    ax.set_xticks(sorted(data["year"].unique()))
    ax.tick_params(axis="x", rotation=45)

    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ {output_file}")


plot_emotion_trend("短評", OUTPUT_DIR / "short_review_emotion_trend.png")
plot_emotion_trend("書評",OUTPUT_DIR / "long_review_emotion_trend.png")

# ============================================================
# 6. 绘制整体极性时序图
def plot_polarity_trend(data_type, output_file):
    data = yearly_polarity[yearly_polarity["data_type"] == data_type].copy()
    data = data.sort_values("year")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        data["year"],
        data["整体极性"],
        marker="o",
        linewidth=2,
        markersize=5
    )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel("年")
    ax.set_ylabel("感情極性スコア")
    ax.set_title(f"{data_type}の感情極性の時系列変化")

    ax.set_xticks(sorted(data["year"].unique()))
    ax.tick_params(axis="x", rotation=45)

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ {output_file}")

plot_polarity_trend("短評",OUTPUT_DIR / "short_review_polarity_trend.png")
plot_polarity_trend("書評",OUTPUT_DIR / "long_review_polarity_trend.png")

# ============================================================
# 7. 输出年度数据
yearly_emotion.to_csv(OUTPUT_DIR / "yearly_emotion_scores.csv",index=False,encoding="utf-8-sig")
yearly_polarity.to_csv(OUTPUT_DIR / "yearly_polarity_scores.csv",index=False,encoding="utf-8-sig")

print("\n" + "=" * 70)
print("绘图完成")
print("=" * 70)

print(f"输出目录：{OUTPUT_DIR}")