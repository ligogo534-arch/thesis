from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#2020-2025月度滑动
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "overall_output" / "output_1" / "result.csv"
OUTPUT_DIR = BASE_DIR / "overall_output" / "output_1" / "figures_monthly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = ["Yu Gothic"]
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

MOVING_WINDOW = 3

# ============================================================
# 4. 读取数据
# ============================================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"]).copy()
df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()

print("=" * 70)
print("月次移動平均による感情時系列分析")
print("=" * 70)

print(f"総投稿数：{len(df):,}")
print(f"短評：{(df['data_type'] == '短評').sum():,}")
print(f"書評：{(df['data_type'] == '書評').sum():,}")

# ============================================================
# 5. 月次平均
# ============================================================

monthly_emotion = (
    df.groupby(["data_type", "month"])[EMOTION_COLUMNS]
    .mean()
    .reset_index()
)

monthly_polarity = (
    df.groupby(["data_type", "month"])["整体极性"]
    .mean()
    .reset_index()
)

monthly_emotion = monthly_emotion[
    (monthly_emotion["month"] >= "2019-01-01") &
    (monthly_emotion["month"] <= "2025-08-31")
].copy()

monthly_polarity = monthly_polarity[
    (monthly_polarity["month"] >= "2019-01-01") &
    (monthly_polarity["month"] <= "2025-08-31")
].copy()
# ============================================================
# 6. 计算3个月移动平均
# ============================================================

monthly_emotion_ma = monthly_emotion.copy()

for data_type in monthly_emotion_ma["data_type"].unique():
    mask = monthly_emotion_ma["data_type"] == data_type
    monthly_emotion_ma.loc[mask, EMOTION_COLUMNS] = (
        monthly_emotion_ma.loc[mask, EMOTION_COLUMNS]
        .rolling(window=MOVING_WINDOW, center=True, min_periods=3)
        .mean()
        .values
    )

monthly_polarity_ma = monthly_polarity.copy()

for data_type in monthly_polarity_ma["data_type"].unique():
    mask = monthly_polarity_ma["data_type"] == data_type
    monthly_polarity_ma.loc[mask, "整体极性"] = (
        monthly_polarity_ma.loc[mask, "整体极性"]
        .rolling(window=MOVING_WINDOW, center=True, min_periods=3)
        .mean()
        .values
    )

# ============================================================
# 7. 绘制七类情感的移动平均
# ============================================================

def plot_emotion_moving_average(data_type, output_file):
    data = monthly_emotion_ma[monthly_emotion_ma["data_type"] == data_type].copy()
    data = data.sort_values("month")

    fig, ax = plt.subplots(figsize=(12, 6))

    for emotion in EMOTION_COLUMNS:
        ax.plot(
            data["month"],
            data[emotion],
            linewidth=2,
            label=EMOTION_LABELS[emotion]
        )

    ax.set_xlabel("年月")
    ax.set_ylabel("正規化感情スコア")
    ax.set_title(f"{data_type}の感情スコアの時系列変化（3か月移動平均）")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ {output_file}")


plot_emotion_moving_average(
    "短評",
    OUTPUT_DIR / "short_review_emotion_3month_ma.png"
)

plot_emotion_moving_average(
    "書評",
    OUTPUT_DIR / "long_review_emotion_3month_ma.png"
)

# ============================================================
# 8. 绘制整体极性移动平均
# ============================================================

def plot_polarity_moving_average(data_type, output_file):
    data = monthly_polarity_ma[monthly_polarity_ma["data_type"] == data_type].copy()
    data = data.sort_values("month")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        data["month"],
        data["整体极性"],
        linewidth=2
    )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel("年月")
    ax.set_ylabel("感情極性スコア")
    ax.set_title(f"{data_type}の感情極性の時系列変化（3か月移動平均）")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ {output_file}")


plot_polarity_moving_average(
    "短評",
    OUTPUT_DIR / "short_review_polarity_3month_ma.png"
)

plot_polarity_moving_average(
    "書評",
    OUTPUT_DIR / "long_review_polarity_3month_ma.png"
)

# ============================================================
# 9. 输出数据
# ============================================================

monthly_emotion.to_csv(
    OUTPUT_DIR / "monthly_emotion_scores.csv",
    index=False,
    encoding="utf-8-sig"
)

monthly_emotion_ma.to_csv(
    OUTPUT_DIR / "monthly_emotion_3month_ma.csv",
    index=False,
    encoding="utf-8-sig"
)

monthly_polarity.to_csv(
    OUTPUT_DIR / "monthly_polarity_scores.csv",
    index=False,
    encoding="utf-8-sig"
)

monthly_polarity_ma.to_csv(
    OUTPUT_DIR / "monthly_polarity_3month_ma.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("月次移動平均图绘制完成")
print("=" * 70)
print(f"移动平均窗口：{MOVING_WINDOW}个月")
print(f"输出目录：{OUTPUT_DIR}")