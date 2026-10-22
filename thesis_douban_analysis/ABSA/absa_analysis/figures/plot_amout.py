from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR.parent / "annual_level" / "annual_ratio.csv"
OUTPUT_DIR = BASE_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

mpl.rcParams["font.family"] = "Yu Gothic"

df = pd.read_csv(INPUT_FILE)
df["语料来源"] = df["语料来源"].replace({"短评": "短評", "书评": "書評"})

aspects = [f"A{i}" for i in range(1, 10)]
years = list(range(2015, 2026))

for source in df["语料来源"].unique():
    data = df[df["语料来源"] == source]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, aspect in zip(axes, aspects):
        sub = data[data["aspect"] == aspect].set_index("year").reindex(years)

        ax.plot(sub.index, sub["N"], marker="o")

        ax.set_title(aspect, fontsize=12)
        ax.set_xlim(2015, 2025)
        ax.set_xticks(years)

        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.5)

        ax.axvline(2019, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(2023, color="grey", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_xticklabels([str(y) for y in range(2015, 2025)] + ["2025（~8月）"], rotation=45, ha="right")

    fig.suptitle(f"{source}におけるアスペクト別レビュー数の推移", fontsize=16)

    fig.supxlabel("年", fontsize=13)
    fig.supylabel("レビュー数（件）", fontsize=13)

    plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])

    output = OUTPUT_DIR / f"ABSA_{source}_review_count_3x3.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"保存完了：{output}")