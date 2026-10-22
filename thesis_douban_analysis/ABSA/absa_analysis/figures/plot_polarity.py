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

for source in df["语料来源"].unique():
    data = df[df["语料来源"] == source]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, aspect in zip(axes, aspects):
        sub = data[data["aspect"] == aspect].sort_values("year")

        if len(sub) > 0:
            ax.plot(sub["year"], sub["Positive"] * 100, marker="o", label="ポジティブ")
            ax.plot(sub["year"], sub["Negative"] * 100, marker="o", label="ネガティブ")
            ax.plot(sub["year"], sub["Neutral"] * 100, marker="o", label="ニュートラル")

        ax.axvline(2019, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(2023, color="grey", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_title(aspect, fontsize=12)
        ax.set_xlim(2015, 2025)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(2015, 2026))
        ax.set_xticklabels([str(y) for y in range(2015, 2025)] + ["2025（~8月）"],rotation=45, ha="right")
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)

    fig.supxlabel("年", fontsize=13)
    fig.supylabel("評価極性の平均構成比（%）", fontsize=13)
    fig.suptitle(f"{source}におけるアスペクト別評価極性構成比の時系列変化", fontsize=16, y=0.98)

    plt.tight_layout(rect=[0.04, 0.05, 1, 0.90])

    filename = {"短評": "short", "書評": "long"}.get(source, source)
    output = OUTPUT_DIR / f"ABSA_{filename}_3x3.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"保存完了：{output}")