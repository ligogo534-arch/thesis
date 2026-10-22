# -*- coding: utf-8 -*-
"""
候補語のtopic内集中度チェック(拡張版:文書頻度パイプラインの続き)

目的:
    「問題」のように全体文書頻度は高いが、それが
      (A) 特定のtopicに真に集中して出現する実質的な語なのか
      (B) 全topicにほぼ均等に散らばっているだけの、単なる高頻度ノイズなのか
    を判別する。判別基準は「その語のtopic内出現率」を「コーパス全体での
    基線出現率」と比較すること。(A)なら基線を大きく上回るtopicが存在する
    はずであり、(B)ならどのtopicも基線付近に留まるはずである。

    加えて、該当語を含むunitを各topicから抽出し、人手で共起語を確認できる
    サンプルを出力する(「性別問題」のような複合的な実質語なのか、
    「有点问题」のような評価的な慣用表現なのかを見分けるため)。

設計方針:
    - run_bertopic_pipeline_v3.py 本体は一切変更・importしない。
    - 入力は本体が既に出力済みの units_with_topics.csv のみ
      (再度embeddingやモデルを読み込む必要はない)。
    - 複数の候補語を一括で判定できるようにする(「問題」以外の語も
      同じロジックで確認したくなった場合に備える)。

出力:
    ./topic_concentration_check_<corpus_name>.csv
        列: word, topic, topic_doc_freq, corpus_baseline_freq, lift,
            topic_unit_count, corpus_total_units
    ./topic_concentration_samples_<corpus_name>.txt
        各(word, topic)の組み合わせについて、該当語を含むunit_textの
        サンプル(最大 SAMPLES_PER_TOPIC 件)を書き出す
"""

from pathlib import Path

import pandas as pd

# ============ CONFIG (本体の output_dir と揃えて手動指定すること) ============
# 停用詞更新後の最新runの output_dir を指すよう、都度書き換えて使用する
RUNS = [
    {
        "name": "短評",
        "units_csv": Path("../../pipeline/output_short/run4_1/units_with_topics.csv"),
    },
    {
        "name": "書評",
        "units_csv": Path("../../pipeline/output_long/run4_1/units_with_topics.csv"),
    },
]

# 判定したい語(「問題」以外を追加したくなった場合はここに足すだけでよい)
TARGET_WORDS = ["问题"]

SAMPLES_PER_TOPIC = 8          # topicごとに抽出する該当語含有サンプル数
MIN_UNIT_COUNT_FOR_DISPLAY = 5  # あまりに小さいtopic(外れ値付近)は判定対象から除く目安
LIFT_FLAG_THRESHOLD = 1.3       # topic内出現率が基線の何倍以上なら「集中している」とみなすかの目安値
                                 # (絶対基準ではなく、結果を見ながら調整する前提の初期値)

OUTPUT_DIR = Path(".")
# =======================================================================


def check_concentration_for_corpus(corpus_name, units_csv, target_words):
    if not units_csv.exists():
        print(f"[エラー] {units_csv} が見つかりません。CONFIGのパスを確認してください。スキップします。")
        return None, None

    df = pd.read_csv(units_csv)
    if "topic" not in df.columns or "unit_text" not in df.columns:
        print(f"[エラー] {units_csv} に必要な列(topic, unit_text)がありません。スキップします。")
        return None, None

    df["unit_text"] = df["unit_text"].astype(str)
    total_units = len(df)

    records = []
    sample_lines = []

    for word in target_words:
        df["_hit"] = df["unit_text"].str.contains(word, regex=False)
        corpus_baseline_freq = df["_hit"].mean()
        print(f"[{corpus_name}] 「{word}」コーパス全体基線出現率: {corpus_baseline_freq:.4f}"
              f"({df['_hit'].sum()} / {total_units} units)")

        topic_ids = sorted(df["topic"].unique().tolist())
        for topic_id in topic_ids:
            topic_df = df[df["topic"] == topic_id]
            topic_unit_count = len(topic_df)
            if topic_unit_count < MIN_UNIT_COUNT_FOR_DISPLAY:
                continue

            topic_doc_freq = topic_df["_hit"].mean()
            lift = (topic_doc_freq / corpus_baseline_freq) if corpus_baseline_freq > 0 else float("nan")

            records.append({
                "word": word,
                "topic": topic_id,
                "topic_unit_count": topic_unit_count,
                "topic_doc_freq": round(topic_doc_freq, 4),
                "corpus_baseline_freq": round(corpus_baseline_freq, 4),
                "lift": round(lift, 3) if lift == lift else None,  # NaNチェック
                "concentration_flag": "集中(要保留候補)" if (lift == lift and lift >= LIFT_FLAG_THRESHOLD) else "",
            })

            # サンプル抽出(該当語を含むunitのみ)
            hits = topic_df[topic_df["_hit"]]["unit_text"].tolist()
            if hits:
                sample_lines.append(f"\n===== word=「{word}」 topic={topic_id} "
                                     f"(topic内出現率={topic_doc_freq:.3f}, lift={lift:.2f}) =====")
                for i, text in enumerate(hits[:SAMPLES_PER_TOPIC], 1):
                    snippet = text if len(text) <= 200 else text[:200] + "..."
                    sample_lines.append(f"[{i}] {snippet}")

        df.drop(columns=["_hit"], inplace=True)

    df_result = pd.DataFrame(records).sort_values(["word", "lift"], ascending=[True, False])
    return df_result, sample_lines


def main():
    for run_cfg in RUNS:
        corpus_name = run_cfg["name"]
        print(f"\n{'='*60}\n[{corpus_name}] 判定開始\n{'='*60}")

        df_result, sample_lines = check_concentration_for_corpus(
            corpus_name, run_cfg["units_csv"], TARGET_WORDS
        )
        if df_result is None:
            continue

        out_csv = OUTPUT_DIR / f"topic_concentration_check_{corpus_name}.csv"
        df_result.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"集中度チェック結果を保存: {out_csv}")

        out_txt = OUTPUT_DIR / f"topic_concentration_samples_{corpus_name}.txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(sample_lines))
        print(f"該当語サンプルを保存: {out_txt}")

        print(f"\n--- [{corpus_name}] lift上位(集中度が高い順) ---")
        print(df_result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()