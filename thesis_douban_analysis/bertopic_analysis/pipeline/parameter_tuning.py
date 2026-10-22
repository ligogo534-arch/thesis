# -*- coding: utf-8 -*-
"""
UMAP / HDBSCAN パラメータ網格搜索スクリプト

目的: run_bertopic_pipeline_v3.py と同じ前処理(辞書・ストップワード・トークナイザ)、
同じembeddingモデルを再利用しつつ、複数のパラメータ組み合わせを効率的に比較する。

効率化のポイント:
1. Embeddingは対象コーパスにつき1回だけ計算し、.npyにキャッシュして再利用する
   (最も計算コストが高い工程のため、パラメータ探索のたびに再計算しない)
2. UMAPは n_neighbors の値ごとに1回だけ計算し、同じ n_neighbors を共有する
   複数のHDBSCANパラメータ組み合わせで使い回す
   (UMAPの次元削減結果はHDBSCANのパラメータに依存しないため)
3. HDBSCANは(min_cluster_size, min_samples)の全組み合わせについて高速にクラスタリングのみ実行し、
   トピック数・外れ値比率などの「軽量な指標」で一次スクリーニングする
4. Topic Coherence(計算コストが高い)は、一次スクリーニングを通過した上位候補のみに対して、
   実際にBERTopic(vectorizer込み)をfit_transformして算出する

実行方法: python grid_search_params.py

事前準備:
- run_bertopic_pipeline_v3.py と同じディレクトリに配置すること(関数を再利用するため)
- 対象コーパスは下記 TARGET_CORPUS で指定
"""

import sys
import json
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic


import run3 as pipeline

# ============ CONFIG ============
# 探索対象のコーパス設定。pipeline.CORPUS_CONFIGS の中から対象を選ぶ(nameで指定)
TARGET_CORPUS_NAME = "短評"   # "短評" または "書評" を指定

# 探索するパラメータ範囲(対象コーパスの規模に応じて調整すること)
GRID_UMAP_N_NEIGHBORS = [15, 20, 25]
GRID_MIN_CLUSTER_SIZE = [30, 40, 50, 60, 70]
GRID_MIN_SAMPLES =  [10, 15]

# 一次スクリーニングで「良さそう」と判断する目安の範囲(この範囲内の組み合わせのみ、
# 後段でCoherenceを計算する。範囲は仮の目安であり、実際の結果を見て調整すること)
TARGET_OUTLIER_RATIO_RANGE = (0.10, 0.4)   # 外れ値比率がこの範囲内
TARGET_N_TOPICS_RANGE = (6, 15)             # トピック数(外れ値除く)がこの範囲内
MAX_CANDIDATES_FOR_COHERENCE = 15             # Coherenceを計算する候補数の上限(計算コスト抑制)

BASE_DIR = Path(__file__).resolve().parent
TUNING_DIR = BASE_DIR / "tuning_results"
EMBEDDING_CACHE_DIR = TUNING_DIR / "cache" / TARGET_CORPUS_NAME
GRID_OUTPUT_DIR = TUNING_DIR / "results" / TARGET_CORPUS_NAME

RANDOM_STATE = 42

# 既存の run_bertopic_pipeline_v3.py 実行結果(例: output_long/run3)を再利用する場合はパスを指定する。
# 指定した場合、embeddingの再計算・grid_search_cacheへの新規保存は行わず、
# 既存run内の unit_embeddings.npy / units_with_topics.csv を直接読み込む。
# 指定しない(None)場合は、従来通り新規計算 or grid_search_cache のキャッシュを使用する。
REUSE_FROM_RUN_DIR = Path("./output_short/run3")   # 例: 短評の場合は Path("./output_short/run3")。使わない場合は None
# =================================

#调试随机种子时填seed_check，不调试的时候不填
RUN_MODE = "seed_check"
SEED_CHECK_N_NEIGHBORS = 15
SEED_CHECK_MIN_CLUSTER_SIZE = 60
SEED_CHECK_MIN_SAMPLES = 15
SEED_CHECK_SEEDS = [1, 7, 21, 42, 100, 123, 2024, 31415]

RUN_TAG = "15_60_15"

def get_target_config(corpus_name):
    for config in pipeline.CORPUS_CONFIGS:
        if config["name"] == corpus_name:
            return config
    raise ValueError(f"CORPUS_CONFIGSに '{corpus_name}' という名前のコーパス設定が見つかりません。")


def load_embeddings_from_existing_run(run_dir):
    """
    既存の run_bertopic_pipeline_v3.py 実行結果(unit_embeddings.npy / units_with_topics.csv)を
    直接読み込む。両ファイルは同一の docs リストから同じ順序で生成されているため、
    行の対応関係(embeddings[i] ⇔ df_units.iloc[i])はそのまま保証される。
    """
    run_dir = Path(run_dir)
    emb_path = run_dir / "unit_embeddings.npy"
    units_path = run_dir / "units_with_topics.csv"

    if not emb_path.exists() or not units_path.exists():
        raise FileNotFoundError(
            f"REUSE_FROM_RUN_DIR に指定されたディレクトリに必要なファイルが見つかりません: "
            f"{emb_path} または {units_path}"
        )

    print(f"[既存run再利用] {run_dir} から embedding を読み込みます...")
    embeddings = np.load(emb_path)
    df_units = pd.read_csv(units_path)
    docs = df_units["unit_text"].astype(str).tolist()

    if len(docs) != embeddings.shape[0]:
        raise ValueError(
            f"docs数({len(docs)})とembedding行数({embeddings.shape[0]})が一致しません。"
            f"units_with_topics.csv と unit_embeddings.npy が同一実行由来のものか確認してください。"
        )

    print(f"読み込み完了: {len(docs)} 件のユニット, embedding shape={embeddings.shape}")
    return embeddings, docs, df_units


def load_or_compute_embeddings(config, embedding_model, tokenizer):
    """
    Embeddingの取得優先順位:
    1. REUSE_FROM_RUN_DIR が指定されていれば、既存の正式run結果をそのまま再利用(再計算なし)
    2. grid_search_cache に過去のgrid search実行でキャッシュされたものがあればそれを使用
    3. どちらもなければ新規に計算し、grid_search_cache に保存する
    """
    if REUSE_FROM_RUN_DIR is not None:
        return load_embeddings_from_existing_run(REUSE_FROM_RUN_DIR)

    EMBEDDING_CACHE_DIR.mkdir(exist_ok=True, parents=True)
    cache_key = f"{config['name']}_{config['split_unit']}"
    emb_cache_path = EMBEDDING_CACHE_DIR / f"{cache_key}_embeddings.npy"
    docs_cache_path = EMBEDDING_CACHE_DIR / f"{cache_key}_docs.json"
    meta_cache_path = EMBEDDING_CACHE_DIR / f"{cache_key}_doc_units.csv"

    if emb_cache_path.exists() and docs_cache_path.exists() and meta_cache_path.exists():
        print(f"[キャッシュ利用] 既存のembeddingを読み込みます: {emb_cache_path}")
        embeddings = np.load(emb_cache_path)
        with open(docs_cache_path, encoding="utf-8") as f:
            docs = json.load(f)
        df_units = pd.read_csv(meta_cache_path)
        return embeddings, docs, df_units

    print(f"[新規計算] embeddingを生成します(対象: {config['name']}, split_unit={config['split_unit']})...")
    df_orig, df_units, docs, split_stats = pipeline.load_corpus(
        config["csv_path"], config["content_col"], config["id_cols"], config["split_unit"]
    )
    print(f"分析対象ユニット数: {split_stats['n_units_final']}")

    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    np.save(emb_cache_path, embeddings)
    with open(docs_cache_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    df_units.to_csv(meta_cache_path, index=False, encoding="utf-8-sig")
    print(f"embeddingをキャッシュに保存しました: {emb_cache_path}")

    return embeddings, docs, df_units


def run_screening_grid(embeddings):
    """
    軽量な一次スクリーニング: UMAP(n_neighborsごとに1回) → HDBSCAN(全組み合わせ)で
    トピック数・外れ値比率のみを高速に算出する。c-TF-IDFやCoherenceはここでは計算しない。
    """
    results = []

    for n_neighbors in GRID_UMAP_N_NEIGHBORS:
        print(f"\n--- UMAP n_neighbors={n_neighbors} で次元削減中... ---")
        t0 = time.time()
        umap_model = UMAP(
            n_neighbors=n_neighbors, n_components=5, min_dist=0.0,
            metric="cosine", random_state=RANDOM_STATE,
        )
        reduced = umap_model.fit_transform(embeddings)
        print(f"  UMAP完了, {time.time()-t0:.1f}秒")

        for min_cluster_size, min_samples in product(GRID_MIN_CLUSTER_SIZE, GRID_MIN_SAMPLES):
            if min_samples > min_cluster_size:
                continue  # min_samples > min_cluster_size は通常意味のない組み合わせのためスキップ

            t1 = time.time()
            hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
            labels = hdbscan_model.fit_predict(reduced)
            elapsed = time.time() - t1

            n_total = len(labels)
            n_outliers = int((labels == -1).sum())
            outlier_ratio = n_outliers / n_total
            unique_topics = sorted(set(labels) - {-1})
            n_topics = len(unique_topics)

            if n_topics > 0:
                cluster_sizes = [int((labels == t).sum()) for t in unique_topics]
                median_cluster_size = float(np.median(cluster_sizes))
                max_cluster_size = int(np.max(cluster_sizes))
            else:
                median_cluster_size = 0.0
                max_cluster_size = 0

            in_target_range = (
                TARGET_OUTLIER_RATIO_RANGE[0] <= outlier_ratio <= TARGET_OUTLIER_RATIO_RANGE[1]
                and TARGET_N_TOPICS_RANGE[0] <= n_topics <= TARGET_N_TOPICS_RANGE[1]
            )

            results.append({
                "umap_n_neighbors": n_neighbors,
                "hdbscan_min_cluster_size": min_cluster_size,
                "hdbscan_min_samples": min_samples,
                "n_topics": n_topics,
                "outlier_ratio": round(outlier_ratio, 4),
                "median_cluster_size": median_cluster_size,
                "max_cluster_size": max_cluster_size,
                "in_target_range": in_target_range,
                "clustering_seconds": round(elapsed, 2),
            })
            print(f"  min_cluster_size={min_cluster_size}, min_samples={min_samples} "
                  f"→ トピック数={n_topics}, 外れ値={outlier_ratio:.2%} ({elapsed:.1f}秒)")

    df_results = pd.DataFrame(results)
    return df_results


def compute_coherence_for_candidates(df_screening, docs, embeddings, tokenizer, config):
    """一次スクリーニングで目標範囲に入った候補について、実際にBERTopicをfitしてCoherenceを算出する。"""
    candidates = df_screening[df_screening["in_target_range"]].copy()
    if candidates.empty:
        print("\n[警告] 目標範囲(外れ値比率・トピック数)に入る組み合わせがありませんでした。")
        print("TARGET_OUTLIER_RATIO_RANGE / TARGET_N_TOPICS_RANGE の設定を見直すか、")
        print("GRID_MIN_CLUSTER_SIZE / GRID_MIN_SAMPLES の探索範囲を広げてください。")
        return pd.DataFrame()

    # 候補が多すぎる場合は、外れ値比率が目標レンジの中央に近い順に上位のみ採用
    if len(candidates) > MAX_CANDIDATES_FOR_COHERENCE:
        target_mid = sum(TARGET_OUTLIER_RATIO_RANGE) / 2
        candidates["_dist"] = (candidates["outlier_ratio"] - target_mid).abs()
        candidates = candidates.sort_values("_dist").head(MAX_CANDIDATES_FOR_COHERENCE)

    print(f"\n=== {len(candidates)} 件の候補についてCoherenceを計算します ===")

    coherence_results = []
    for _, row in candidates.iterrows():
        n_neighbors = int(row["umap_n_neighbors"])
        min_cluster_size = int(row["hdbscan_min_cluster_size"])
        min_samples = int(row["hdbscan_min_samples"])

        print(f"\n--- n_neighbors={n_neighbors}, min_cluster_size={min_cluster_size}, "
              f"min_samples={min_samples} でBERTopicを実行中... ---")

        umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0,
                           metric="cosine", random_state=RANDOM_STATE)
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                 metric="euclidean", cluster_selection_method="eom",
                                 prediction_data=True)
        vectorizer_model = CountVectorizer(tokenizer=tokenizer, token_pattern=None, lowercase=False)

        topic_model = BERTopic(
            umap_model=umap_model, hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model, top_n_words=10,
            calculate_probabilities=False, verbose=False,
        )
        topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

        topic_info = topic_model.get_topic_info()
        valid_topic_ids = [t for t in topic_info["Topic"].tolist() if t != -1]

        tokenized_docs = [tokenizer(d) for d in docs]
        try:
            df_coh, overall_coherence, n_dropped = pipeline.compute_coherence(
                topic_model, tokenized_docs, valid_topic_ids
            )
        except Exception as e:
            print(f"  [警告] Coherence計算に失敗: {e}")
            overall_coherence = None

        coherence_results.append({
            "umap_n_neighbors": n_neighbors,
            "hdbscan_min_cluster_size": min_cluster_size,
            "hdbscan_min_samples": min_samples,
            "n_topics": int(row["n_topics"]),
            "outlier_ratio": float(row["outlier_ratio"]),
            "overall_coherence_c_v": round(overall_coherence, 4) if overall_coherence is not None else None,
        })
        print(f"  → Coherence(c_v) = {overall_coherence}")

    return pd.DataFrame(coherence_results)


def run_seed_stability_check(embeddings, n_neighbors, min_cluster_size, min_samples, seeds):
    """
    特定のパラメータ組み合わせについて、UMAPのrandom_stateを変えながら複数回実行し、
    結果(トピック数・外れ値比率)がどの程度安定しているかを検証する。
    n_neighbors軸で孤立した「好調な窓」が見られた場合、それが真の構造か、
    単なる乱数の偶然かを切り分けるための簡易チェック。
    """
    results = []
    for seed in seeds:
        t0 = time.time()
        umap_model = UMAP(
            n_neighbors=n_neighbors, n_components=5, min_dist=0.0,
            metric="cosine", random_state=seed,
        )
        reduced = umap_model.fit_transform(embeddings)

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples,
            metric="euclidean", cluster_selection_method="eom",
            prediction_data=False,
        )
        labels = hdbscan_model.fit_predict(reduced)

        n_total = len(labels)
        n_outliers = int((labels == -1).sum())
        outlier_ratio = n_outliers / n_total
        n_topics = len(set(labels) - {-1})
        elapsed = time.time() - t0

        results.append({
            "random_state": seed,
            "n_topics": n_topics,
            "outlier_ratio": round(outlier_ratio, 4),
            "elapsed_seconds": round(elapsed, 1),
        })
        print(f"  random_state={seed} → トピック数={n_topics}, 外れ値={outlier_ratio:.2%} ({elapsed:.1f}秒)")

    return pd.DataFrame(results)

def main():
    GRID_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    config = get_target_config(TARGET_CORPUS_NAME)
    print(f"対象コーパス: {config['name']} (split_unit={config['split_unit']})")

    # --- 辞書・ストップワードの準備(pipeline.pyのロジックを再利用) ---
    if not Path(pipeline.MERGED_STOPWORDS_PATH).exists():
        pipeline.merge_stopwords(
            pipeline.BASE_STOPWORDS_PATH, pipeline.CANDIDATE_STOPWORDS_PATH, pipeline.MERGED_STOPWORDS_PATH
        )
    stopwords = pipeline.load_stopwords(pipeline.MERGED_STOPWORDS_PATH)
    print(f"ストップワード数: {len(stopwords)}")

    if Path(pipeline.CUSTOM_DICT_PATH).exists():
        import jieba
        jieba.load_userdict(pipeline.CUSTOM_DICT_PATH)
        print(f"自定义词典を読み込みました: {pipeline.CUSTOM_DICT_PATH}")

    tokenizer = pipeline.build_tokenizer(stopwords)

    # --- Embeddingモデル読み込み ---
    # REUSE_FROM_RUN_DIR が指定されている場合、embeddingは既存ファイルから読み込むため
    # embeddingモデル自体のロードは不要(時間・メモリの節約)
    if REUSE_FROM_RUN_DIR is not None:
        print("REUSE_FROM_RUN_DIR が指定されているため、embeddingモデルの読み込みをスキップします。")
        embedding_model = None
    else:
        from sentence_transformers import SentenceTransformer
        print("Embeddingモデルを読み込み中...")
        embedding_model = SentenceTransformer(pipeline.EMBEDDING_MODEL_NAME)

    # --- Embedding取得(キャッシュ利用) ---
    embeddings, docs, df_units = load_or_compute_embeddings(config, embedding_model, tokenizer)
    print(f"Embedding shape: {embeddings.shape}")

    if RUN_MODE == "seed_check":
        print(f"\n{'=' * 60}")
        print(f"乱数種子安定性検証: n_neighbors={SEED_CHECK_N_NEIGHBORS}, "
              f"min_cluster_size={SEED_CHECK_MIN_CLUSTER_SIZE}, min_samples={SEED_CHECK_MIN_SAMPLES}")
        print(f"{'=' * 60}")
        df_seed_check = run_seed_stability_check(
            embeddings, SEED_CHECK_N_NEIGHBORS, SEED_CHECK_MIN_CLUSTER_SIZE,
            SEED_CHECK_MIN_SAMPLES, SEED_CHECK_SEEDS,
        )
        seed_check_path = GRID_OUTPUT_DIR / f"{config['name']}_{RUN_TAG}_seed_stability.csv"
        df_seed_check.to_csv(seed_check_path, index=False, encoding="utf-8-sig")
        print(f"\n結果を保存: {seed_check_path}")
        print(f"\nトピック数の標準偏差: {df_seed_check['n_topics'].std():.2f}")
        print(f"外れ値比率の標準偏差: {df_seed_check['outlier_ratio'].std():.4f}")
        print("\n標準偏差が小さければ(トピック数の変動が1-2個程度、外れ値比率の変動が数%程度)、")
        print("このパラメータ領域は乱数に対して安定していると判断できる。")
        print("逆に大きく変動する場合、n_neighbors=50付近の好調な結果は偶然の産物である可能性が高い。")
        return

    # --- Step A: 軽量スクリーニング(全組み合わせ) ---
    print(f"\n{'='*60}\nStep A: 軽量スクリーニング開始\n{'='*60}")
    df_screening = run_screening_grid(embeddings)

    #screening_path = GRID_OUTPUT_DIR / f"{config['name']}_{RUN_TAG}_screening_results.csv"
    screening_path = GRID_OUTPUT_DIR / f"{config['name']}_screening_results3.csv"
    df_screening.to_csv(screening_path, index=False, encoding="utf-8-sig")
    print(f"\nスクリーニング結果を保存: {screening_path}")
    print(f"目標範囲内の組み合わせ数: {df_screening['in_target_range'].sum()} / {len(df_screening)}")

    # --- Step B: 候補についてCoherenceを計算 ---
    print(f"\n{'='*60}\nStep B: 候補のCoherence計算\n{'='*60}")
    df_coherence_results = compute_coherence_for_candidates(df_screening, docs, embeddings, tokenizer, config)

    if not df_coherence_results.empty:
        #coherence_path = GRID_OUTPUT_DIR / f"{config['name']}_{RUN_TAG}_coherence_comparison2.csv"
        coherence_path = GRID_OUTPUT_DIR / f"{config['name']}_coherence_comparison3.csv"
        df_coherence_results = df_coherence_results.sort_values("overall_coherence_c_v", ascending=False)
        df_coherence_results.to_csv(coherence_path, index=False, encoding="utf-8-sig")
        print(f"\nCoherence比較結果を保存: {coherence_path}")
        print("\n=== 上位候補 ===")
        print(df_coherence_results.head(5).to_string(index=False))

    print("\n=== 完了 ===")
    print(f"1. {screening_path} で全組み合わせの外れ値比率・トピック数を確認")
    print(f"2. Coherence比較結果があれば、そこから最終パラメータを選定")
    print(f"3. 選定したパラメータを run_bertopic_pipeline_v3.py の CORPUS_CONFIGS に反映して正式実行")


if __name__ == "__main__":
    main()