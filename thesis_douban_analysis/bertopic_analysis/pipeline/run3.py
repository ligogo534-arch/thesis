# -*- coding: utf-8 -*-
#BERTopic 完全パイプライン(正式モデル・分析単位を可変とする版)

import re
import json
import hashlib
import platform
import datetime
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import jieba
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
#from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# ============ CONFIG ============
CUSTOM_DICT_PATH = "../dict_stopword/output/custom_dict_template.txt"
BASE_STOPWORDS_PATH = "../dict_stopword/hit_stopwords.txt"
CANDIDATE_STOPWORDS_PATH = "../dict_stopword/output/stopwords_candidate.txt"
MERGED_STOPWORDS_PATH = "stopwords_final.txt"

EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
JIEBA_HMM_ENABLED = True

CORPUS_CONFIGS = [
    {
        "name": "短評",
        "csv_path": "../../data_cleaning2/text_cleaning/duanping_clean_base.csv",
        "content_col": "content",
        "id_cols": ["book_id", "user_name", "rating", "time", "vote"],
        "output_dir": Path("output_short/run5&re"),
        "split_unit": "document",
        "hdbscan_min_cluster_size": 30,
        "hdbscan_min_samples": 15,
        "umap_n_neighbors": 15,
        "reuse_embeddings_from": Path("./output_short/run3") #不复用embedding结果的时候去除这一行
    },
    {
        "name": "書評",
        "csv_path": "../../data_cleaning2/text_cleaning/shuping_clean_base.csv",
        "content_col": "content",
        "id_cols": ["book_id", "book_name", "user_name", "rating", "time", "id",
                    "title", "useful_count", "useless_count", "reply", "page"],
        "output_dir": Path("output_long/run5&re"),
        "split_unit": "sentence_chunk",
        "hdbscan_min_cluster_size": 300,
        "hdbscan_min_samples": 20,
        "umap_n_neighbors": 20,
        "reuse_embeddings_from": Path("./output_long/run3") #不复用embedding结果的时候去除这一行
    },
]

TOP_N_WORDS = 10
RANDOM_STATE = 42
SINGLE_CHAR_RATIO_THRESHOLD = 0.3
N_REPRESENTATIVE_DOCS = 5

# --- テキスト分割の設定 ---
UNIT_MIN_CHARS = 2   # これより短いユニット(段落)は「意味情報が乏しい」として除外する。
                      # 除外件数は必ずメタデータに記録し、隠蔽しない。
                      # document単位の場合は通常この閾値に該当しない。

UMAP_FIXED_PARAMS = {"n_components": 5, "min_dist": 0.0, "metric": "cosine"}
HDBSCAN_FIXED_PARAMS = {"metric": "euclidean", "cluster_selection_method": "eom"}
# =================================


# ---------- 辞書・ストップワード ----------

def merge_stopwords(base_path, candidate_path, output_path):
    words = set()
    base_file = Path(base_path)
    if base_file.exists():
        with open(base_file, encoding="utf-8") as f:
            words.update(line.strip() for line in f if line.strip())
    else:
        print(f"[警告] 基礎ストップワード表が見つかりません: {base_path}")

    cand_file = Path(candidate_path)
    if cand_file.exists():
        with open(cand_file, encoding="utf-8") as f:
            words.update(line.strip() for line in f if line.strip() and not line.startswith("#"))
    else:
        print(f"[警告] 候補ストップワード表が見つかりません: {candidate_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for w in sorted(words):
            f.write(w + "\n")
    print(f"統合ストップワード表を生成: {output_path} (計 {len(words)} 語)")
    return words


def load_stopwords(path):
    with open(path, encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def file_md5(path):
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.md5(p.read_bytes()).hexdigest()


def build_tokenizer(stopwords):
    def tokenizer(text):
        tokens = jieba.cut(str(text), HMM=JIEBA_HMM_ENABLED)
        return [
            t.strip() for t in tokens
            if t.strip()
            and t.strip() not in stopwords
            and re.search(r"[\u4e00-\u9fffA-Za-z0-9]", t.strip())
        ]
    return tokenizer


# ---------- テキスト分割(document / sentence_chunk) ----------
# 注: 当初は改行コードによる段落分割(paragraph)を計画していたが、実データ検証の結果、
# 抓取スクリプト側のclean_text()が抓取時点で改行を全て空白に置換していたことが判明し、
# 段落境界情報は既存データから復元不可能であることが確認された。
# そのため、句読点による文分割 + 目標文字数(CHUNK_TARGET_CHARS)への文の累積結合という
# 「文境界を尊重した固定長チャンク分割」に方針変更した。
CHUNK_SENTENCE_SPLIT_PATTERN = r"(?<=[。！？…；\n])"
CHUNK_TARGET_CHARS = 150   # チャンクの目標文字数。文境界でのみ区切るため、実際の長さはこの値の前後で変動する。


def split_into_sentences_for_chunking(text):
    """chunk構築の下準備として、句読点(。！？…；改行)を境界に文を分割する。"""
    text = str(text)
    raw_sentences = re.split(CHUNK_SENTENCE_SPLIT_PATTERN, text)
    return [s.strip() for s in raw_sentences if s.strip()]


def merge_sentences_into_chunks(sentences, target_chars):
    """
    文を先頭から順に累積結合し、累積文字数が target_chars 以上になった時点でチャンクを確定する。
    文の途中で切ることは絶対にしない(常に文境界でチャンクを区切る)ため、
    各チャンクの実際の長さは target_chars ちょうどにはならず、その前後で変動する。
    文書末尾の残り(target_charsに満たない断片)も、内容を保持するため独立したチャンクとして残す。
    """
    chunks = []
    current = ""
    for s in sentences:
        current += s
        if len(current) >= target_chars:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def split_into_paragraphs(text):
    """[非推奨/未使用] 改行コードによる段落分割。実データに改行情報が存在しないため現在は使用しない。
    将来、段落構造を保持した形でデータを再取得した場合に備えて実装のみ残す。"""
    text = str(text)
    raw_paragraphs = re.split(r"\n+", text)
    return [p.strip() for p in raw_paragraphs if p.strip()]


def split_into_units(text, split_unit):
    """split_unit ("document" / "sentence_chunk") に応じて分割方式を切り替える。"""
    text = str(text).strip()
    if split_unit == "document":
        return [text] if text else []
    elif split_unit == "sentence_chunk":
        sentences = split_into_sentences_for_chunking(text)
        if not sentences:
            return []
        return merge_sentences_into_chunks(sentences, CHUNK_TARGET_CHARS)
    elif split_unit == "paragraph":
        raise ValueError(
            "paragraph単位は、改行情報が実データに存在しないことが確認されたため使用不可。"
            "'sentence_chunk' を使用してください。"
        )
    else:
        raise ValueError(
            f"未対応の split_unit: {split_unit}。"
            f"'document' または 'sentence_chunk' を指定してください。"
        )


def build_unit_corpus(df, content_col, id_cols, split_unit):
    """
    元の文書DataFrameを、指定された分析単位(document/sentence_chunk)に展開する。
    各ユニットについて、元の文書を特定するための original_doc_id(=元DataFrameのindex)、
    文書内でのユニット番号 unit_seq、および元のメタデータ列(id_cols)を保持する。
    極端に短いユニット(UNIT_MIN_CHARS未満)は除外し、除外件数を記録して返す。
    document単位の場合は分割自体が発生しないため、フィルタも通常は作用しない。
    """
    records = []
    n_total_units = 0
    n_dropped_short = 0

    for orig_idx, row in df.iterrows():
        units = split_into_units(row[content_col], split_unit)
        for seq, unit_text in enumerate(units):
            n_total_units += 1
            if len(unit_text) < UNIT_MIN_CHARS:
                n_dropped_short += 1
                continue
            rec = {
                "original_doc_id": orig_idx,
                "unit_seq": seq,
                "unit_text": unit_text,
            }
            for col in id_cols:
                if col in df.columns:
                    rec[col] = row[col]
            records.append(rec)

    df_units = pd.DataFrame(records)
    stats = {
        "split_unit": split_unit,
        "n_original_documents": len(df),
        "n_units_before_filtering": n_total_units,
        "n_units_dropped_too_short": n_dropped_short,
        "n_units_final": len(df_units),
        "unit_min_chars_threshold": UNIT_MIN_CHARS,
    }
    return df_units, stats


def load_corpus(csv_path, content_col, id_cols, split_unit):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[content_col]).reset_index(drop=True)
    df_units, split_stats = build_unit_corpus(df, content_col, id_cols, split_unit)
    docs = df_units["unit_text"].tolist()
    return df, df_units, docs, split_stats


# ---------- 環境・メタデータ ----------

def get_library_versions():
    libs = ["bertopic", "sentence-transformers", "umap-learn", "hdbscan",
            "scikit-learn", "jieba", "gensim", "pandas", "numpy", "torch"]
    versions = {}
    for lib in libs:
        try:
            versions[lib] = version(lib)
        except PackageNotFoundError:
            versions[lib] = "not found"
    return versions


def get_embedding_model_revision(model_name):
    try:
        from huggingface_hub import model_info
        info = model_info(model_name)
        return info.sha
    except Exception as e:
        print(f"[情報] embedding modelのrevision取得に失敗(必須ではありません): {e}")
        return None


def build_run_metadata(config, umap_params, hdbscan_params, split_stats):
    return {
        "corpus_name": config["name"],
        "csv_path": config["csv_path"],
        "unit_of_analysis": config["split_unit"],
        "split_stats": split_stats,
        "run_timestamp": datetime.datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "library_versions": get_library_versions(),
        "embedding_model": {
            "name": EMBEDDING_MODEL_NAME,
            "revision": get_embedding_model_revision(EMBEDDING_MODEL_NAME),
        },
        "jieba_hmm_enabled": JIEBA_HMM_ENABLED,
        "custom_dict_path": CUSTOM_DICT_PATH,
        "custom_dict_md5": file_md5(CUSTOM_DICT_PATH),
        "stopwords_final_path": MERGED_STOPWORDS_PATH,
        "stopwords_final_md5": file_md5(MERGED_STOPWORDS_PATH),
        "umap_params": umap_params,
        "hdbscan_params": hdbscan_params,
        "bertopic_params": {
            "top_n_words": TOP_N_WORDS,
            "calculate_probabilities": True,
            "representation_model": "c-TF-IDF only (MMR not applied)",
        },
        "random_state": RANDOM_STATE,
    }


# ---------- Topic Coherence ----------

def compute_coherence(topic_model, tokenized_docs, valid_topic_ids):
    import time
    t0 = time.time()
    print(f"  [Coherence] 対象文書数(フィルタ前): {len(tokenized_docs)}, 対象トピック数: {len(valid_topic_ids)}")

    # gensimのスライディングウィンドウ累加器は、トークン数0の文書が含まれると
    # IndexErrorで異常終了する(空文書に対する内部deque操作でindex out of boundsが発生するため)。
    # 分かち書き後にストップワード除去等で全トークンが除去された文書(=空文書)を除外する。
    # この除外は隠蔽せず、件数を明示的に記録する。
    n_before = len(tokenized_docs)
    tokenized_docs_nonempty = [d for d in tokenized_docs if len(d) > 0]
    n_empty_dropped = n_before - len(tokenized_docs_nonempty)
    if n_empty_dropped > 0:
        print(f"  [Coherence] [注意] 分かち書き後にトークンが0個になった文書を {n_empty_dropped} 件除外"
              f"(Coherence計算のみに影響。トピックモデリング自体の結果には影響しない)")
    tokenized_docs = tokenized_docs_nonempty

    dictionary = Dictionary(tokenized_docs)
    print(f"  [Coherence] 辞書構築完了(語彙数: {len(dictionary)}), {time.time()-t0:.1f}秒経過")

    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    print(f"  [Coherence] コーパス変換完了, {time.time()-t0:.1f}秒経過")

    topics_words = []
    topic_id_order = []
    for topic_id in valid_topic_ids:
        words = [w for w, _ in topic_model.get_topic(topic_id)]
        words = [w for w in words if w in dictionary.token2id]
        if len(words) < 2:
            continue
        topics_words.append(words)
        topic_id_order.append(topic_id)

    print(f"  [Coherence] CoherenceModel計算開始(processes=1で強制シングルプロセス実行)...")
    coherence_model = CoherenceModel(
        topics=topics_words, texts=tokenized_docs, corpus=corpus,
        dictionary=dictionary, coherence="c_v",
        processes=1,  # Windowsでのmultiprocessing起因のハング/大幅な遅延を回避するため明示的に単一プロセス化
    )
    per_topic_scores = coherence_model.get_coherence_per_topic()
    print(f"  [Coherence] トピック別スコア計算完了, {time.time()-t0:.1f}秒経過")

    overall_score = coherence_model.get_coherence()
    print(f"  [Coherence] 全体スコア計算完了, 合計 {time.time()-t0:.1f}秒")

    df_coherence = pd.DataFrame({"Topic": topic_id_order, "Coherence_c_v": per_topic_scores})
    return df_coherence, overall_score, n_empty_dropped


# ---------- 文書レベルへの集約 ----------

def aggregate_to_document_level(df_units, topics, probs, split_unit):
    """
    ユニット(文/段落/文書そのもの)レベルのトピック割当結果を、元の文書(評論)単位に集約する。
    RQ1の時系列分析等、文書=評論を分析単位とする後続処理のために必要。
    split_unit="document" の場合、1文書=1ユニットなので、この集約は実質的に
    topics列をそのまま転記するだけになる(それでも後続処理の入出力インタフェースを
    split_unitに依らず統一するために、常にこの関数を通す)。

    出力される情報:
    - 各評論内でのトピック分布(どのユニットが何個、どのトピックに属したか)
    - 主導トピック(最もユニット数の多いトピック。外れ値-1は除外して判定)
    - 外れ値ユニットの割合
    """
    df_units = df_units.copy()
    df_units["topic"] = topics

    agg_records = []
    for orig_doc_id, group in df_units.groupby("original_doc_id"):
        topic_counts = group["topic"].value_counts()
        n_units = len(group)

        non_outlier_counts = topic_counts[topic_counts.index != -1]
        if len(non_outlier_counts) > 0:
            dominant_topic = non_outlier_counts.idxmax()
            dominant_topic_ratio = non_outlier_counts.max() / n_units
        else:
            dominant_topic = -1
            dominant_topic_ratio = 0.0

        outlier_ratio = topic_counts.get(-1, 0) / n_units

        agg_records.append({
            "original_doc_id": orig_doc_id,
            "n_units": n_units,
            "dominant_topic": dominant_topic,
            "dominant_topic_ratio": round(dominant_topic_ratio, 3),
            "outlier_unit_ratio": round(outlier_ratio, 3),
            "topic_distribution": json.dumps(topic_counts.to_dict(), ensure_ascii=False),
        })

    return pd.DataFrame(agg_records)


# ---------- Step 1: パイプライン本体 ----------

def run_pipeline(config, embedding_model, tokenizer):
    split_unit = config["split_unit"]
    print(f"\n{'='*60}\n[{config['name']}] パイプライン開始(分析単位: {split_unit})\n{'='*60}")
    config["output_dir"].mkdir(exist_ok=True, parents=True)

    df_orig, df_units, docs, split_stats = load_corpus(
        config["csv_path"], config["content_col"], config["id_cols"], split_unit
    )
    print(f"元文書数: {split_stats['n_original_documents']}")
    print(f"分割後のユニット数(フィルタ前): {split_stats['n_units_before_filtering']}")
    print(f"短すぎるため除外したユニット数: {split_stats['n_units_dropped_too_short']}")
    print(f"分析対象のユニット数(最終): {split_stats['n_units_final']}")

    # embeddingの再利用: config内で "reuse_embeddings_from" が指定されている場合、
    # 既存run(またはgrid search cache)から embedding をそのまま読み込み、再計算をスキップする。
    # embeddingはembeddingモデルとテキスト内容(chunk分割方式)のみに依存し、
    # UMAP/HDBSCANのパラメータには依存しないため、パラメータ確定後の正式実行では
    # 再計算不要であることを前提としている。
    reuse_path = config.get("reuse_embeddings_from")
    if reuse_path is not None:
        reuse_path = Path(reuse_path)
        cached_emb_path = reuse_path / "unit_embeddings.npy"
        cached_units_path = reuse_path / "units_with_topics.csv"
        print(f"[embedding再利用] {reuse_path} から既存のembeddingを読み込みます...")
        embeddings = np.load(cached_emb_path)
        cached_docs = pd.read_csv(cached_units_path)["unit_text"].astype(str).tolist()

        # 安全確認: 今回生成したdocsと、再利用しようとしているembeddingのdocsが完全一致するか検証する。
        # 一致しない場合(データやchunk分割方式が変わっている場合)は、再計算するよう促すために停止する。
        if cached_docs != docs:
            raise ValueError(
                f"[embedding再利用エラー] {reuse_path} のdocsが現在のdocsと一致しません。"
                f"CSVデータやchunk分割設定が変更されている可能性があります。"
                f"reuse_embeddings_from を None にして再計算するか、正しいキャッシュ元を指定してください。"
            )
        print(f"検証OK: {len(docs)}件のdocsが完全一致。embeddingを再利用します(再計算スキップ)。")
    else:
        print("Embedding生成中...")
        embeddings = embedding_model.encode(docs, show_progress_bar=True)

    umap_params = {"n_neighbors": config["umap_n_neighbors"], "random_state": RANDOM_STATE, **UMAP_FIXED_PARAMS}
    umap_model = UMAP(**umap_params)

    hdbscan_params = {"min_cluster_size": config["hdbscan_min_cluster_size"], "prediction_data": True, **HDBSCAN_FIXED_PARAMS}
    if config.get("hdbscan_min_samples") is not None:
        hdbscan_params["min_samples"] = config["hdbscan_min_samples"]
    # min_samples未指定の場合、HDBSCANはmin_cluster_sizeと同値をデフォルトとして暗黙的に使用する。
    # この暗黙のデフォルト値も再現性のためmetadataに明示的に記録する。
    else:
        hdbscan_params["min_samples"] = config["hdbscan_min_cluster_size"]  # HDBSCANの暗黙デフォルトを明示化
    hdbscan_model = HDBSCAN(**hdbscan_params)

    vectorizer_model = CountVectorizer(tokenizer=tokenizer, token_pattern=None, lowercase=False)
    #representation_model = MaximalMarginalRelevance(diversity=0.2)  #加入MMR
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=False) #调整高频词权重

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
       #representation_model=representation_model,  #MMR调整词序
        ctfidf_model=ctfidf_model, #调整高频词权重
        top_n_words=TOP_N_WORDS,
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    df_units["topic"] = topics

    # --- 保存: ユニットレベル結果 ---
    topic_model.save(str(config["output_dir"] / "bertopic_model"), serialization="safetensors",
                      save_ctfidf=True, save_embedding_model=EMBEDDING_MODEL_NAME)
    np.save(config["output_dir"] / "topic_probabilities.npy", probs)
    np.save(config["output_dir"] / "unit_embeddings.npy", embeddings)
    df_units.to_csv(config["output_dir"] / "units_with_topics.csv", index=False, encoding="utf-8-sig")

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(config["output_dir"] / "topic_info.csv", index=False, encoding="utf-8-sig")

    outlier_ratio = (df_units["topic"] == -1).mean()
    print(f"トピック数(外れ値除く): {len(topic_info) - 1}")
    print(f"外れ値(-1)比率({split_unit}単位): {outlier_ratio:.2%}")

    # --- 保存: 文書レベルへの集約結果(RQ1時系列分析等で使用) ---
    df_doc_agg = aggregate_to_document_level(df_units, topics, probs, split_unit)
    # 元文書のメタデータ(time列等)を結合
    df_doc_agg = df_doc_agg.merge(
        df_orig.reset_index().rename(columns={"index": "original_doc_id"}),
        on="original_doc_id", how="left"
    )
    df_doc_agg.to_csv(config["output_dir"] / "document_topic_aggregation.csv", index=False, encoding="utf-8-sig")
    print(f"文書レベル集約結果を保存: {config['output_dir'] / 'document_topic_aggregation.csv'}")

    # --- Topic Coherence ---
    print("Topic Coherence (c_v) を計算中...")
    tokenized_docs = [tokenizer(d) for d in docs]
    valid_topic_ids = [t for t in topic_info["Topic"].tolist() if t != -1]
    df_coherence, overall_coherence, n_empty_dropped = compute_coherence(topic_model, tokenized_docs, valid_topic_ids)
    df_coherence.to_csv(config["output_dir"] / "topic_coherence.csv", index=False, encoding="utf-8-sig")
    print(f"全体Coherence(c_v)平均: {overall_coherence:.4f}")

    # --- メタデータ ---
    metadata = build_run_metadata(config, umap_params, hdbscan_params, split_stats)
    metadata["n_topics_excl_outlier"] = len(topic_info) - 1
    metadata["outlier_ratio_unit_level"] = round(outlier_ratio, 4)
    metadata["overall_coherence_c_v"] = round(overall_coherence, 4)
    metadata["coherence_calc_empty_docs_dropped"] = n_empty_dropped

    with open(config["output_dir"] / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"実行メタデータを保存: {config['output_dir'] / 'run_metadata.json'}")

    return topic_model, df_units, docs, tokenized_docs


# ---------- Step 2: 断片化/意味混在トピックの検出 ----------

def check_fragmented_topics(topic_model, df_units, docs, config):
    print(f"\n--- [{config['name']}] トピック品質チェック ---")
    split_unit = config["split_unit"]
    if split_unit == "document":
        print("[注意] 分析単位=document のため、以下の断片化検出(単一文字比率・拼接検出)は")
        print("       文/段落レベル用に設計されたヒューリスティックであり、長文書では")
        print("       偶然の語の隣接によって誤検出(false positive)が発生しやすい。")
        print("       Coherenceスコアおよびtopic_info.csvの実際のキーワードを優先して判断すること。")

    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()

    results = []
    for topic_id in valid_topics:
        word_score_pairs = topic_model.get_topic(topic_id)
        words = [w for w, _ in word_score_pairs]

        single_char_words = [w for w in words if len(w) == 1]
        single_char_ratio = len(single_char_words) / len(words) if words else 0

        topic_docs = [docs[i] for i, t in enumerate(topic_model.topics_) if t == topic_id]
        sample_docs = topic_docs[:50]

        concat_hits = []
        for i in range(len(words)):
            for j in range(len(words)):
                if i == j:
                    continue
                concat = words[i] + words[j]
                if len(concat) < 2:
                    continue
                hit_count = sum(1 for d in sample_docs if concat in d)
                if hit_count >= 3:
                    concat_hits.append(f"{words[i]}+{words[j]}(出現{hit_count}件)")

        flagged = (single_char_ratio > SINGLE_CHAR_RATIO_THRESHOLD) or (len(concat_hits) > 0)

        try:
            rep_docs = topic_model.get_representative_docs(topic_id)[:N_REPRESENTATIVE_DOCS]
        except Exception:
            rep_docs = sample_docs[:N_REPRESENTATIVE_DOCS]

        results.append({
            "Topic": topic_id,
            "Count": topic_info.loc[topic_info["Topic"] == topic_id, "Count"].values[0],
            "上位語": " / ".join(words),
            "単一文字比率": round(single_char_ratio, 3),
            "連結候補(分割疑い語)": " ; ".join(concat_hits) if concat_hits else "",
            "フラグ": "Y" if flagged else "",
            "代表文1": rep_docs[0] if len(rep_docs) > 0 else "",
            "代表文2": rep_docs[1] if len(rep_docs) > 1 else "",
            "代表文3": rep_docs[2] if len(rep_docs) > 2 else "",
        })

    df_check = pd.DataFrame(results)

    coherence_path = config["output_dir"] / "topic_coherence.csv"
    if coherence_path.exists():
        df_coherence = pd.read_csv(coherence_path)
        df_check = df_check.merge(df_coherence, on="Topic", how="left")

    df_check = df_check.sort_values(["フラグ", "Coherence_c_v"], ascending=[False, True])

    out_path = config["output_dir"] / "topic_quality_check.csv"
    df_check.to_csv(out_path, index=False, encoding="utf-8-sig")

    n_flagged = (df_check["フラグ"] == "Y").sum()
    print(f"検査対象トピック数: {len(df_check)}, 断片化フラグされたトピック数: {n_flagged}")
    print(f"結果を保存: {out_path}")

    return df_check

'''
# ---------- Step 2.5: 階層的トピック分析(hierarchical_topics) ----------

def run_hierarchical_topics(topic_model, docs, config):
    print(f"\n--- [{config['name']}] 階層的トピック分析 ---")

    hierarchical_topics = topic_model.hierarchical_topics(docs)
    hierarchical_topics.to_csv(
        config["output_dir"] / "hierarchical_topics.csv", index=False, encoding="utf-8-sig"
    )

    tree = topic_model.get_topic_tree(hierarchical_topics)
    tree_path = config["output_dir"] / "topic_tree.txt"
    with open(tree_path, "w", encoding="utf-8") as f:
        f.write(tree)
    print(f"トピック階層ツリーを保存: {tree_path}")

    try:
        fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        fig_path = config["output_dir"] / "hierarchy_dendrogram.html"
        fig.write_html(str(fig_path))
        print(f"階層デンドログラムを保存: {fig_path}")
    except Exception as e:
        print(f"[警告] デンドログラム生成に失敗(必須ではありません): {e}")

    return hierarchical_topics, tree
'''

# ---------- Step 3: 辞書更新ワークシート ----------

def export_dict_update_template(df_check, config):
    flagged = df_check[df_check["フラグ"] == "Y"].copy()
    if flagged.empty:
        print(f"[{config['name']}] フラグされたトピックはありませんでした。辞書更新は不要です。")
        return

    template_path = config["output_dir"] / "dict_update_worksheet.csv"
    cols = ["Topic", "Count", "上位語", "連結候補(分割疑い語)", "代表文1", "代表文2", "代表文3"]
    if "Coherence_c_v" in flagged.columns:
        cols.insert(2, "Coherence_c_v")
    flagged_out = flagged[cols].copy()
    flagged_out["追加すべき固有名詞(人手記入)"] = ""
    flagged_out["追加すべきストップワード(人手記入)"] = ""
    flagged_out["対応方針メモ"] = ""
    flagged_out.to_csv(template_path, index=False, encoding="utf-8-sig")

    print(f"[{config['name']}] 辞書更新用ワークシートを出力: {template_path}")


# ---------- メイン ----------

def main():
    if not Path(MERGED_STOPWORDS_PATH).exists():
        merge_stopwords(BASE_STOPWORDS_PATH, CANDIDATE_STOPWORDS_PATH, MERGED_STOPWORDS_PATH)
    else:
        print(f"既存の統合ストップワード表を使用: {MERGED_STOPWORDS_PATH}")

    stopwords = load_stopwords(MERGED_STOPWORDS_PATH)
    print(f"ストップワード数: {len(stopwords)}")

    if Path(CUSTOM_DICT_PATH).exists():
        jieba.load_userdict(CUSTOM_DICT_PATH)
        print(f"自定义词典を読み込みました: {CUSTOM_DICT_PATH}")
    else:
        print(f"[警告] 自定义词典が見つかりません: {CUSTOM_DICT_PATH}")

    tokenizer = build_tokenizer(stopwords)

    print("Embeddingモデルを読み込み中...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    for config in CORPUS_CONFIGS:
        topic_model, df_units, docs, tokenized_docs = run_pipeline(config, embedding_model, tokenizer)
        df_check = check_fragmented_topics(topic_model, df_units, docs, config)
        export_dict_update_template(df_check, config)
        #run_hierarchical_topics(topic_model, docs, config) 需要层次的时候再调用函数

    print("\n=== 全工程完了 ===")
    print("各 output_*/ 配下の主要ファイル:")
    print("  - run_metadata.json              : 全パラメータ・環境・辞書バージョンの記録")
    print("  - units_with_topics.csv           : 分析単位(document/sentence_chunk)レベルのトピック割当結果")
    print("  - document_topic_aggregation.csv  : 評論(文書)単位への集約結果(時系列分析用)")
    print("  - topic_info.csv                  : トピックごとの文数・上位語")
    print("  - topic_coherence.csv             : トピックごとのCoherence(c_v)")
    print("  - topic_quality_check.csv         : 断片化フラグ + Coherence 統合チェック表")
    print("  - dict_update_worksheet.csv       : 辞書更新用ワークシート(フラグ有トピックのみ)")
    print("  - bertopic_model/, *.npy          : モデル・埋め込み・確率分布(可視化を後で生成可能)")
    print("\n次のステップ:")
    print("1. topic_quality_check.csv でフラグ済み・低Coherenceのトピックを確認")
    print("2. dict_update_worksheet.csv を埋め、custom_dict.txt / stopwords_candidate.txt を更新")
    print("3. stopwords_final.txt を削除し、本スクリプトを再実行")
    print("4. run_metadata.json のoverall_coherence_c_vが安定するまで1-3を繰り返す")


if __name__ == "__main__":
    main()