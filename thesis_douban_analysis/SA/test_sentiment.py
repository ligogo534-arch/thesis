from pathlib import Path
import random
import pandas as pd
import jieba

# ============================================================
# 1. 路径
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

DUTIR_FILE = BASE_DIR / "word_list" / "DUTIR" / "cleaned" / "DUTIR_clean.txt"
DEGREE_FILE = BASE_DIR / "word_list" / "degree_words.txt"
NEGATION_FILE = BASE_DIR / "word_list" / "negation_words.txt"

SHORT_FILE = BASE_DIR.parent / "data_cleaning2" / "text_cleaning" / "duanping_clean_base.csv"
LONG_FILE = BASE_DIR.parent / "data_cleaning2" / "text_cleaning" / "shuping_clean_base.csv"

OUTPUT_DIR = BASE_DIR / "validation_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORD_DETAIL_FILE = OUTPUT_DIR / "validation_word_detail.csv"
REVIEW_RESULT_FILE = OUTPUT_DIR / "validation_review_result.csv"

SAMPLE_SIZE = 10
RANDOM_STATE = 42
NEGATION_WINDOW = 3
DEGREE_WINDOW = 3

# ============================================================
# 2. 读取 DUTIR
# ============================================================

def load_dutir(path):
    dutir = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                print(f"⚠ DUTIR第{line_no}行格式异常：{line}")
                continue
            word, emotion, intensity, polarity = parts[:4]
            try:
                intensity = float(intensity)
                polarity = int(float(polarity))
            except ValueError:
                print(f"⚠ DUTIR第{line_no}行数值异常：{line}")
                continue
            dutir[word] = {"emotion": emotion, "intensity": intensity, "polarity": polarity}
    return dutir


# ============================================================
# 3. 读取程度词表
# ============================================================

def load_degree_words(path):
    degree_words = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"⚠ 程度词表第{line_no}行格式异常：{line}")
                continue
            word = parts[0].strip()
            try:
                weight = float(parts[1].strip())
            except ValueError:
                print(f"⚠ 程度词表第{line_no}行权重异常：{line}")
                continue
            degree_words[word] = weight
    return degree_words


# ============================================================
# 4. 读取否定词表
# ============================================================

def load_negation_words(path):
    negation_words = set()
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            word = line.strip()
            if word:
                negation_words.add(word)
    return negation_words


# ============================================================
# 5. 读取评论
# ============================================================

def load_reviews(path, data_type):
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["book_id", "time", "content"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{data_type}缺少列：{missing}")
    df = df.dropna(subset=["content"]).copy()
    df["content"] = df["content"].astype(str)
    df["data_type"] = data_type
    df["source_index"] = df.index
    return df


# ============================================================
# 6. 计算单条评论
# ============================================================

def analyze_review(content, dutir, degree_words, negation_words):
    tokens = list(jieba.cut(content, cut_all=False))
    tokens = [token.strip() for token in tokens if token.strip()]

    emotion_scores = {"乐": 0.0, "好": 0.0, "怒": 0.0, "哀": 0.0, "惧": 0.0, "恶": 0.0, "惊": 0.0}
    word_details = []
    total_positive = 0.0
    total_negative = 0.0

    emotion_map = {"PA": "乐", "PE": "乐", "PD": "好", "PH": "好", "PG": "好", "PB": "好", "PK": "好", "NA": "怒",
                   "NB": "哀", "NJ": "哀", "NH": "哀", "PF": "哀", "NI": "惧", "NC": "惧", "NG": "惧", "NE": "恶",
                   "ND": "恶", "NN": "恶", "NK": "恶", "NL": "恶", "PC": "惊"}

    for i, token in enumerate(tokens):
        if token not in dutir:
            continue

        info = dutir[token]
        emotion_code = info["emotion"]
        if emotion_code not in emotion_map:
            continue

        emotion_category = emotion_map[emotion_code]
        intensity = info["intensity"]
        polarity = info["polarity"]

        start = max(0, i - NEGATION_WINDOW)
        previous_tokens = tokens[start:i]

        negation_hits = [w for w in previous_tokens if w in negation_words]
        negation_count = len(negation_hits)
        negation_sign = -1 if negation_count % 2 == 1 else 1

        degree_word = ""
        degree_weight = 1.0
        degree_distance = None

        degree_candidates = []
        degree_start = max(0, i - DEGREE_WINDOW)

        for j in range(i - 1, degree_start - 1, -1):
            if tokens[j] in degree_words:
                degree_candidates.append((j, tokens[j], degree_words[tokens[j]]))
                break

        if degree_candidates:
            j, degree_word, degree_weight = degree_candidates[0]
            degree_distance = i - j

        score = intensity * degree_weight * negation_sign

        emotion_scores[emotion_category] += score

        if polarity == 1:
            total_positive += score
        elif polarity == 2:
            total_negative += score

        word_details.append({
            "感情词": token,
            "DUTIR情感分类": emotion_code,
            "情感类别": emotion_category,
            "DUTIR强度": intensity,
            "DUTIR极性": polarity,
            "程度词": degree_word,
            "程度权重": degree_weight,
            "程度词距离": degree_distance,
            "否定词": " / ".join(negation_hits),
            "否定词数量": negation_count,
            "否定符号": negation_sign,
            "调整后情感分数": score
        })

    length = len(content)

    print("\nDEBUG emotion_scores:", emotion_scores)
    print("DEBUG length:", length)
    print("DEBUG 好 raw score:", emotion_scores["好"])
    print("DEBUG 好 normalized:", emotion_scores["好"] / length)

    if length > 0:
        normalized_scores = {k: v / length for k, v in emotion_scores.items()}
        polarity_score = (total_positive - total_negative) / length
    else:
        normalized_scores = {k: 0.0 for k in emotion_scores}
        polarity_score = 0.0

    return tokens, word_details, normalized_scores, polarity_score


# ============================================================
# 7. 主程序
# ============================================================

print("=" * 70)
print("DUTIR 情感分析验证程序")
print("=" * 70)

print("\n正在读取词典...")

dutir = load_dutir(DUTIR_FILE)
degree_words = load_degree_words(DEGREE_FILE)
negation_words = load_negation_words(NEGATION_FILE)

print(f"DUTIR词语数：{len(dutir):,}")
print(f"程度词数：{len(degree_words):,}")
print(f"否定词数：{len(negation_words):,}")

print("\n正在读取评论数据...")

short_df = load_reviews(SHORT_FILE, "短評")
long_df = load_reviews(LONG_FILE, "書評")

print(f"短评数量：{len(short_df):,}")
print(f"书评数量：{len(long_df):,}")


# ============================================================
# 8. 合并并随机抽取10条
# ============================================================

short_available = len(short_df)
long_available = len(long_df)

all_reviews = pd.concat([short_df, long_df], ignore_index=True)

sample_n = min(SAMPLE_SIZE, len(all_reviews))

sample_df = all_reviews.sample(
    n=sample_n,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("\n" + "=" * 70)
print(f"随机抽取 {sample_n} 条评论进行验证")
print("=" * 70)

print("\n样本构成：")
print(sample_df["data_type"].value_counts())


# ============================================================
# 9. 逐条计算
# ============================================================

all_word_details = []
all_review_results = []

for sample_id, row in sample_df.iterrows():

    content = row["content"]
    data_type = row["data_type"]

    print("\n" + "-" * 70)
    print(f"样本 {sample_id + 1}/{sample_n}")
    print(f"数据类型：{data_type}")
    print(f"book_id：{row['book_id']}")
    print(f"时间：{row['time']}")
    print(f"原文：{content}")

    tokens, word_details, normalized_scores, polarity_score = analyze_review(content, dutir, degree_words, negation_words)

    print(f"\njieba分词：{' / '.join(tokens)}")

    if word_details:
        print("\n识别出的DUTIR情感词：")

        for detail in word_details:
            print(
                f"  {detail['感情词']} → "
                f"{detail['DUTIR情感分类']} / "
                f"强度={detail['DUTIR强度']} / "
                f"极性={detail['DUTIR极性']} / "
                f"程度词={detail['程度词'] or '无'} / "
                f"D={detail['程度权重']} / "
                f"否定词={detail['否定词'] or '无'} / "
                f"N={detail['否定符号']} / "
                f"S={detail['调整后情感分数']:.4f}"
            )

            detail["sample_id"] = sample_id + 1
            detail["data_type"] = data_type
            detail["book_id"] = row["book_id"]
            detail["time"] = row["time"]
            all_word_details.append(detail)

    else:
        print("\n⚠ 未识别到DUTIR情感词。")

    print("\n评论级结果：")

    for category, value in normalized_scores.items():
        print(f"  {category}: {value:.6f}")

    print(f"  整体极性：{polarity_score:.6f}")

    result = {
        "sample_id": sample_id + 1,
        "data_type": data_type,
        "book_id": row["book_id"],
        "time": row["time"],
        "content": content,
        "文字数": len(content),
        "DUTIR情感词数": len(word_details),
        "乐": normalized_scores["乐"],
        "好": normalized_scores["好"],
        "怒": normalized_scores["怒"],
        "哀": normalized_scores["哀"],
        "惧": normalized_scores["惧"],
        "恶": normalized_scores["恶"],
        "惊": normalized_scores["惊"],
        "整体极性": polarity_score
    }

    all_review_results.append(result)


# ============================================================
# 10. 输出逐词结果
# ============================================================

word_detail_df = pd.DataFrame(all_word_details)

if not word_detail_df.empty:
    word_detail_df.to_csv(
        WORD_DETAIL_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n✓ 逐词计算结果：{WORD_DETAIL_FILE}")
else:
    print("\n⚠ 没有识别到任何DUTIR情感词，未生成逐词结果。")


# ============================================================
# 11. 输出评论级结果
# ============================================================

review_result_df = pd.DataFrame(all_review_results)

review_result_df.to_csv(
    REVIEW_RESULT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"✓ 评论级计算结果：{REVIEW_RESULT_FILE}")


# ============================================================
# 12. 最终结束
# ============================================================

print("\n" + "=" * 70)
print("验证完成")
print("=" * 70)

print(f"""
验证样本数：{sample_n}

输出文件：
1. {WORD_DETAIL_FILE}
2. {REVIEW_RESULT_FILE}

请重点人工检查：
① jieba分词是否合理
② DUTIR情感词识别是否合理
③ 程度词是否正确匹配
④ 否定词是否正确匹配
⑤ S_i是否计算正确
⑥ 七类情感归类是否正确
⑦ 整体极性是否符合预期
""")