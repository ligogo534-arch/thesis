from pathlib import Path
from collections import defaultdict
import json
import math
import pandas as pd
import numpy as np

# ============================================================
# 1. 路径与参数
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "sample_output" / "ABSA_stability_rerun_50.xlsx"
OUTPUT_FILE = BASE_DIR / "metrics_2.xlsx"

ID_COL = "ABSA_ID"
BOOK_COL = "book"
TYPE_COL = "语料来源"
FIRST_JSON_COL = "LLM_JSON解析"
SECOND_JSON_COL = "重跑_LLM_JSON解析"
FIRST_STATUS_COL = "LLM_JSON解析状态"
SECOND_STATUS_COL = "重跑_LLM_JSON解析状态"

# 按实际标注体系修改；当前默认A1-A9
ASPECTS = [f"A{i}" for i in range(1, 10)]
POLARITIES = ["正", "中立", "负"]

# ============================================================
# 2. 读取数据并检查列名
# ============================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"找不到输入文件：{INPUT_FILE}\n"
        "请确认脚本与 sample_output 文件夹的位置。"
    )

df = pd.read_excel(INPUT_FILE)
required_cols = [ID_COL, FIRST_JSON_COL, SECOND_JSON_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"缺少必要列：{missing}\n当前列名：{df.columns.tolist()}"
    )

df[ID_COL] = df[ID_COL].astype(str)
if df[ID_COL].duplicated().any():
    duplicated = df.loc[df[ID_COL].duplicated(keep=False), ID_COL].tolist()
    raise ValueError(f"{ID_COL}存在重复值，请先检查：{duplicated}")

print(f"读取样本数：{len(df)}")

# ============================================================
# 3. JSON解析及标准化
# ============================================================
def is_success_status(row, status_col):
    """若状态列不存在或为空，则不据此判为失败；明确非成功则判为失败。"""
    if status_col in row.index:
        status = row[status_col]
        if pd.notna(status) and str(status).strip() not in ("", "成功"):
            return False
    return True


def parse_absa_output(value, status_ok=True):
    """
    返回 (valid, records, error)。
    [] 是合法输出，表示模型没有识别出任何标注单元。
    允许同一 text_span 对应多个 aspect。
    """
    if not status_ok:
        return False, [], "API调用或解析状态非成功"

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False, [], "输出为空"

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return False, [], "输出为空"

        if value.startswith("```json"):
            value = value[len("```json"):].strip()
        elif value.startswith("```"):
            value = value[len("```"):].strip()
        if value.endswith("```"):
            value = value[:-3].strip()

        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            return False, [], f"JSON解析失败：{exc}"

    if not isinstance(value, list):
        return False, [], "JSON顶层结构不是列表"

    records = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            return False, [], f"第{i + 1}个元素不是对象"

        for key in ("text_span", "aspect", "polarity"):
            if key not in item:
                return False, [], f"第{i + 1}个元素缺少字段：{key}"

        span, aspect, polarity = (
            item["text_span"], item["aspect"], item["polarity"]
        )
        if not all(isinstance(x, str) for x in (span, aspect, polarity)):
            return False, [], f"第{i + 1}个元素字段类型错误"

        # 只规范化空白，不改写文本内容
        span = " ".join(span.split())
        aspect = aspect.strip()
        polarity = polarity.strip()

        if not span or not aspect or not polarity:
            return False, [], f"第{i + 1}个元素存在空字段"

        records.append({
            "text_span": span,
            "aspect": aspect,
            "polarity": polarity
        })

    return True, records, ""


# ============================================================
# 4. 逐条评论计算
# ============================================================
review_results = []
all_first_records = []
all_second_records = []
invalid_rows = []

for _, row in df.iterrows():
    sample_id = str(row[ID_COL])
    valid1, records1, err1 = parse_absa_output(
        row[FIRST_JSON_COL], is_success_status(row, FIRST_STATUS_COL)
    )
    valid2, records2, err2 = parse_absa_output(
        row[SECOND_JSON_COL], is_success_status(row, SECOND_STATUS_COL)
    )

    if not valid1 or not valid2:
        invalid_rows.append({
            ID_COL: sample_id,
            "第一次有效": valid1,
            "第二次有效": valid2,
            "第一次问题": err1,
            "第二次问题": err2
        })
        continue

    spans1 = {r["text_span"] for r in records1}
    spans2 = {r["text_span"] for r in records2}
    pairs1 = {(r["text_span"], r["aspect"]) for r in records1}
    pairs2 = {(r["text_span"], r["aspect"]) for r in records2}
    triples1 = {
        (r["text_span"], r["aspect"], r["polarity"]) for r in records1
    }
    triples2 = {
        (r["text_span"], r["aspect"], r["polarity"]) for r in records2
    }

    span_union, span_intersection = spans1 | spans2, spans1 & spans2
    span_jaccard = (
        len(span_intersection) / len(span_union) if span_union else np.nan
    )

    pair_union, pair_intersection = pairs1 | pairs2, pairs1 & pairs2
    aspect_jaccard = (
        len(pair_intersection) / len(pair_union) if pair_union else np.nan
    )

    # 共同Span上的Aspect集合完全一致率：
    # 仅以两次都检测到的Span为分母；比较每个Span的Aspect集合是否相同。
    map1 = defaultdict(set)
    map2 = defaultdict(set)
    for r in records1:
        map1[r["text_span"]].add(r["aspect"])
    for r in records2:
        map2[r["text_span"]].add(r["aspect"])

    shared_spans = set(map1) & set(map2)
    exact_aspect_span_count = sum(
        map1[span] == map2[span] for span in shared_spans
    )
    aspect_exact_span_rate = (
        exact_aspect_span_count / len(shared_spans)
        if shared_spans else np.nan
    )

    # 极性一致率：仅比较两次都识别到的(Span, Aspect)。
    polarity_map1 = defaultdict(set)
    polarity_map2 = defaultdict(set)
    for r in records1:
        polarity_map1[(r["text_span"], r["aspect"])].add(r["polarity"])
    for r in records2:
        polarity_map2[(r["text_span"], r["aspect"])].add(r["polarity"])

    shared_pairs = set(polarity_map1) & set(polarity_map2)
    polarity_exact_count = sum(
        polarity_map1[p] == polarity_map2[p] for p in shared_pairs
    )
    polarity_agreement = (
        polarity_exact_count / len(shared_pairs) if shared_pairs else np.nan
    )

    both_empty = not triples1 and not triples2
    review_results.append({
        ID_COL: sample_id,
        BOOK_COL: row.get(BOOK_COL, ""),
        TYPE_COL: row.get(TYPE_COL, ""),
        "第一次Span数": len(spans1),
        "第二次Span数": len(spans2),
        "匹配Span数": len(span_intersection),
        "Span_Jaccard": span_jaccard,
        "第一次Span_Aspect数": len(pairs1),
        "第二次Span_Aspect数": len(pairs2),
        "匹配Span_Aspect数": len(pair_intersection),
        "Aspect_Jaccard": aspect_jaccard,
        "共同Span数": len(shared_spans),
        "Aspect集合完全一致的共同Span数": exact_aspect_span_count,
        "共同Span上的Aspect完全一致率": aspect_exact_span_rate,
        "第一次三元组数": len(triples1),
        "第二次三元组数": len(triples2),
        "匹配三元组数": len(triples1 & triples2),
        "端到端完全一致": triples1 == triples2,
        "两次均无Span": both_empty,
        "Polarity共同Span_Aspect数": len(shared_pairs),
        "Polarity一致数": polarity_exact_count,
        "Polarity一致率": polarity_agreement,
    })

    for r in records1:
        all_first_records.append({ID_COL: sample_id, **r})
    for r in records2:
        all_second_records.append({ID_COL: sample_id, **r})

review_df = pd.DataFrame(review_results)
first_records_df = pd.DataFrame(
    all_first_records, columns=[ID_COL, "text_span", "aspect", "polarity"]
)
second_records_df = pd.DataFrame(
    all_second_records, columns=[ID_COL, "text_span", "aspect", "polarity"]
)
invalid_df = pd.DataFrame(invalid_rows)

print(f"两次输出均有效的评论：{len(review_df)}")
print(f"存在无效输出的评论：{len(invalid_df)}")

# ============================================================
# 5. 通用集合指标：微平均 Precision / Recall / F1
# ============================================================
def prf_from_sets(set1, set2):
    """第一次运行作为参考集合，第二次运行作为比较集合。"""
    matched = len(set1 & set2)
    precision = matched / len(set2) if len(set2) else np.nan
    recall = matched / len(set1) if len(set1) else np.nan
    if pd.notna(precision) and pd.notna(recall):
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )
    else:
        f1 = np.nan
    return {
        "第一次单元数": len(set1),
        "第二次单元数": len(set2),
        "匹配单元数": matched,
        "TP": matched,
        "FP": len(set2) - matched,
        "FN": len(set1) - matched,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


valid_ids = set(review_df[ID_COL].astype(str))


def make_key_set(records_df, key_cols):
    if records_df.empty:
        return set()
    temp = records_df[records_df[ID_COL].astype(str).isin(valid_ids)]
    return set(tuple(row) for row in temp[key_cols].itertuples(
        index=False, name=None
    ))


span_set1 = make_key_set(first_records_df, [ID_COL, "text_span"])
span_set2 = make_key_set(second_records_df, [ID_COL, "text_span"])
aspect_set1 = make_key_set(
    first_records_df, [ID_COL, "text_span", "aspect"]
)
aspect_set2 = make_key_set(
    second_records_df, [ID_COL, "text_span", "aspect"]
)
triple_set1 = make_key_set(
    first_records_df, [ID_COL, "text_span", "aspect", "polarity"]
)
triple_set2 = make_key_set(
    second_records_df, [ID_COL, "text_span", "aspect", "polarity"]
)

span_metrics = prf_from_sets(span_set1, span_set2)
aspect_metrics = prf_from_sets(aspect_set1, aspect_set2)
e2e_metrics = prf_from_sets(triple_set1, triple_set2)

# ============================================================
# 6. A1-A9各Aspect类别的独立指标
# ============================================================
# 每个类别分别以(ABSA_ID, text_span)为匹配单位。
# 同一个Span可以属于多个Aspect，各类别分别计入。
aspect_class_rows = []
observed_aspects = set(first_records_df["aspect"]) | set(second_records_df["aspect"])
aspect_labels = list(ASPECTS)
aspect_labels.extend(sorted(observed_aspects - set(aspect_labels)))

for aspect in aspect_labels:
    set1 = make_key_set(
        first_records_df[first_records_df["aspect"] == aspect],
        [ID_COL, "text_span"]
    )
    set2 = make_key_set(
        second_records_df[second_records_df["aspect"] == aspect],
        [ID_COL, "text_span"]
    )
    metrics = prf_from_sets(set1, set2)
    aspect_class_rows.append({
        "Aspect": aspect,
        **metrics
    })

aspect_class_metrics_df = pd.DataFrame(aspect_class_rows)

# ============================================================
# 7. Aspect变化明细（保留共同Span的集合比较，不生成笛卡尔积矩阵）
# ============================================================
aspect_changes = []
aspect_set_rows = []

for sample_id in valid_ids:
    r1 = first_records_df[first_records_df[ID_COL].astype(str) == sample_id]
    r2 = second_records_df[second_records_df[ID_COL].astype(str) == sample_id]

    map1 = defaultdict(set)
    map2 = defaultdict(set)
    for _, r in r1.iterrows():
        map1[r["text_span"]].add(r["aspect"])
    for _, r in r2.iterrows():
        map2[r["text_span"]].add(r["aspect"])

    for span in sorted(set(map1) & set(map2)):
        set_a1, set_a2 = map1[span], map2[span]
        aspect_set_rows.append({
            ID_COL: sample_id,
            "text_span": span,
            "第一次Aspect集合": ", ".join(sorted(set_a1)),
            "第二次Aspect集合": ", ".join(sorted(set_a2)),
            "Aspect集合完全一致": set_a1 == set_a2,
            "第一次Aspect数": len(set_a1),
            "第二次Aspect数": len(set_a2),
            "共同Aspect数": len(set_a1 & set_a2),
            "Aspect集合交集": ", ".join(sorted(set_a1 & set_a2)),
            "仅第一次存在的Aspect": ", ".join(sorted(set_a1 - set_a2)),
            "仅第二次存在的Aspect": ", ".join(sorted(set_a2 - set_a1)),
        })

        # 逐Aspect记录存在状态，便于追踪变化；这不是多分类混淆矩阵。
        for aspect in sorted(set_a1 | set_a2):
            aspect_changes.append({
                ID_COL: sample_id,
                "text_span": span,
                "Aspect": aspect,
                "第一次是否存在": aspect in set_a1,
                "第二次是否存在": aspect in set_a2,
                "状态是否一致": (aspect in set_a1) == (aspect in set_a2)
            })

aspect_changes_df = pd.DataFrame(aspect_changes)
aspect_set_details_df = pd.DataFrame(aspect_set_rows)

# 各Aspect二元存在/缺失的稳定性统计
aspect_presence_rows = []
for aspect in aspect_labels:
    cells = []
    for sample_id in valid_ids:
        r1 = first_records_df[first_records_df[ID_COL].astype(str) == sample_id]
        r2 = second_records_df[second_records_df[ID_COL].astype(str) == sample_id]
        m1 = defaultdict(set)
        m2 = defaultdict(set)
        for _, r in r1.iterrows():
            m1[r["text_span"]].add(r["aspect"])
        for _, r in r2.iterrows():
            m2[r["text_span"]].add(r["aspect"])
        for span in set(m1) & set(m2):
            cells.append((aspect in m1[span], aspect in m2[span]))

    tp = sum(a and b for a, b in cells)
    fp = sum((not a) and b for a, b in cells)
    fn = sum(a and (not b) for a, b in cells)
    tn = sum((not a) and (not b) for a, b in cells)
    n = len(cells)
    aspect_presence_rows.append({
        "Aspect": aspect,
        "共同Span数": n,
        "TP_两次均存在": tp,
        "FP_仅第二次存在": fp,
        "FN_仅第一次存在": fn,
        "TN_两次均不存在": tn,
        "二元状态一致率": (tp + tn) / n if n else np.nan
    })
aspect_presence_df = pd.DataFrame(aspect_presence_rows)

# ============================================================
# 8. Polarity一致性与混淆矩阵
# ============================================================
polarity_pairs = []
for sample_id in valid_ids:
    r1 = first_records_df[first_records_df[ID_COL].astype(str) == sample_id]
    r2 = second_records_df[second_records_df[ID_COL].astype(str) == sample_id]

    map1 = defaultdict(set)
    map2 = defaultdict(set)
    for _, r in r1.iterrows():
        map1[(r["text_span"], r["aspect"])].add(r["polarity"])
    for _, r in r2.iterrows():
        map2[(r["text_span"], r["aspect"])].add(r["polarity"])

    for pair in set(map1) & set(map2):
        p1, p2 = map1[pair], map2[pair]
        # 混淆矩阵要求每个匹配单位在每次运行中各有一个极性标签
        if len(p1) == 1 and len(p2) == 1:
            polarity_pairs.append({
                ID_COL: sample_id,
                "text_span": pair[0],
                "Aspect": pair[1],
                "第一次Polarity": next(iter(p1)),
                "第二次Polarity": next(iter(p2))
            })

polarity_pairs_df = pd.DataFrame(polarity_pairs)
if not polarity_pairs_df.empty:
    observed_polarities = (
        set(polarity_pairs_df["第一次Polarity"]) |
        set(polarity_pairs_df["第二次Polarity"])
    )
    polarity_labels = [p for p in POLARITIES if p in observed_polarities]
    polarity_labels += sorted(observed_polarities - set(polarity_labels))

    polarity_confusion = pd.crosstab(
        polarity_pairs_df["第一次Polarity"],
        polarity_pairs_df["第二次Polarity"],
        rownames=["第一次Polarity"],
        colnames=["第二次Polarity"]
    ).reindex(
        index=polarity_labels, columns=polarity_labels, fill_value=0
    )

    polarity_agreement_overall = (
        polarity_pairs_df["第一次Polarity"]
        == polarity_pairs_df["第二次Polarity"]
    ).mean()

    po = polarity_agreement_overall
    row_props = polarity_pairs_df["第一次Polarity"].value_counts(normalize=True)
    col_props = polarity_pairs_df["第二次Polarity"].value_counts(normalize=True)
    pe = sum(
        row_props.get(label, 0) * col_props.get(label, 0)
        for label in polarity_labels
    )
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
else:
    polarity_confusion = pd.DataFrame()
    polarity_agreement_overall = np.nan
    kappa = np.nan

# ============================================================
# 9. 总体指标汇总
# ============================================================
n_total = len(df)
n_valid = len(review_df)
n_both_empty = int(review_df["两次均无Span"].sum()) if n_valid else 0

summary_rows = []


def add_summary(section, metric, value, denominator=None):
    summary_rows.append({
        "评价层次": section,
        "指标": metric,
        "结果": value,
        "分母或样本数": denominator
    })


add_summary("输出完整性", "抽样评论总数", n_total, n_total)
add_summary("输出完整性", "两次输出均有效的评论数", n_valid, n_valid)
add_summary(
    "输出完整性", "两次输出均有效的评论比例",
    n_valid / n_total if n_total else np.nan, n_total
)
add_summary("输出完整性", "存在无效输出的评论数", len(invalid_df), n_total)
add_summary("输出完整性", "两次均无Span的评论数", n_both_empty, n_valid)

for section, metrics in [
    ("Span", span_metrics),
    ("Aspect总体", aspect_metrics),
    ("端到端", e2e_metrics)
]:
    for metric in [
        "第一次单元数", "第二次单元数", "匹配单元数",
        "TP", "FP", "FN", "Precision", "Recall", "F1"
    ]:
        add_summary(section, metric, metrics[metric])

add_summary(
    "Span", "评论级Jaccard均值（两次均为空的评论不纳入）",
    review_df["Span_Jaccard"].mean(),
    review_df["Span_Jaccard"].notna().sum()
)
add_summary(
    "Aspect总体", "Aspect集合Jaccard均值",
    review_df["Aspect_Jaccard"].mean(),
    review_df["Aspect_Jaccard"].notna().sum()
)
add_summary(
    "Aspect总体", "共同Span上的Aspect集合完全一致率（先逐评论计算再取均值）",
    review_df["共同Span上的Aspect完全一致率"].mean(),
    review_df["共同Span上的Aspect完全一致率"].notna().sum()
)
add_summary(
    "Aspect总体", "共同Span上的Aspect集合完全一致率（合并全部共同Span）",
    (
        aspect_set_details_df["Aspect集合完全一致"].mean()
        if not aspect_set_details_df.empty else np.nan
    ),
    len(aspect_set_details_df)
)
add_summary(
    "Polarity", "共同Span-Aspect上的极性一致率",
    polarity_agreement_overall, len(polarity_pairs_df)
)
add_summary("Polarity", "Cohen's Kappa", kappa, len(polarity_pairs_df))
add_summary(
    "端到端", "评论级完全一致率（含两次均为空）",
    review_df["端到端完全一致"].mean() if n_valid else np.nan, n_valid
)
nonempty_reviews = review_df[~review_df["两次均无Span"]]
add_summary(
    "端到端", "评论级完全一致率（排除两次均为空）",
    nonempty_reviews["端到端完全一致"].mean()
    if len(nonempty_reviews) else np.nan,
    len(nonempty_reviews)
)

summary_df = pd.DataFrame(summary_rows)

# ============================================================
# 10. 按书籍与投稿类型分组
# ============================================================
group_rows = []
if n_valid and BOOK_COL in review_df.columns and TYPE_COL in review_df.columns:
    for (book, source), group in review_df.groupby(
        [BOOK_COL, TYPE_COL], dropna=False
    ):
        group_rows.append({
            "book": book,
            "语料来源": source,
            "有效评论数": len(group),
            "Span_Jaccard均值": group["Span_Jaccard"].mean(),
            "Aspect_Jaccard均值": group["Aspect_Jaccard"].mean(),
            "共同Span上的Aspect完全一致率": (
                group["共同Span上的Aspect完全一致率"].mean()
            ),
            "Polarity一致率": group["Polarity一致率"].mean(),
            "评论级完全一致率": group["端到端完全一致"].mean(),
            "两次均无Span数": int(group["两次均无Span"].sum())
        })
group_df = pd.DataFrame(group_rows)

# ============================================================
# 11. 保存结果
# ============================================================
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="总体指标", index=False)
    review_df.to_excel(writer, sheet_name="逐条评论指标", index=False)
    group_df.to_excel(writer, sheet_name="书籍与投稿类型", index=False)
    aspect_class_metrics_df.to_excel(
        writer, sheet_name="Aspect分类指标A1-A9", index=False
    )
    aspect_presence_df.to_excel(
        writer, sheet_name="Aspect二元一致性", index=False
    )
    aspect_set_details_df.to_excel(
        writer, sheet_name="共同Span的Aspect集合", index=False
    )
    aspect_changes_df.to_excel(
        writer, sheet_name="Aspect存在状态明细", index=False
    )
    polarity_confusion.to_excel(writer, sheet_name="Polarity混淆矩阵")
    polarity_pairs_df.to_excel(
        writer, sheet_name="Polarity匹配明细", index=False
    )
    invalid_df.to_excel(writer, sheet_name="无效输出", index=False)
    first_records_df.to_excel(writer, sheet_name="第一次标注单元", index=False)
    second_records_df.to_excel(writer, sheet_name="第二次标注单元", index=False)

print("\n" + "=" * 68)
print("LLM-ABSA重复输出稳定性指标计算完成")
print("=" * 68)
print(summary_df.to_string(index=False))
print("\nA1-A9分类指标：")
print(aspect_class_metrics_df.to_string(index=False))
print(f"\n结果文件：{OUTPUT_FILE}")
print("=" * 68)