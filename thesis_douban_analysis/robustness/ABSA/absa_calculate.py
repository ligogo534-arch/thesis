
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

# ============================================================
# 1. 文件路径设置
# ============================================================

INPUT_FILE = Path("absa_matching.xlsx")
SHEET_NAME = "LLM_Spans"
OUTPUT_FILE = Path("absa_evaluation_results.xlsx")


# ============================================================
# 2. 读取数据与检查列名
# ============================================================

df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

required_cols = [
    "ABSA_ID",
    "LLM_Span_ID",
    "LLM_Span",
    "LLM_Aspect",
    "LLM_Polarity",
    "Human_Span_ID",
    "Human_Span",
    "HUMAN_Aspect",
    "HUMAN_Polarity",
    "Span一致",
    "Aspect一致",
    "Polarity一致",
]

missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"缺少以下列：{missing_cols}")


# 去除 Span ID 两端空格，空字符串视为缺失
id_cols = ["LLM_Span_ID", "Human_Span_ID"]

for col in id_cols:
    df[col] = df[col].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    df[col] = df[col].replace("", pd.NA)


def has_value(value):
    """判断单元格是否有有效值。"""
    return pd.notna(value) and str(value).strip() != ""


def to_flag(value):
    """将一致性标记转换为 1/0，无法识别的值返回 NA。"""
    if not has_value(value):
        return pd.NA

    s = str(value).strip().lower()

    if s in {"1", "1.0", "是", "一致", "yes", "true"}:
        return 1

    if s in {"0", "0.0", "否", "不一致", "no", "false"}:
        return 0

    return pd.NA


for col in ["Span一致", "Aspect一致", "Polarity一致"]:
    df[col] = df[col].apply(to_flag).astype("Int64")


# ============================================================
# 3. 检查一对一匹配
# ============================================================

# Span ID 全局唯一，因此直接检查 ID 是否重复
duplicate_llm = df.loc[
    df["LLM_Span_ID"].notna()
    & df["LLM_Span_ID"].duplicated(keep=False),
    "LLM_Span_ID",
].dropna().unique()

duplicate_human = df.loc[
    df["Human_Span_ID"].notna()
    & df["Human_Span_ID"].duplicated(keep=False),
    "Human_Span_ID",
].dropna().unique()

if len(duplicate_llm) or len(duplicate_human):
    raise ValueError(
        "发现重复的 Span ID，请先检查一对一匹配关系。\n"
        f"重复 LLM_Span_ID：{list(duplicate_llm)}\n"
        f"重复 Human_Span_ID：{list(duplicate_human)}"
    )


has_llm = df["LLM_Span_ID"].notna()
has_human = df["Human_Span_ID"].notna()

# 两侧均有 ID 且 Span一致=1，视为匹配成功
matched = has_llm & has_human & df["Span一致"].eq(1)

# 未成功匹配的 LLM Span 计入 FP
llm_unmatched = has_llm & ~matched

# 未成功匹配的人工 Span 计入 FN
human_unmatched = has_human & ~matched

matched_df = df.loc[matched].copy()


# 分类评价：只使用已匹配且人工标签有效的样本
aspect_valid = matched_df["HUMAN_Aspect"].apply(has_value)

polarity_valid = matched_df["HUMAN_Polarity"].apply(has_value)

if not aspect_valid.all():
    print(
        f"警告：{int((~aspect_valid).sum())} 个匹配 Span "
        "缺少人工 Aspect 标签，将不参与 Aspect 分类评价。"
    )

if not polarity_valid.all():
    print(
        f"警告：{int((~polarity_valid).sum())} 个匹配 Span "
        "缺少人工 Polarity 标签，将不参与极性分类评价。"
    )


# ============================================================
# 4. 指标计算函数
# ============================================================

def prf(tp, fp, fn):
    """根据 TP、FP、FN 计算 Precision、Recall、F1。"""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


summary = []
class_results = []


def add_summary(
    name,
    n,
    accuracy=None,
    p=None,
    r=None,
    f1=None,
    tp=None,
    fp=None,
    fn=None,
    macro_p=None,
    macro_r=None,
    macro_f1=None,
    weighted_p=None,
    weighted_r=None,
    weighted_f1=None,
):
    """
    汇总表：
    Precision / Recall / F1 与 TP / FP / FN 对应。
    分类任务的 P/R/F1 使用 Micro 平均。
    """
    summary.append({
        "评价任务": name,
        "样本数": n,
        "Accuracy": accuracy,
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Macro_Precision": macro_p,
        "Macro_Recall": macro_r,
        "Macro_F1": macro_f1,
        "Weighted_Precision": weighted_p,
        "Weighted_Recall": weighted_r,
        "Weighted_F1": weighted_f1,
    })


def add_class_metrics(task_name, y_true, y_pred):
    """
    输出各类别及 Macro / Micro / Weighted 指标。
    所有指标均基于同一组真实标签和预测标签。
    """
    if len(y_true) == 0:
        return

    labels = sorted(set(y_true) | set(y_pred))

    # 各类别指标
    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    for label, pi, ri, fi, si in zip(labels, p, r, f1, support):
        class_results.append({
            "评价任务": task_name,
            "类别": label,
            "Precision": pi,
            "Recall": ri,
            "F1": fi,
            "Support": si,
        })

    # Macro / Micro / Weighted 平均指标
    for average in ["macro", "micro", "weighted"]:
        pi, ri, fi, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=average,
            zero_division=0,
        )

        class_results.append({
            "评价任务": task_name,
            "类别": average.capitalize(),
            "Precision": pi,
            "Recall": ri,
            "F1": fi,
            "Support": len(y_true),
        })


def get_multiclass_counts(y_true, y_pred):
    """
    单标签多分类的 Micro 计数：
    TP = 预测正确的样本数；
    FP = 预测错误的样本数；
    FN = 预测错误的样本数。

    因而 Micro Precision、Recall、F1 与这些计数严格对应。
    """
    tp = sum(
        true == pred
        for true, pred in zip(y_true, y_pred)
    )

    total = len(y_true)
    fp = total - tp
    fn = total - tp

    return tp, fp, fn


# ============================================================
# 5. Span 检测
# ============================================================

# Span 检测不计算 Accuracy
span_tp = int(matched.sum())
span_fp = int(llm_unmatched.sum())
span_fn = int(human_unmatched.sum())

span_p, span_r, span_f1 = prf(
    span_tp, span_fp, span_fn
)

add_summary(
    name="Span检测",
    n=None,  # 不使用含义不明确的样本数
    p=span_p,
    r=span_r,
    f1=span_f1,
    tp=span_tp,
    fp=span_fp,
    fn=span_fn,
)

# ============================================================
# 6. Aspect 分类
# ============================================================

aspect_df = matched_df.loc[aspect_valid].copy()

if len(aspect_df):
    y_aspect_true = (
        aspect_df["HUMAN_Aspect"].astype(str).tolist()
    )
    y_aspect_pred = (
        aspect_df["LLM_Aspect"].astype(str).tolist()
    )

    aspect_acc = accuracy_score(
        y_aspect_true, y_aspect_pred
    )

    # Micro TP / FP / FN
    aspect_tp, aspect_fp, aspect_fn = get_multiclass_counts(
        y_aspect_true, y_aspect_pred
    )

    # 与 Micro TP / FP / FN 完全对应
    aspect_micro_p, aspect_micro_r, aspect_micro_f1 = prf(
        aspect_tp, aspect_fp, aspect_fn
    )

    # Macro 指标
    aspect_macro_p, aspect_macro_r, aspect_macro_f1, _ = (
        precision_recall_fscore_support(
            y_aspect_true,
            y_aspect_pred,
            average="macro",
            zero_division=0,
        )
    )

    # Weighted 指标
    aspect_weighted_p, aspect_weighted_r, aspect_weighted_f1, _ = (
        precision_recall_fscore_support(
            y_aspect_true,
            y_aspect_pred,
            average="weighted",
            zero_division=0,
        )
    )

    add_summary(
        name="Aspect分类（匹配样本）",
        n=len(aspect_df),
        accuracy=aspect_acc,
        p=aspect_micro_p,
        r=aspect_micro_r,
        f1=aspect_micro_f1,
        tp=aspect_tp,
        fp=aspect_fp,
        fn=aspect_fn,
        macro_p=aspect_macro_p,
        macro_r=aspect_macro_r,
        macro_f1=aspect_macro_f1,
        weighted_p=aspect_weighted_p,
        weighted_r=aspect_weighted_r,
        weighted_f1=aspect_weighted_f1,
    )

    add_class_metrics(
        "Aspect分类",
        y_aspect_true,
        y_aspect_pred,
    )

else:
    print("警告：没有可用于 Aspect 分类评价的匹配样本。")


# ============================================================
# 7. 情感极性分类
# ============================================================

polarity_df = matched_df.loc[polarity_valid].copy()

if len(polarity_df):
    y_pol_true = (
        polarity_df["HUMAN_Polarity"].astype(str).tolist()
    )
    y_pol_pred = (
        polarity_df["LLM_Polarity"].astype(str).tolist()
    )

    polarity_acc = accuracy_score(
        y_pol_true, y_pol_pred
    )

    # Micro TP / FP / FN
    polarity_tp, polarity_fp, polarity_fn = (
        get_multiclass_counts(
            y_pol_true, y_pol_pred
        )
    )

    # 与 Micro TP / FP / FN 完全对应
    polarity_micro_p, polarity_micro_r, polarity_micro_f1 = prf(
        polarity_tp, polarity_fp, polarity_fn
    )

    # Macro 指标
    polarity_macro_p, polarity_macro_r, polarity_macro_f1, _ = (
        precision_recall_fscore_support(
            y_pol_true,
            y_pol_pred,
            average="macro",
            zero_division=0,
        )
    )

    # Weighted 指标
    (
        polarity_weighted_p,
        polarity_weighted_r,
        polarity_weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        y_pol_true,
        y_pol_pred,
        average="weighted",
        zero_division=0,
    )

    add_summary(
        name="极性分类（匹配样本）",
        n=len(polarity_df),
        accuracy=polarity_acc,
        p=polarity_micro_p,
        r=polarity_micro_r,
        f1=polarity_micro_f1,
        tp=polarity_tp,
        fp=polarity_fp,
        fn=polarity_fn,
        macro_p=polarity_macro_p,
        macro_r=polarity_macro_r,
        macro_f1=polarity_macro_f1,
        weighted_p=polarity_weighted_p,
        weighted_r=polarity_weighted_r,
        weighted_f1=polarity_weighted_f1,
    )

    add_class_metrics(
        "极性分类",
        y_pol_true,
        y_pol_pred,
    )

else:
    print("警告：没有可用于极性分类评价的匹配样本。")


# ============================================================
# 8. 端到端评价
# ============================================================

# 端到端 TP 要求：
# 1. Span 匹配成功；
# 2. Aspect 正确；
# 3. Polarity 正确。

e2e_correct = (
    matched
    & df["LLM_Aspect"].apply(has_value)
    & df["HUMAN_Aspect"].apply(has_value)
    & df["LLM_Polarity"].apply(has_value)
    & df["HUMAN_Polarity"].apply(has_value)
    & df["LLM_Aspect"].eq(df["HUMAN_Aspect"])
    & df["LLM_Polarity"].eq(df["HUMAN_Polarity"])
)

e2e_tp = int(e2e_correct.sum())

# 有 LLM Span，但未实现完全正确的抽取，计入 FP
e2e_fp = int((has_llm & ~e2e_correct).sum())

# 有人工 Span，但未被完全正确地抽取，计入 FN
e2e_fn = int((has_human & ~e2e_correct).sum())

e2e_p, e2e_r, e2e_f1 = prf(
    e2e_tp, e2e_fp, e2e_fn
)

add_summary(
    name="端到端评价",
    n=int(has_llm.sum()),
    p=e2e_p,
    r=e2e_r,
    f1=e2e_f1,
    tp=e2e_tp,
    fp=e2e_fp,
    fn=e2e_fn,
)


# ============================================================
# 9. 输出结果到 Excel
# ============================================================

summary_df = pd.DataFrame(summary)
class_df = pd.DataFrame(class_results)

# 为 Span 检测增加单独的数量统计列
summary_df["LLM预测Span数"] = pd.NA
summary_df["人工标注Span数"] = pd.NA
summary_df["匹配成功Span数"] = pd.NA

span_row = summary_df["评价任务"] == "Span检测"

summary_df.loc[span_row, "LLM预测Span数"] = int(has_llm.sum())
summary_df.loc[span_row, "人工标注Span数"] = int(has_human.sum())
summary_df.loc[span_row, "匹配成功Span数"] = span_tp

# 输出便于复核的匹配结果
review_df = df.copy()

review_df["实际Span匹配"] = matched.astype(int)

review_df["Aspect标签正确"] = (
    review_df["LLM_Aspect"].eq(review_df["HUMAN_Aspect"])
    & review_df["LLM_Aspect"].apply(has_value)
    & review_df["HUMAN_Aspect"].apply(has_value)
).astype(int)

review_df["极性标签正确"] = (
    review_df["LLM_Polarity"].eq(review_df["HUMAN_Polarity"])
    & review_df["LLM_Polarity"].apply(has_value)
    & review_df["HUMAN_Polarity"].apply(has_value)
).astype(int)

review_df["端到端完全正确"] = e2e_correct.astype(int)

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl",
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="总体指标",
        index=False,
    )

    class_df.to_excel(
        writer,
        sheet_name="分类别指标",
        index=False,
    )

    review_df.to_excel(
        writer,
        sheet_name="匹配结果复核",
        index=False,
    )


# ============================================================
# 10. 输出控制台结果
# ============================================================

print("\n===== 总体评价指标 =====")
print(summary_df.round(4).to_string(index=False))

print("\n===== 分类别评价指标 =====")
print(class_df.round(4).to_string(index=False))

print(f"\n结果已保存至：{OUTPUT_FILE.resolve()}")
