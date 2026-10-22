import json
import pandas as pd
from pathlib import Path

# 输入、输出文件
INPUT_FILE = Path("absa_validation_sample_40.csv")
OUTPUT_FILE = Path("absa_llm_span_40.xlsx")

JSON_COL = "LLM_JSON解析"

# 读取40条样本
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

if "ABSA_ID" not in df.columns or JSON_COL not in df.columns:
    raise ValueError(
        f"文件必须包含 ABSA_ID 和 {JSON_COL} 两列。"
        f"当前列名：{df.columns.tolist()}"
    )

if df["ABSA_ID"].duplicated().any():
    raise ValueError("存在重复的 ABSA_ID，请检查抽样文件。")

rows = []
empty_ids = []
errors = []
span_counter = 1

for _, row in df.iterrows():
    absa_id = row["ABSA_ID"]
    raw_json = row[JSON_COL]

    if pd.isna(raw_json) or not str(raw_json).strip():
        errors.append({
            "ABSA_ID": absa_id,
            "问题": "JSON为空，请检查原始结果"
        })
        continue

    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError) as e:
        errors.append({
            "ABSA_ID": absa_id,
            "问题": f"JSON解析失败：{e}"
        })
        continue

    if not isinstance(data, list):
        errors.append({
            "ABSA_ID": absa_id,
            "问题": "JSON顶层不是列表"
        })
        continue

    if len(data) == 0:
        empty_ids.append(absa_id)
        continue

    for item in data:
        if not isinstance(item, dict):
            errors.append({
                "ABSA_ID": absa_id,
                "问题": "存在非字典格式的span"
            })
            continue

        text_span = item.get("text_span")

        if not isinstance(text_span, str) or not text_span.strip():
            errors.append({
                "ABSA_ID": absa_id,
                "问题": "text_span为空"
            })
            continue

        rows.append({
            "ABSA_ID": absa_id,
            "LLM_Span_ID": f"L{span_counter:04d}",
            "LLM_Span": text_span,
            "LLM_Aspect": item.get("aspect"),
            "LLM_Polarity": item.get("polarity"),
            "LLM_Target": item.get("target")
        })

        span_counter += 1

span_columns = [
    "ABSA_ID", "LLM_Span_ID", "LLM_Span",
    "LLM_Aspect", "LLM_Polarity", "LLM_Target"
]
span_df = pd.DataFrame(rows, columns=span_columns)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    span_df.to_excel(writer, sheet_name="LLM_Spans", index=False)

    pd.DataFrame({"ABSA_ID": empty_ids}).to_excel(
        writer, sheet_name="No_Span", index=False
    )

    pd.DataFrame(errors, columns=["ABSA_ID", "问题"]).to_excel(
        writer, sheet_name="Check_Errors", index=False
    )

print(f"抽样评论数：{len(df)}")
print(f"提取的 LLM span 数：{len(span_df)}")
print(f"有效空列表评论数：{len(empty_ids)}")
print(f"需要检查的记录数：{len(errors)}")
print(f"输出文件：{OUTPUT_FILE}")