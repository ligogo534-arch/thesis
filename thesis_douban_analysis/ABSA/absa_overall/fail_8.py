import pandas as pd
import json
from pathlib import Path

RESULT_FILE = Path("absa_full_output/ABSA_Qwen3_全量结果.xlsx")
CHECK_OUTPUT_FILE = Path("absa_full_output/ABSA_Qwen3_JSON失败检查.xlsx")

df = pd.read_excel(RESULT_FILE)

failed = df[df["LLM_JSON解析状态"] == "JSON解析失败"].copy()
print(f"解析失败：{len(failed)} 条")

def clean_json_output(text):
    if pd.isna(text) or not str(text).strip():
        return None
    text = str(text).strip()
    # 去除 Markdown JSON 代码块
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    # 修复非法的 \' 转义
    text = text.replace("\\'", "'")
    return text

def check_json(text):
    if pd.isna(text) or not str(text).strip():
        return "空输出", None, None
    cleaned = clean_json_output(text)
    try:
        parsed = json.loads(cleaned)
        return "清洗后JSON合法", cleaned, parsed
    except json.JSONDecodeError as e:
        return f"清洗后仍非法：第{e.pos}字符附近，{e.msg}", cleaned, None

check_results = []

for _, row in failed.iterrows():
    status, cleaned, parsed = check_json(row["LLM_原始输出"])
    check_results.append({
        "ABSA_ID": row["ABSA_ID"],
        "语料来源": row["语料来源"],
        "book": row["book"],
        "time": row["time"],
        "content": row["content"],
        "LLM_原始输出": row["LLM_原始输出"],
        "清洗后输出": cleaned,
        "检查结果": status,
        "解析结果": json.dumps(parsed, ensure_ascii=False) if parsed is not None else None
    })

check_df = pd.DataFrame(check_results)
check_df.to_excel(CHECK_OUTPUT_FILE, index=False)

print("\n检查完成")
print(f"结果已保存至：{CHECK_OUTPUT_FILE}")
print("\n检查结果统计：")
print(check_df["检查结果"].value_counts())