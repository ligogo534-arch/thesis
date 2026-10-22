import os
import json
import re
import pandas as pd
from pathlib import Path
from openai import OpenAI

'''
BASE_DIR = Path(__file__).resolve().parent

#8条重跑
RESULT_FILE = BASE_DIR / "absa_full_output" / "ABSA_Qwen3_全量结果.xlsx"
RERUN_FILE = BASE_DIR / "absa_full_output" / "ABSA_Qwen3_8条重跑结果.xlsx"

#最后1条重跑
RESULT_FILE = BASE_DIR / "absa_full_output" / "ABSA_Qwen3_8条重跑结果.xlsx"
RERUN_FILE = BASE_DIR / "absa_full_output" / "ABSA_Qwen3_1条重跑结果.xlsx"

MODEL = "qwen/qwen3-235b-a22b-2507"
MAX_TOKENS = 8192
TEMPERATURE = 0

client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

PROMPT_FILE = BASE_DIR.parent /"prompt&codebook"/ "ABSA_LLM_system_prompt_v2.md"
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

df = pd.read_excel(RESULT_FILE)
failed = df[df["LLM_JSON解析状态_重跑"] == "JSON解析失败"].copy()
print(f"发现 JSON 解析失败：{len(failed)} 条")

def clean_output(text):
    text = str(text).strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

def parse_json(text):
    cleaned = clean_output(text)
    try:
        return json.loads(cleaned), "成功"
    except json.JSONDecodeError:
        return None, "JSON解析失败"

results = []

for i, (_, row) in enumerate(failed.iterrows(), 1):
    absa_id = row["ABSA_ID"]
    content = str(row["content"])
    print(f"\n[{i}/{len(failed)}] 正在重跑：{absa_id}")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(row["content"])}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        raw_output = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        parsed, status = parse_json(raw_output)

        results.append({
            "ABSA_ID": absa_id,
            "语料来源": row["语料来源"],
            "book": row["book"],
            "time": row["time"],
            "content": content,
            "LLM_原始输出_重跑": raw_output,
            "LLM_JSON解析_重跑": json.dumps(parsed, ensure_ascii=False) if parsed is not None else None,
            "LLM_JSON解析状态_重跑": status,
            "finish_reason_重跑": finish_reason
        })

        print(f"解析状态：{status}，finish_reason：{finish_reason}")

    except Exception as e:
        print(f"API调用失败：{e}")
        results.append({
            "ABSA_ID": absa_id,
            "语料来源": row["语料来源"],
            "book": row["book"],
            "time": row["time"],
            "content": content,
            "LLM_原始输出_重跑": None,
            "LLM_JSON解析_重跑": None,
            "LLM_JSON解析状态_重跑": f"API调用失败：{e}",
            "finish_reason_重跑": None
        })

rerun_df = pd.DataFrame(results)
rerun_df.to_excel(RERUN_FILE, index=False)

print("\n" + "=" * 60)
print("重跑完成")
print("=" * 60)
print(f"重跑数量：{len(rerun_df)}")
print(f"JSON解析成功：{(rerun_df['LLM_JSON解析状态_重跑'] == '成功').sum()}")
print(f"JSON解析失败：{(rerun_df['LLM_JSON解析状态_重跑'] == 'JSON解析失败').sum()}")
print(f"结果保存：{RERUN_FILE}")
'''

#重跑的8条合并回去，最后1条再合并回去
BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_FILE = BASE_DIR / "absa_full_output" / "ABSA_Qwen3_全量结果.xlsx"
RERUN_8_FILE = BASE_DIR / "absa_full_output" / "ABSA_Qwen3_8条重跑结果.xlsx"
RERUN_1_FILE = BASE_DIR / "absa_full_output" / "ABSA_Qwen3_1条重跑结果.xlsx"
FINAL_FILE = BASE_DIR / "absa_full_output" / "ABSA_Qwen3_最终全量结果.xlsx"

original = pd.read_excel(ORIGINAL_FILE)
rerun_8 = pd.read_excel(RERUN_8_FILE)
rerun_1 = pd.read_excel(RERUN_1_FILE)

print(f"原始结果：{len(original)} 条")
print(f"8条重跑：{len(rerun_8)} 条")
print(f"1条重跑：{len(rerun_1)} 条")

if len(original) != 5500:
    raise ValueError(f"原始结果不是5500条，而是 {len(original)} 条")

if rerun_8["ABSA_ID"].duplicated().any():
    raise ValueError("8条重跑结果中存在重复 ABSA_ID")

if rerun_1["ABSA_ID"].duplicated().any():
    raise ValueError("1条重跑结果中存在重复 ABSA_ID")

rerun_8_ids = set(rerun_8["ABSA_ID"])
rerun_1_ids = set(rerun_1["ABSA_ID"])

if not rerun_1_ids.issubset(rerun_8_ids):
    raise ValueError("最后1条重跑的 ABSA_ID 不属于8条重跑结果")

if not rerun_8_ids.issubset(set(original["ABSA_ID"])):
    raise ValueError("8条重跑结果中存在原始数据没有的 ABSA_ID")

print("\n开始替换8条重跑结果...")

final_df = original.copy()

for _, row in rerun_8.iterrows():
    absa_id = row["ABSA_ID"]
    idx = final_df.index[final_df["ABSA_ID"] == absa_id]
    if len(idx) != 1:
        raise ValueError(f"ABSA_ID {absa_id} 在原始结果中不是唯一一条")
    i = idx[0]
    final_df.loc[i, "LLM_原始输出"] = row["LLM_原始输出_重跑"]
    final_df.loc[i, "LLM_JSON解析"] = row["LLM_JSON解析_重跑"]
    final_df.loc[i, "LLM_JSON解析状态"] = row["LLM_JSON解析状态_重跑"]

print("8条重跑结果替换完成")

print("\n开始替换最后1条重跑结果...")

for _, row in rerun_1.iterrows():
    absa_id = row["ABSA_ID"]
    idx = final_df.index[final_df["ABSA_ID"] == absa_id]
    if len(idx) != 1:
        raise ValueError(f"ABSA_ID {absa_id} 在最终结果中不是唯一一条")
    i = idx[0]
    final_df.loc[i, "LLM_原始输出"] = row["LLM_原始输出_重跑"]
    final_df.loc[i, "LLM_JSON解析"] = row["LLM_JSON解析_重跑"]
    final_df.loc[i, "LLM_JSON解析状态"] = row["LLM_JSON解析状态_重跑"]

print("最后1条重跑结果替换完成")

final_df.to_excel(FINAL_FILE, index=False)

print("\n" + "=" * 60)
print("最终结果检查")
print("=" * 60)

print(f"最终数据量：{len(final_df)} 条")
print(f"唯一 ABSA_ID：{final_df['ABSA_ID'].nunique()} 个")
print(f"JSON解析成功：{(final_df['LLM_JSON解析状态'] == 'JSON解析成功').sum()} 条")
print(f"JSON解析失败：{(final_df['LLM_JSON解析状态'] == 'JSON解析失败').sum()} 条")
print(f"输出文件：{FINAL_FILE}")

if len(final_df) != 5500:
    raise ValueError("最终数据不是5500条")

if final_df["ABSA_ID"].nunique() != 5500:
    raise ValueError("最终 ABSA_ID 不是5500个唯一值")

failed = final_df[final_df["LLM_JSON解析状态"] == "JSON解析失败"]

if len(failed) > 0:
    print("\n警告：仍存在JSON解析失败：")
    print(failed[["ABSA_ID", "语料来源"]].to_string(index=False))
else:
    print("\n✓ 5500条数据全部JSON解析成功")

print(f"最终数据量：{len(df)} 条")
print(f"唯一 ABSA_ID：{df['ABSA_ID'].nunique()} 个")
print(f"JSON解析状态：")
print(df["LLM_JSON解析状态"].value_counts(dropna=False))