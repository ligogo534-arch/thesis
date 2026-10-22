from pathlib import Path
import os
import json
import time
import pandas as pd
from openai import OpenAI

# ============================================================
# 1. 路径设置
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_FILE = (BASE_DIR / "prompt&codebook" / "ABSA_LLM_system_prompt_v2.md")
SAMPLE_FILE = BASE_DIR / "robust_check" / "absa_stability_sample_50.csv"
OUTPUT_DIR = BASE_DIR / "robust_check" / "sample_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "ABSA_stability_rerun_50.xlsx"
CHECKPOINT_FILE = OUTPUT_DIR / "ABSA_stability_rerun_checkpoint.csv"

# 2. 参数设置
MODEL = "qwen/qwen3-235b-a22b-2507"
TEMPERATURE = 0
MAX_RETRIES = 3
SLEEP_SECONDS = 1
SAVE_EVERY = 50
RETRY_WAIT_SECONDS=5

# 3. OpenRouter API
# ============================================================
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError(
        "没有找到 OPENROUTER_API_KEY，请先设置 OpenRouter API Key。"
    )

client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=API_KEY)

# ============================================================
# 4. 读取 System Prompt
# ============================================================
if not PROMPT_FILE.exists():
    raise FileNotFoundError(f"找不到 Prompt 文件：{PROMPT_FILE}")

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

print(f"Prompt 文件：{PROMPT_FILE}")
print(f"Prompt 字符数：{len(SYSTEM_PROMPT)}")

# ============================================================
# 5. 读取50条抽样结果
# ============================================================
if not SAMPLE_FILE.exists():
    raise FileNotFoundError(
        f"找不到50条样本文件：{SAMPLE_FILE}\n"
        "请修改 SAMPLE_FILE 为实际文件路径。"
    )

if SAMPLE_FILE.suffix.lower() == ".xlsx":
    sample_df = pd.read_excel(SAMPLE_FILE)
else:
    sample_df = pd.read_csv(SAMPLE_FILE, encoding="utf-8-sig")

REQUIRED_COLUMNS = [
    "ABSA_ID",
    "语料来源",
    "book",
    "time",
    "content",
]

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in sample_df.columns
]

if missing_columns:
    raise ValueError(
        f"抽样文件缺少必要列：{missing_columns}\n"
        f"当前列名：{sample_df.columns.tolist()}"
    )

# 保留原始ID，不重新生成
sample_df = sample_df.dropna(subset=["ABSA_ID", "content"]).copy()
sample_df["ABSA_ID"] = sample_df["ABSA_ID"].astype(str).str.strip()
sample_df["content"] = sample_df["content"].astype(str).str.strip()

sample_df = sample_df[
    (sample_df["ABSA_ID"] != "") &
    (sample_df["content"] != "")
].copy()

if sample_df["ABSA_ID"].duplicated().any():
    duplicated_ids = sample_df.loc[
        sample_df["ABSA_ID"].duplicated(keep=False), "ABSA_ID"
    ].tolist()
    raise ValueError(f"抽样文件中存在重复 ABSA_ID：{duplicated_ids}")

if len(sample_df) != 50:
    raise ValueError(
        f"预期抽取50条，实际读取到 {len(sample_df)} 条。"
        "请确认输入文件。"
    )

sample_df = sample_df.reset_index(drop=True)

print(f"\n抽样文件：{SAMPLE_FILE}")
print(f"样本数：{len(sample_df)}")
print("\n样本分布：")
print(
    sample_df.groupby(["book", "语料来源"], dropna=False)
    .size()
    .to_string()
)

# ============================================================
# 6. JSON 清理与解析
# ============================================================
def clean_json_output(text):
    if not text:
        return None

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def parse_json_output(text):
    cleaned = clean_json_output(text)

    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None

# ============================================================
# 7. API 调用函数
# ============================================================
def call_llm(content):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=TEMPERATURE,
            )

            output = response.choices[0].message.content

            if output is None or not output.strip():
                raise ValueError("API 返回空内容")

            return output, "成功"

        except Exception as e:
            print(
                f"API调用失败，第 {attempt}/{MAX_RETRIES} 次："
                f"{type(e).__name__}: {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)

    return None, "调用失败"

# ============================================================
# 8. 读取 checkpoint，支持断点续跑
# ============================================================
if CHECKPOINT_FILE.exists():
    checkpoint_df = pd.read_csv(
        CHECKPOINT_FILE,
        encoding="utf-8-sig",
        dtype={"ABSA_ID": str},
    )

    # 只恢复当前50条样本对应的结果
    valid_ids = set(sample_df["ABSA_ID"])
    checkpoint_df = checkpoint_df[
        checkpoint_df["ABSA_ID"].isin(valid_ids)
    ].copy()

    results_by_id = {
        str(row["ABSA_ID"]): row
        for row in checkpoint_df.to_dict("records")
    }

    # 只有API已成功返回的记录才视为已完成。
    # JSON解析失败但API正常返回的结果也保留，不重复调用。
    successful_ids = {
        str(row["ABSA_ID"])
        for row in checkpoint_df.to_dict("records")
        if row.get("重跑_API调用状态") == "成功"
    }

    print(f"\n检测到 checkpoint：{len(results_by_id)} 条已有记录")
    print(f"其中已完成API调用：{len(successful_ids)} 条")

else:
    results_by_id = {}
    successful_ids = set()
    print("\n未检测到 checkpoint，从头开始。")

# ============================================================
# 9. 保存函数
# ============================================================
def save_checkpoint():
    result_df = pd.DataFrame(
        list(results_by_id.values())
    )

    if not result_df.empty:
        result_df.to_csv(
            CHECKPOINT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

# ============================================================
# 10. 重跑50条样本
# ============================================================
total = len(sample_df)

for i, row in sample_df.iterrows():
    absa_id = str(row["ABSA_ID"])

    if absa_id in successful_ids:
        print(f"[跳过] {absa_id}：已完成API调用")
        continue

    content = row["content"]

    print(f"\n[{i + 1}/{total}] {absa_id}")
    print(f"来源：{row['语料来源']} | 书籍：{row['book']}")
    print(
        f"评论：{content[:100]}"
        f"{'...' if len(content) > 100 else ''}"
    )

    llm_output, api_status = call_llm(content)
    parsed = parse_json_output(llm_output)

    if api_status != "成功":
        parse_status = "API调用失败"
        parsed_output = None
    elif parsed is None:
        parse_status = "JSON解析失败"
        parsed_output = None
    else:
        parse_status = "成功"
        parsed_output = json.dumps(
            parsed,
            ensure_ascii=False,
        )

    # 保留原抽样记录的所有字段
    result = row.to_dict()

    # 第二次运行字段使用独立前缀，避免覆盖第一次输出
    result.update({
        "重跑_模型": MODEL,
        "重跑_Temperature": TEMPERATURE,
        "重跑_API调用状态": api_status,
        "重跑_LLM_原始输出": llm_output,
        "重跑_LLM_JSON解析": parsed_output,
        "重跑_LLM_JSON解析状态": parse_status,
    })

    results_by_id[absa_id] = result

    if api_status == "成功":
        successful_ids.add(absa_id)

    print(f"API调用：{api_status}")
    print(f"JSON解析：{parse_status}")

    # 定期保存，避免中断后丢失进度
    if (i + 1) % SAVE_EVERY == 0:
        save_checkpoint()
        print(f"Checkpoint 已保存：{len(results_by_id)} 条")

    # API调用失败的记录下次启动时会自动重试
    time.sleep(SLEEP_SECONDS)

# ============================================================
# 11. 保存最终结果
# ============================================================
save_checkpoint()

# 按原始抽样顺序输出，确保结果方便与首次分析对齐
result_df = sample_df.merge(
    pd.DataFrame(list(results_by_id.values())).drop(
        columns=[
            col for col in sample_df.columns
            if col != "ABSA_ID"
        ],
        errors="ignore",
    ),
    on="ABSA_ID",
    how="left",
    validate="one_to_one",
)

result_df.to_excel(OUTPUT_FILE, index=False)

api_success = (result_df["重跑_API调用状态"] == "成功").sum()
parse_success = (result_df["重跑_LLM_JSON解析状态"] == "成功").sum()
parse_failed = (result_df["重跑_LLM_JSON解析状态"] == "JSON解析失败").sum()
api_failed = (result_df["重跑_LLM_JSON解析状态"] == "API调用失败").sum()

print("\n" + "=" * 60)
print("LLM-ABSA 稳定性检验：第二次运行完成")
print("=" * 60)
print(f"抽样总数：{len(sample_df)}")
print(f"API调用成功：{api_success}")
print(f"JSON解析成功：{parse_success}")
print(f"JSON解析失败：{parse_failed}")
print(f"API调用失败：{api_failed}")
print(f"Excel结果：{OUTPUT_FILE}")
print(f"Checkpoint：{CHECKPOINT_FILE}")
print("=" * 60)