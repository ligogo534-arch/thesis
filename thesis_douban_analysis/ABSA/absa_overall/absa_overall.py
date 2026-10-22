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
PROMPT_FILE = BASE_DIR /"prompt&codebook"/ "ABSA_LLM_system_prompt_v2.md"

FULL_DATA_FILES = [(BASE_DIR.parent / "data_cleaning2" / "text_cleaning" / "duanping_clean_base.csv", "短评"),
                   (BASE_DIR.parent / "data_cleaning2" / "text_cleaning" / "shuping_clean_base.csv", "书评"),]

OUTPUT_DIR = BASE_DIR / "absa_overall" / "absa_full_output"

OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "ABSA_Qwen3_全量结果.xlsx"
CHECKPOINT_FILE = OUTPUT_DIR / "ABSA_Qwen3_checkpoint.csv"

# ============================================================
# 2. 参数设置
# ============================================================

MODEL = "qwen/qwen3-235b-a22b-2507"
TEMPERATURE = 0
MAX_RETRIES = 3
SLEEP_SECONDS = 1
SAVE_EVERY = 50

# ============================================================
# 3. OpenRouter API
# ============================================================

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("没有找到 OPENROUTER_API_KEY，请先设置 OpenRouter API Key 环境变量。")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# ============================================================
# 4. 读取 System Prompt
# ============================================================

if not PROMPT_FILE.exists():
    raise FileNotFoundError(f"找不到 Prompt 文件：{PROMPT_FILE}")

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

print(f"Prompt 已读取：{PROMPT_FILE}")
print(f"Prompt 字符数：{len(SYSTEM_PROMPT)}")

# ============================================================
# 5. 读取两个全量数据文件，并添加语料来源
# ============================================================
dfs = []

for file, source in FULL_DATA_FILES:
    if not file.exists():
        raise FileNotFoundError(f"找不到数据文件：{file}")
    temp_df = pd.read_csv(file)
    if source == "短评":
        temp_df["book"] = temp_df["book_id"]
    else:
        temp_df["book"] = temp_df["book_name"]
    temp_df["语料来源"] = source
    print(f"读取：{file.name}，{len(temp_df)} 条，来源：{source}")
    dfs.append(temp_df)

df = pd.concat(dfs, ignore_index=True)
print(f"合并后原始数据：{len(df)} 条")

# ============================================================
# 6. 检查必要列
# ============================================================

REQUIRED_COLUMNS = ["book", "time", "content"]
missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

if missing_columns:
    raise ValueError(f"数据缺少必要列：{missing_columns}")
# ============================================================
# 7. 清理空评论
# ============================================================

df = df[df["content"].notna()].copy()
df["content"] = df["content"].astype(str).str.strip()
df = df[df["content"] != ""].copy()
df = df.reset_index(drop=True)

print(f"去除空评论后：{len(df)} 条")

# ============================================================
# 8. 生成 ABSA_ID
# ============================================================

df["ABSA_ID"] = [f"ABSA_{i:06d}" for i in range(1, len(df) + 1)]

print(f"ABSA_ID：ABSA_000001 ～ ABSA_{len(df):06d}")

# ============================================================
# 9. JSON 清理与解析
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
    except json.JSONDecodeError:
        return None

# ============================================================
# 10. API 调用函数
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

            return response.choices[0].message.content

        except Exception as e:
            print(f"API 调用失败，第 {attempt}/{MAX_RETRIES} 次：{e}")

            if attempt < MAX_RETRIES:
                time.sleep(5)

    return None

# ============================================================
# 11. 读取已有 checkpoint
# ============================================================

if CHECKPOINT_FILE.exists():
    checkpoint_df = pd.read_csv(CHECKPOINT_FILE, encoding="utf-8-sig")
    processed_ids = set(checkpoint_df["ABSA_ID"].astype(str))
    results = checkpoint_df.to_dict("records")

    print(f"检测到 checkpoint，已经完成：{len(results)} 条")

else:
    processed_ids = set()
    results = []

    print("没有检测到 checkpoint，从头开始。")

# ============================================================
# 12. 全量运行
# ============================================================

total = len(df)

for i, row in df.iterrows():

    absa_id = str(row["ABSA_ID"])

    if absa_id in processed_ids:
        continue

    content = row["content"]

    current_number = len(results) + 1

    print(f"\n[{current_number}/{total}] {absa_id} | {row['语料来源']}")
    print(f"评论：{content[:100]}{'...' if len(content) > 100 else ''}")

    llm_output = call_llm(content)
    parsed = parse_json_output(llm_output)

    if parsed is None:
        parse_status = "JSON解析失败"
        parsed_output = None
    else:
        parse_status = "成功"
        parsed_output = json.dumps(parsed, ensure_ascii=False)

    result = {
        "ABSA_ID": absa_id,
        "语料来源": row["语料来源"],
        "book": row["book"],
        "time": row["time"],
        "content": content,
        "LLM_原始输出": llm_output,
        "LLM_JSON解析": parsed_output,
        "LLM_JSON解析状态": parse_status,
    }

    results.append(result)
    processed_ids.add(absa_id)

    print(f"JSON解析：{parse_status}")

    # ========================================================
    # 13. 定期保存 checkpoint
    # ========================================================

    if len(results) % SAVE_EVERY == 0:
        checkpoint_df = pd.DataFrame(results)
        checkpoint_df.to_csv(
            CHECKPOINT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"Checkpoint 已保存：{len(results)} 条")

    time.sleep(SLEEP_SECONDS)

# ============================================================
# 14. 保存最终结果
# ============================================================

RESULT_COLUMNS = [
    "ABSA_ID",
    "语料来源",
    "book",
    "time",
    "content",
    "LLM_原始输出",
    "LLM_JSON解析",
    "LLM_JSON解析状态",
]

result_df = pd.DataFrame(results)
result_df = result_df[RESULT_COLUMNS]

result_df.to_excel(
    OUTPUT_FILE,
    index=False
)

result_df.to_csv(
    CHECKPOINT_FILE,
    index=False,
    encoding="utf-8-sig"
)

success_count = (
    result_df["LLM_JSON解析状态"] == "成功"
).sum()

failed_count = (
    result_df["LLM_JSON解析状态"] == "JSON解析失败"
).sum()

print("\n" + "=" * 60)
print("全量 ABSA 分析完成")
print("=" * 60)
print(f"总数据：{len(df)} 条")
print(f"LLM 结果：{len(result_df)} 条")
print(f"JSON 解析成功：{success_count} 条")
print(f"JSON 解析失败：{failed_count} 条")
print(f"短评：{(result_df['语料来源'] == '短评').sum()} 条")
print(f"长评：{(result_df['语料来源'] == '长评').sum()} 条")
print(f"Excel 结果：{OUTPUT_FILE}")
print(f"Checkpoint：{CHECKPOINT_FILE}")
print("=" * 60)