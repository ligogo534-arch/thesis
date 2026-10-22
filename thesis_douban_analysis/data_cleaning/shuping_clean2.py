#对书评进行第二种文本清洗，保留换行符
'''
import pandas as pd
import re
import os
from opencc import OpenCC

# OpenCC 简繁转换
cc = OpenCC("t2s")

def clean_text(text):
    """
    基础文本清洗（保留换行符版本，用于BERTopic段落级切分）：
    1. 空白字符统一
    2. 删除不可见字符
    3. 删除URL
    4. 繁体转简体
    5. 压缩空格
    6. 统一换行符格式（保留段落边界，供后续段落切分使用）
    """

    if pd.isna(text):
        return text
    text = str(text)

    # 1. 删除不可见字符BOM / zero width / variation selector
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF\uFE0F]", "", text)
    # 2. 删除控制字符
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    # 3. 删除URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # 4. 空白字符统一
    # 全角空格
    text = text.replace("\u3000", " ")
    # Tab
    text = text.replace("\t", " ")
    # 统一换行符：\r\n 和孤立的 \r 都归一化为 \n（不再替换成空格，保留段落边界）
    text = re.sub(r"\r\n|\r", "\n", text)
    # 连续3个以上换行压缩为2个（避免大量空段落）
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 5. 繁体转简体
    text = cc.convert(text)
    # 6. 多个空格压缩（不影响换行符）
    text = re.sub(r" {2,}", " ", text)
    # 7. 删除首尾空格（注意：strip()也会去除首尾的换行符，属预期行为）
    text = text.strip()
    return text

def clean_csv(input_file,output_file,encoding="utf-8-sig"):
    df = pd.read_csv(input_file,encoding=encoding)
    # 只清洗content
    df["content"] = df["content"].apply(clean_text)
    # 输出
    df.to_csv(output_file,index=False,encoding="utf-8-sig")
    print("完成：", output_file)
    print("数据量：", len(df))

# 书评
clean_csv(
    input_file="../../data_cleaning1/data_cleaned/shuping_clean.csv",
    output_file="shuping_clean2.csv"
)
'''

#检查是否有换行符（文本本身就没有换行符）
import pandas as pd

CSV_PATH = "shuping_clean2.csv"

df = pd.read_csv(CSV_PATH)
content = df["content"]

print(f"総行数: {len(df)}")

# 1. 检查是否存在“真正的换行符”（控制字符 \n）
n_real_newline = content.str.contains("\n", regex=False, na=False).sum()
print(f"包含【真正换行符】(\\n控制字符)的记录数: {n_real_newline}")

# 2. 检查是否存在“字面上的反斜杠+n”（即被转成字符串的假换行）
n_literal_backslash_n = content.astype(str).str.contains(r"\\n", regex=True, na=False).sum()
print(f"包含【字面反斜杠n】(\\\\n两个字符)的记录数: {n_literal_backslash_n}")

# 3. 打印第一条同时看两种情况，用repr()看清楚真实字符
sample = content.iloc[0]
print("\n--- 第一条记录的 repr() 输出（可以清楚看到是 \\n 还是 \\\\n）---")
print(repr(sample[:300]))

# 4. 找一条“看起来应该有多段落”的记录，比如字数较长的
long_idx = content.astype(str).str.len().idxmax()
print(f"\n--- 最长记录(index={long_idx}, 长度={len(content[long_idx])})的 repr() 输出 ---")
print(repr(content[long_idx][:500]))