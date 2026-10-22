#对短评和书评进行统一的基础文本清洗
import pandas as pd
import re
import os
from opencc import OpenCC

# OpenCC 简繁转换
cc = OpenCC("t2s")

def clean_text(text):
    """
    基础文本清洗：
    1. 空白字符统一
    2. 删除不可见字符
    3. 删除URL
    4. 繁体转简体
    5. 压缩空格
    """

    if pd.isna(text):
        return text
    text = str(text)

    # 1. 删除不可见字符BOM / zero width / variation selector
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF\uFE0F]","",text)
    # 2. 删除控制字符
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]","", text)
    # 3. 删除URL
    text = re.sub(r"https?://\S+|www\.\S+","",text)
    # 4. 空白字符统一
    # 全角空格
    text = text.replace("\u3000"," ")
    # Tab
    text = text.replace("\t"," ")
    # 换行
    text = re.sub(r"\r|\n"," ", text)
    # 5. 繁体转简体
    text = cc.convert(text)
    # 6. 多个空格压缩
    text = re.sub(r" {2,}"," ",text)
    # 7. 删除首尾空格
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

# 短评
clean_csv(
    input_file="../../data_cleaning1/data_cleaned/duanping_clean.csv",
    output_file="duanping_clean_base.csv"
)
# 书评
clean_csv(
    input_file="../../data_cleaning1/data_cleaned/shuping_clean.csv",
    output_file="shuping_clean_base.csv"
)