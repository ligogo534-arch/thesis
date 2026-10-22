import pandas as pd
import numpy as np

URL_COL = "详情页url"
BOOK_COL = "书名"
USER_COL = "用户名"
TIME_COL = "时间"
REPLY_COL = "回复数"
LIKE_COL = "点赞数"
TEXT_COL = "文本内容"
SOURCE_COL = "引用出处"
JSON_COL = "正文结构JSON"
FIGURE_COL = "figure数量"

#检查清洗后数据
INPUT_FILE = "../data_cleaning1/data_cleaned/zhailu_clean.csv"
OUTPUT_FILE = "zhailu6_clean_report.xlsx"

df = pd.read_csv(INPUT_FILE)

time_series = pd.to_datetime(df[TIME_COL], errors="coerce")
text_length = df[TEXT_COL].fillna("").astype(str).str.len()

summary = pd.DataFrame({
    "Metric": [
        "Number of Rows",
        "Number of Columns",
        "Total Missing Values",
        "Duplicate Rows",
        "Duplicate Content",
        "Duplicate URLs",
        "Duplicate User-Time-Text",
        "Missing Full Text",
        "Missing Citation",
        "Missing JSON",
        "Missing Figure Count",
        "Invalid Dates",
        "Empty Text",
        "Average Text Length",
        "Negative Replies",
        "Negative Likes",
        "Number of Books",
        "Number of Users",
        "Earliest Date",
        "Latest Date"
    ],
    "Value": [
        len(df),
        len(df.columns),
        int(df.isna().sum().sum()),
        int(df.duplicated().sum()),
        int(df[TEXT_COL].duplicated().sum()),
        int(df[URL_COL].duplicated().sum()),
        int(df.duplicated(subset=[USER_COL, TIME_COL, TEXT_COL]).sum()),
        int(df[TEXT_COL].isna().sum()),
        int(df[SOURCE_COL].isna().sum()),
        int(df[JSON_COL].isna().sum()),
        int(df[FIGURE_COL].isna().sum()),
        int(time_series.isna().sum()),
        int(df[TEXT_COL].fillna("").astype(str).str.strip().eq("").sum()),
        round(text_length.mean(), 2),
        int((df[REPLY_COL] < 0).sum()),
        int((df[LIKE_COL] < 0).sum()),
        df[BOOK_COL].nunique(),
        df[USER_COL].nunique(),
        time_series.min(),
        time_series.max()
    ]
})

variable_report = pd.DataFrame(index=df.columns)

variable_report["Data Type"] = df.dtypes.astype(str)
variable_report["Missing Count"] = df.isna().sum()
variable_report["Missing Rate (%)"] = (df.isna().mean() * 100).round(2)
variable_report["Unique Values"] = df.nunique(dropna=False)

empty_strings = []

for col in df.columns:
    if df[col].dtype == "object":
        empty_strings.append(int(df[col].fillna("").astype(str).str.strip().eq("").sum()))
    else:
        empty_strings.append(np.nan)

variable_report["Empty Strings"] = empty_strings

statistics = []

statistics.append([REPLY_COL, "Minimum", df[REPLY_COL].min()])
statistics.append([REPLY_COL, "Maximum", df[REPLY_COL].max()])
statistics.append([REPLY_COL, "Mean", round(df[REPLY_COL].mean(), 2)])
statistics.append([REPLY_COL, "Median", df[REPLY_COL].median()])
statistics.append([REPLY_COL, "Standard Deviation", round(df[REPLY_COL].std(), 2)])

statistics.append([LIKE_COL, "Minimum", df[LIKE_COL].min()])
statistics.append([LIKE_COL, "Maximum", df[LIKE_COL].max()])
statistics.append([LIKE_COL, "Mean", round(df[LIKE_COL].mean(), 2)])
statistics.append([LIKE_COL, "Median", df[LIKE_COL].median()])
statistics.append([LIKE_COL, "Standard Deviation", round(df[LIKE_COL].std(), 2)])

statistics.append([FIGURE_COL, "Minimum", df[FIGURE_COL].min()])
statistics.append([FIGURE_COL, "Maximum", df[FIGURE_COL].max()])
statistics.append([FIGURE_COL, "Mean", round(df[FIGURE_COL].mean(), 2)])
statistics.append([FIGURE_COL, "Median", df[FIGURE_COL].median()])
statistics.append([FIGURE_COL, "Standard Deviation", round(df[FIGURE_COL].std(), 2)])

statistics.append([TEXT_COL, "Minimum Length", int(text_length.min())])
statistics.append([TEXT_COL, "Maximum Length", int(text_length.max())])
statistics.append([TEXT_COL, "Mean Length", round(text_length.mean(), 2)])
statistics.append([TEXT_COL, "Median Length", float(text_length.median())])
statistics.append([TEXT_COL, "Encoding Errors (�)", int(df[TEXT_COL].fillna("").astype(str).str.contains("�", regex=False).sum())])
statistics.append([TEXT_COL, "Leading/Trailing Spaces", int((df[TEXT_COL].fillna("").astype(str) != df[TEXT_COL].fillna("").astype(str).str.strip()).sum())])

statistics.append([URL_COL, "Duplicate URLs", int(df[URL_COL].duplicated().sum())])
statistics.append([URL_COL, "Missing URLs", int(df[URL_COL].isna().sum())])

statistics.append([SOURCE_COL, "Missing Citation", int(df[SOURCE_COL].isna().sum())])

statistics.append([JSON_COL, "Missing JSON", int(df[JSON_COL].isna().sum())])

statistics.append([TIME_COL, "Invalid Dates", int(time_series.isna().sum())])
statistics.append([TIME_COL, "Earliest Date", time_series.min()])
statistics.append([TIME_COL, "Latest Date", time_series.max()])
statistics = pd.DataFrame(statistics, columns=["Variable", "Statistic", "Value"])

book_distribution = df[BOOK_COL].value_counts().reset_index()
book_distribution.columns = ["Book", "Number of Records"]
book_distribution["Percentage (%)"] = (book_distribution["Number of Records"] / len(df) * 100).round(2)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    variable_report.to_excel(writer, sheet_name="Variable_Report")
    statistics.to_excel(writer, sheet_name="Statistics", index=False)
    start_row = len(statistics) + 3
    book_distribution.to_excel(writer,sheet_name="Statistics",startrow=start_row,index=False)

print("=" * 50)
print("Data quality assessment completed successfully.")
print(f"Rows: {len(df)}")
print(f"Books: {df[BOOK_COL].nunique()}")
print(f"Users: {df[USER_COL].nunique()}")
print(f"Duplicate Rows: {df.duplicated().sum()}")
print(f"Duplicate URLs: {df[URL_COL].duplicated().sum()}")
print(f"Missing Full Text: {df[TEXT_COL].isna().sum()}")
print(f"Missing Citation: {df[SOURCE_COL].isna().sum()}")
print(f"Missing JSON: {df[JSON_COL].isna().sum()}")
print(f"Missing Figure Count: {df[FIGURE_COL].isna().sum()}")
print(f"Report saved to: {OUTPUT_FILE}")
print("=" * 50)