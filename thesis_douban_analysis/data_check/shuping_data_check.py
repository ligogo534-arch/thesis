import pandas as pd
import numpy as np

BOOK_ID_COL = "book_id"
BOOK_NAME_COL = "book_name"
USER_COL = "user_name"
RATING_COL = "rating"
TIME_COL = "time"
COMMENT_ID_COL = "id"
TITLE_COL = "title"
CONTENT_COL = "content"
USEFUL_COL = "useful_count"
USELESS_COL = "useless_count"
REPLY_COL = "reply"
PAGE_COL = "page"

INPUT_FILE = "../data_cleaning1/data_cleaned/shuping_clean.csv"
OUTPUT_FILE = "shuping_clean_check.xlsx"

df = pd.read_csv(INPUT_FILE)
time_series = pd.to_datetime(df[TIME_COL], errors="coerce")
content_length = df[CONTENT_COL].astype(str).str.len()

summary = pd.DataFrame({
    "Metric": [
        "Number of Rows",
        "Number of Columns",
        "Total Missing Values",
        "Duplicate Rows",
        "Duplicate Content",
        "Duplicate User-Time-Content",
        "Duplicate Comment IDs",
        "Invalid Dates",
        "Empty Content",
        "Average Content Length",
        "Negative Useful Counts",
        "Negative Useless Counts",
        "Negative Replies",
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
        int(df[CONTENT_COL].duplicated().sum()),
        int(df.duplicated(subset=[USER_COL, TIME_COL, CONTENT_COL]).sum()),
        int(df[COMMENT_ID_COL].duplicated().sum()),
        int(time_series.isna().sum()),
        int(df[CONTENT_COL].astype(str).str.strip().eq("").sum()),
        round(content_length.mean(), 2),
        int((df[USEFUL_COL] < 0).sum()),
        int((df[USELESS_COL] < 0).sum()),
        int((df[REPLY_COL] < 0).sum()),
        df[BOOK_ID_COL].nunique(),
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
        empty_strings.append(int(df[col].astype(str).str.strip().eq("").sum()))
    else:
        empty_strings.append(np.nan)

variable_report["Empty Strings"] = empty_strings

statistics = []
statistics.append([RATING_COL, "Minimum", df[RATING_COL].min()])
statistics.append([RATING_COL, "Maximum", df[RATING_COL].max()])
statistics.append([RATING_COL, "Mean", round(df[RATING_COL].mean(), 2)])
statistics.append([RATING_COL, "Median", df[RATING_COL].median()])
statistics.append([RATING_COL, "Standard Deviation", round(df[RATING_COL].std(), 2)])

statistics.append([USEFUL_COL, "Minimum", df[USEFUL_COL].min()])
statistics.append([USEFUL_COL, "Maximum", df[USEFUL_COL].max()])
statistics.append([USEFUL_COL, "Mean", round(df[USEFUL_COL].mean(), 2)])
statistics.append([USEFUL_COL, "Median", df[USEFUL_COL].median()])
statistics.append([USEFUL_COL, "Standard Deviation", round(df[USEFUL_COL].std(), 2)])
statistics.append([USEFUL_COL, "Negative Values", int((df[USEFUL_COL] < 0).sum())])

statistics.append([USELESS_COL, "Minimum", df[USELESS_COL].min()])
statistics.append([USELESS_COL, "Maximum", df[USELESS_COL].max()])
statistics.append([USELESS_COL, "Mean", round(df[USELESS_COL].mean(), 2)])
statistics.append([USELESS_COL, "Median", df[USELESS_COL].median()])
statistics.append([USELESS_COL, "Standard Deviation", round(df[USELESS_COL].std(), 2)])
statistics.append([USELESS_COL, "Negative Values", int((df[USELESS_COL] < 0).sum())])

statistics.append([REPLY_COL, "Minimum", df[REPLY_COL].min()])
statistics.append([REPLY_COL, "Maximum", df[REPLY_COL].max()])
statistics.append([REPLY_COL, "Mean", round(df[REPLY_COL].mean(), 2)])
statistics.append([REPLY_COL, "Median", df[REPLY_COL].median()])
statistics.append([REPLY_COL, "Standard Deviation", round(df[REPLY_COL].std(), 2)])
statistics.append([REPLY_COL, "Negative Values", int((df[REPLY_COL] < 0).sum())])

statistics.append([CONTENT_COL, "Minimum Length", int(content_length.min())])
statistics.append([CONTENT_COL, "Maximum Length", int(content_length.max())])
statistics.append([CONTENT_COL, "Mean Length", round(content_length.mean(), 2)])
statistics.append([CONTENT_COL, "Median Length", float(content_length.median())])
statistics.append([CONTENT_COL, "Duplicate Content", int(df[CONTENT_COL].duplicated().sum())])
statistics.append([CONTENT_COL, "Encoding Errors (�)", int(df[CONTENT_COL].astype(str).str.contains("�", regex=False).sum())])
statistics.append([CONTENT_COL, "Leading/Trailing Spaces", int((df[CONTENT_COL].astype(str) != df[CONTENT_COL].astype(str).str.strip()).sum())])

statistics.append([TIME_COL, "Invalid Dates", int(time_series.isna().sum())])
statistics.append([TIME_COL, "Earliest Date", time_series.min()])
statistics.append([TIME_COL, "Latest Date", time_series.max()])

statistics.append([COMMENT_ID_COL, "Unique IDs", df[COMMENT_ID_COL].nunique()])
statistics.append([COMMENT_ID_COL, "Duplicate IDs", int(df[COMMENT_ID_COL].duplicated().sum())])

statistics = pd.DataFrame(statistics, columns=["Variable", "Statistic", "Value"])

book_distribution = df[BOOK_NAME_COL].value_counts().reset_index()
book_distribution.columns = ["Book", "Number of Reviews"]
book_distribution["Percentage (%)"] = (book_distribution["Number of Reviews"] / len(df) * 100).round(2)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    variable_report.to_excel(writer, sheet_name="Variable_Report")
    statistics.to_excel(writer, sheet_name="Statistics", index=False)
    start_row = len(statistics) + 3
    book_distribution.to_excel(
        writer,
        sheet_name="Statistics",
        startrow=start_row,
        index=False
    )
print("Data quality assessment completed successfully.")
print(f"Report saved to: {OUTPUT_FILE}")