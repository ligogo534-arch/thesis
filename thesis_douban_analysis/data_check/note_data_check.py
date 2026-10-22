import pandas as pd
import numpy as np

BOOK_ID_COL = "book_id"
BOOK_NAME_COL = "book_name"
USER_COL = "username"
TITLE_COL = "title"
TIME_COL = "time"
LIKE_COL = "like_count"

INPUT_FILE = "../data/note_4books.csv"
OUTPUT_FILE = "note_data_check.xlsx"

df = pd.read_csv(INPUT_FILE)
time_series = pd.to_datetime(df[TIME_COL], errors="coerce")
title_length = df[TITLE_COL].astype(str).str.len()

summary = pd.DataFrame({
    "Metric": [
        "Number of Rows",
        "Number of Columns",
        "Total Missing Values",
        "Duplicate Rows",
        "Duplicate Titles",
        "Duplicate User-Time-Title",
        "Invalid Dates",
        "Empty Titles",
        "Average Title Length",
        "Negative Likes",
        "Number of Books",
        "Number of Book Names",
        "Number of Users",
        "Earliest Date",
        "Latest Date"
    ],
    "Value": [
        len(df),
        len(df.columns),
        int(df.isna().sum().sum()),
        int(df.duplicated().sum()),
        int(df[TITLE_COL].duplicated().sum()),
        int(df.duplicated(subset=[USER_COL, TIME_COL, TITLE_COL]).sum()),
        int(time_series.isna().sum()),
        int(df[TITLE_COL].astype(str).str.strip().eq("").sum()),
        round(title_length.mean(), 2),
        int((df[LIKE_COL] < 0).sum()),
        df[BOOK_ID_COL].nunique(),
        df[BOOK_NAME_COL].nunique(),
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

statistics.append([LIKE_COL, "Minimum", df[LIKE_COL].min()])
statistics.append([LIKE_COL, "Maximum", df[LIKE_COL].max()])
statistics.append([LIKE_COL, "Mean", round(df[LIKE_COL].mean(), 2)])
statistics.append([LIKE_COL, "Median", df[LIKE_COL].median()])
statistics.append([LIKE_COL, "Standard Deviation", round(df[LIKE_COL].std(), 2)])
statistics.append([LIKE_COL, "Negative Values", int((df[LIKE_COL] < 0).sum())])

statistics.append([TITLE_COL, "Minimum Length", int(title_length.min())])
statistics.append([TITLE_COL, "Maximum Length", int(title_length.max())])
statistics.append([TITLE_COL, "Mean Length", round(title_length.mean(), 2)])
statistics.append([TITLE_COL, "Median Length", float(title_length.median())])
statistics.append([TITLE_COL, "Duplicate Titles", int(df[TITLE_COL].duplicated().sum())])
statistics.append([TITLE_COL, "Encoding Errors (�)", int(df[TITLE_COL].astype(str).str.contains("�", regex=False).sum())])
statistics.append([TITLE_COL, "Leading/Trailing Spaces", int((df[TITLE_COL].astype(str) != df[TITLE_COL].astype(str).str.strip()).sum())])

statistics.append([TIME_COL, "Invalid Dates", int(time_series.isna().sum())])
statistics.append([TIME_COL, "Earliest Date", time_series.min()])
statistics.append([TIME_COL, "Latest Date", time_series.max()])

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