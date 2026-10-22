import pandas as pd
import numpy as np

BOOK_COL = "book_id"
USER_COL = "user_name"
RATING_COL = "rating"
TIME_COL = "time"
TEXT_COL = "content"
VOTE_COL = "vote"

#检查清洗前数据duanping_4books.csv
'''
INPUT_FILE = "../data/duanping_4books.csv"
OUTPUT_FILE = "duanping_data_report.xlsx"
'''
#检查清洗后数据duanping_clean.csv
INPUT_FILE = "../data_cleaning1/data_cleaned/duanping_clean.csv"
OUTPUT_FILE = "duanping_clean_report.xlsx"

df = pd.read_csv(INPUT_FILE)

time_series = pd.to_datetime(df[TIME_COL], errors="coerce")
text_length = df[TEXT_COL].astype(str).str.len()

summary = pd.DataFrame({
    "Metric": [
        "Number of Rows",
        "Number of Columns",
        "Total Missing Values",
        "Duplicate Rows",
        "Duplicate Text",
        "Duplicate User-Time-Text",
        "Invalid Dates",
        "Empty Text",
        "Average Text Length",
        "Negative Votes",
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
        int(df.duplicated(subset=[USER_COL, TIME_COL, TEXT_COL]).sum()),
        int(time_series.isna().sum()),
        int(df[TEXT_COL].astype(str).str.strip().eq("").sum()),
        round(text_length.mean(), 2),
        int((df[VOTE_COL] < 0).sum()),
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

statistics.append([VOTE_COL, "Minimum", df[VOTE_COL].min()])
statistics.append([VOTE_COL, "Maximum", df[VOTE_COL].max()])
statistics.append([VOTE_COL, "Mean", round(df[VOTE_COL].mean(), 2)])
statistics.append([VOTE_COL, "Median", df[VOTE_COL].median()])
statistics.append([VOTE_COL, "Standard Deviation", round(df[VOTE_COL].std(), 2)])
statistics.append([VOTE_COL, "Negative Values", int((df[VOTE_COL] < 0).sum())])

statistics.append([TEXT_COL, "Minimum Length", int(text_length.min())])
statistics.append([TEXT_COL, "Maximum Length", int(text_length.max())])
statistics.append([TEXT_COL, "Mean Length", round(text_length.mean(), 2)])
statistics.append([TEXT_COL, "Median Length", float(text_length.median())])
statistics.append([TEXT_COL, "Duplicate Text", int(df[TEXT_COL].duplicated().sum())])
statistics.append([TEXT_COL, "Encoding Errors (�)", int(df[TEXT_COL].astype(str).str.contains("�", regex=False).sum())])
statistics.append([TEXT_COL, "Leading/Trailing Spaces", int((df[TEXT_COL].astype(str) != df[TEXT_COL].astype(str).str.strip()).sum())])

statistics.append([TIME_COL, "Invalid Dates", int(time_series.isna().sum())])
statistics.append([TIME_COL, "Earliest Date", time_series.min()])
statistics.append([TIME_COL, "Latest Date", time_series.max()])

statistics = pd.DataFrame(statistics, columns=["Variable", "Statistic", "Value"])

book_distribution = df[BOOK_COL].value_counts().reset_index()
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