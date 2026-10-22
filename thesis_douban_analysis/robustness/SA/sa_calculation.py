from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# File paths

INPUT_FILE = Path("sa_sample_80.csv")
OUTPUT_FILE = Path("sa_validation_result_80.xlsx")

# Read data
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

y_true = df["人工标注极性"]
y_pred = df["原标签"]
labels = ["正", "负", "中立"]

# Overall metrics
accuracy = accuracy_score(y_true, y_pred)
macro_precision = precision_score(y_true, y_pred,labels=labels,average="macro", zero_division=0)
macro_recall = recall_score(y_true, y_pred,labels=labels,average="macro",zero_division=0)
macro_f1 = f1_score(y_true, y_pred,labels=labels,average="macro",zero_division=0)
weighted_precision = precision_score(y_true, y_pred,labels=labels,average="weighted", zero_division=0)
weighted_recall = recall_score(y_true, y_pred,labels=labels,average="weighted",zero_division=0)
weighted_f1 = f1_score(y_true, y_pred,labels=labels,average="weighted",zero_division=0)

# Overall DataFrame
overall_df = pd.DataFrame({
    "指標": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1"
    ],
    "数値": [
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        weighted_precision,
        weighted_recall,
        weighted_f1
    ]
})

# Per-class metrics
report = classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=labels,
    output_dict=True,
    zero_division=0
)

per_class_df = pd.DataFrame({
    "感情極性": labels,
    "Precision": [
        report["正"]["precision"],
        report["负"]["precision"],
        report["中立"]["precision"]
    ],
    "Recall": [
        report["正"]["recall"],
        report["负"]["recall"],
        report["中立"]["recall"]
    ],
    "F1": [
        report["正"]["f1-score"],
        report["负"]["f1-score"],
        report["中立"]["f1-score"]
    ],
    "Support": [
        report["正"]["support"],
        report["负"]["support"],
        report["中立"]["support"]
    ]
})

# Confusion Matrix
cm = confusion_matrix(y_true,y_pred,labels=labels)
confusion_df = pd.DataFrame(cm,
    index=[f"人工_{x}" for x in labels],
    columns=[f"予測_{x}" for x in labels])

confusion_df.index.name = "実際のラベル"

# Save to Excel
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    overall_df.to_excel(writer,sheet_name="Overall",index=False)
    per_class_df.to_excel(writer,sheet_name="Per_Class",index=False)
    confusion_df.to_excel(writer,sheet_name="Confusion_Matrix")

print(f"検証結果を保存しました：{OUTPUT_FILE}")