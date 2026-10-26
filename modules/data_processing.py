"""Dataset schema checks and cleaning rules aligned with the SAD midterm notebook."""
from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "Reservation_ID", "Facility_Type", "User_Type", "Time_Category",
    "Reservation_Lead_Days", "Participants", "Day_of_Week", "Purpose",
    "Usage_Status", "Actual_Attendance", "Cancellation_Days_Before"
]
CATEGORICAL_COLUMNS = ["Facility_Type", "User_Type", "Time_Category", "Day_of_Week", "Purpose", "Usage_Status"]
PREDICTORS = ["Facility_Type", "User_Type", "Time_Category", "Reservation_Lead_Days", "Participants", "Day_of_Week"]


def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if df.empty:
        missing.append("The uploaded CSV has no data rows.")
    return not missing, missing


def clean_reservations(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning/feature-engineering rules used in the midterm notebook."""
    ok, issues = validate_schema(raw)
    if not ok:
        raise ValueError("Dataset schema validation failed: " + ", ".join(issues))
    df = raw.copy()
    df = df.drop_duplicates(subset=["Reservation_ID"]).reset_index(drop=True)
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("string").str.strip().str.title()
        df[col] = df[col].replace({"": pd.NA, "Nan": pd.NA, "None": pd.NA})
    for col in ["Reservation_Lead_Days", "Participants", "Actual_Attendance", "Cancellation_Days_Before"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Reservation_Lead_Days"] = df["Reservation_Lead_Days"].abs()
    # Cap implausible attendance typos at the expected participant count, as in the notebook.
    over = df["Actual_Attendance"] > df["Participants"]
    df.loc[over, "Actual_Attendance"] = df.loc[over, "Participants"]
    if df["User_Type"].notna().any():
        df["User_Type"] = df["User_Type"].fillna(df["User_Type"].mode().iloc[0])
    if df["Purpose"].notna().any():
        df["Purpose"] = df["Purpose"].fillna(df["Purpose"].mode().iloc[0])
    if df["Reservation_Lead_Days"].notna().any():
        df["Reservation_Lead_Days"] = df["Reservation_Lead_Days"].fillna(df["Reservation_Lead_Days"].median())
    if df["Participants"].notna().any():
        df["Participants"] = df["Participants"].fillna(df["Participants"].median())
    df.loc[df["Usage_Status"].isin(["Cancelled", "No-Show"]) & df["Actual_Attendance"].isna(), "Actual_Attendance"] = 0.0
    df["Reservation_Outcome"] = np.where(df["Usage_Status"].eq("Used"), "Completed", "Not Completed")
    df["Lead_Time_Category"] = pd.cut(
        df["Reservation_Lead_Days"],
        bins=[-np.inf, 2, 7, 14, np.inf],
        labels=["Last Minute", "Short Lead", "Medium Lead", "Early Booking"]
    ).astype("string")
    df["Attendance_Ratio"] = np.where(
        df["Participants"].fillna(0) > 0,
        (df["Actual_Attendance"] / df["Participants"]).fillna(0).round(4),
        0.0,
    )
    return df


def quality_summary(raw: pd.DataFrame) -> dict:
    return {
        "rows": int(len(raw)),
        "columns": int(len(raw.columns)),
        "duplicate_ids": int(raw["Reservation_ID"].duplicated().sum()) if "Reservation_ID" in raw.columns else 0,
        "missing_cells": int(raw.isna().sum().sum()),
        "missing_columns": [c for c in REQUIRED_COLUMNS if c not in raw.columns],
    }
