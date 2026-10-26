"""Decision-support dashboard helpers."""
import pandas as pd


def get_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    completed = int(df["Reservation_Outcome"].eq("Completed").sum())
    not_completed = int(df["Reservation_Outcome"].eq("Not Completed").sum())
    return {
        "total": total,
        "completed": completed,
        "not_completed": not_completed,
        "noncompletion_rate": (not_completed / total * 100) if total else 0.0,
        "facility_types": int(df["Facility_Type"].nunique()),
    }
