"""Reusable descriptive statistics and EDA chart helpers."""
import pandas as pd
import plotly.express as px


def outcome_counts(df):
    return df["Reservation_Outcome"].value_counts().rename_axis("Outcome").reset_index(name="Reservations")


def facility_counts(df):
    return df["Facility_Type"].value_counts().rename_axis("Facility").reset_index(name="Reservations")


def outcome_by_user(df):
    return pd.crosstab(df["User_Type"], df["Reservation_Outcome"]).reset_index().melt(
        id_vars="User_Type", var_name="Outcome", value_name="Reservations"
    )


def noncompletion_by_lead(df):
    grouped = df.groupby("Lead_Time_Category", observed=False)["Reservation_Outcome"].apply(
        lambda s: (s.eq("Not Completed").mean() * 100) if len(s) else 0
    ).reset_index(name="Non-completion rate (%)")
    return grouped


def make_facility_chart(df):
    counts = facility_counts(df).sort_values("Reservations")
    return px.bar(counts, x="Reservations", y="Facility", orientation="h", text="Reservations",
                  title="Reservations by facility type", color_discrete_sequence=["#287D8E"])


def make_outcome_chart(df):
    counts = outcome_counts(df)
    return px.pie(counts, names="Outcome", values="Reservations", hole=0.58,
                  title="Observed reservation outcomes", color_discrete_sequence=["#287D8E", "#D97745"])


def make_user_chart(df):
    counts = outcome_by_user(df)
    return px.bar(counts, x="User_Type", y="Reservations", color="Outcome", barmode="group",
                  title="Observed outcome by user type", color_discrete_map={"Completed":"#287D8E", "Not Completed":"#D97745"})


def make_lead_chart(df):
    counts = noncompletion_by_lead(df)
    return px.bar(counts, x="Lead_Time_Category", y="Non-completion rate (%)", text_auto=".1f",
                  title="Observed non-completion rate by booking lead time", color_discrete_sequence=["#526D82"])
