import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Campus Facility Reservation Dashboard", page_icon="🏫", layout="wide")

st.title("🏫 Campus Facility Reservation & Monitoring System")
st.caption("Decision-support dashboard • Historical records and model predictions are shown separately.")

DATA_PATH = "facility_reservations.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    required = [
        "Reservation_ID", "Facility_Type", "User_Type", "Time_Category",
        "Reservation_Lead_Days", "Participants", "Day_of_Week", "Purpose",
        "Usage_Status", "Actual_Attendance", "Cancellation_Days_Before"
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    data = data.drop_duplicates(subset=["Reservation_ID"]).copy()
    for col in ["Facility_Type", "User_Type", "Time_Category", "Day_of_Week", "Purpose", "Usage_Status"]:
        data[col] = data[col].astype("string").str.strip().str.title()
    data["Reservation_Lead_Days"] = pd.to_numeric(data["Reservation_Lead_Days"], errors="coerce").abs()
    data["Participants"] = pd.to_numeric(data["Participants"], errors="coerce")
    data["Actual_Attendance"] = pd.to_numeric(data["Actual_Attendance"], errors="coerce")
    data.loc[data["Actual_Attendance"] > data["Participants"], "Actual_Attendance"] = data["Participants"]
    data["Reservation_Outcome"] = np.where(data["Usage_Status"].eq("Used"), "Completed", "Not Completed")
    data["Lead_Time_Category"] = pd.cut(
        data["Reservation_Lead_Days"],
        bins=[-np.inf, 2, 7, 14, np.inf],
        labels=["Last Minute", "Short Lead", "Medium Lead", "Early Booking"]
    ).astype("string")
    data["Attendance_Ratio"] = np.where(
        data["Participants"].fillna(0) > 0,
        (data["Actual_Attendance"] / data["Participants"]).fillna(0).round(4),
        0.0
    )
    return data

@st.cache_resource
def train_model(data):
    # Use only information available at submission time; exclude post-outcome fields.
    predictors = [
        "Facility_Type", "User_Type", "Time_Category",
        "Reservation_Lead_Days", "Participants", "Day_of_Week"
    ]
    model_data = data.dropna(subset=predictors + ["Reservation_Outcome"]).copy()
    X = pd.get_dummies(model_data[predictors], drop_first=True)
    y = model_data["Reservation_Outcome"].map({"Completed": 1, "Not Completed": 0})
    if y.nunique() < 2 or len(model_data) < 10:
        return None, None, None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, pred, pos_label=1, zero_division=0),
        "F1-score": f1_score(y_test, pred, pos_label=1, zero_division=0)
    }
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, X.columns.tolist(), metrics, cm, importance, model_data

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.info("Make sure facility_reservations.csv is in the same GitHub repository folder as app.py.")
    st.stop()

model, model_columns, metrics, cm, importances, model_data = train_model(df)

with st.sidebar:
    st.header("Filters")
    facilities = sorted(df["Facility_Type"].dropna().unique().tolist())
    users = sorted(df["User_Type"].dropna().unique().tolist())
    statuses = sorted(df["Reservation_Outcome"].dropna().unique().tolist())
    selected_facilities = st.multiselect("Facility type", facilities, default=facilities)
    selected_users = st.multiselect("User type", users, default=users)
    selected_outcomes = st.multiselect("Reservation outcome", statuses, default=statuses)

filtered = df[
    df["Facility_Type"].isin(selected_facilities)
    & df["User_Type"].isin(selected_users)
    & df["Reservation_Outcome"].isin(selected_outcomes)
].copy()

tab_overview, tab_data, tab_model, tab_predict = st.tabs([
    "📊 Overview", "🗂️ Reservation Records", "🤖 Model Evaluation", "🔎 Risk Check"
])

with tab_overview:
    st.subheader("Reservation Overview")
    total = len(filtered)
    not_completed = int(filtered["Reservation_Outcome"].eq("Not Completed").sum())
    rate = (not_completed / total * 100) if total else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reservations", f"{total:,}")
    c2.metric("Completed", f"{int(filtered['Reservation_Outcome'].eq('Completed').sum()):,}")
    c3.metric("Not Completed", f"{not_completed:,}")
    c4.metric("Non-completion rate", f"{rate:.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Reservations by facility")
        counts = filtered["Facility_Type"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Number of reservations")
        ax.set_ylabel("Facility type")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### Outcome by user type")
        cross = pd.crosstab(filtered["User_Type"], filtered["Reservation_Outcome"])
        fig, ax = plt.subplots(figsize=(7, 4))
        cross.plot(kind="bar", ax=ax)
        ax.set_xlabel("User type")
        ax.set_ylabel("Number of reservations")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Booking lead time")
    lead_counts = filtered["Lead_Time_Category"].value_counts().reindex(
        ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"], fill_value=0
    )
    st.bar_chart(lead_counts)

with tab_data:
    st.subheader("Reservation Records")
    st.write(f"Showing **{len(filtered):,}** of **{len(df):,}** records based on the sidebar filters.")
    search = st.text_input("Search reservation ID or purpose")
    display_df = filtered
    if search.strip():
        mask = (
            display_df["Reservation_ID"].astype(str).str.contains(search, case=False, na=False)
            | display_df["Purpose"].astype(str).str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]
    columns_to_show = [
        "Reservation_ID", "Facility_Type", "User_Type", "Time_Category",
        "Reservation_Lead_Days", "Participants", "Day_of_Week", "Purpose",
        "Usage_Status", "Reservation_Outcome", "Actual_Attendance", "Lead_Time_Category"
    ]
    st.dataframe(display_df[columns_to_show], use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered records as CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_facility_reservations.csv",
        mime="text/csv"
    )

with tab_model:
    st.subheader("Decision Tree Model Evaluation")
    st.warning("This model is for decision support only. Do not automatically reject reservations based on a risk score.")
    if model is None:
        st.error("There is not enough valid data or outcome variation to train the model.")
    else:
        cols = st.columns(4)
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value * 100:.1f}%")
        st.caption("Metrics are recalculated from a reproducible 80/20 stratified split (random_state=42). Positive class: Completed.")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Confusion matrix")
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: Not Completed", "Actual: Completed"],
                columns=["Predicted: Not Completed", "Predicted: Completed"]
            )
            st.dataframe(cm_df, use_container_width=True)
        with right:
            st.markdown("#### Most influential features")
            top = importances.head(8).sort_values()
            st.bar_chart(top)
        st.markdown("#### Model limitations")
        st.write("Predictions can be wrong and may reflect patterns in historical data. Use a risk flag to prompt a reminder or human review, not as an automatic cancellation decision.")

with tab_predict:
    st.subheader("Check a proposed reservation")
    st.write("Enter details available when a reservation is submitted. The model excludes fields that are only known after the event.")
    if model is None:
        st.error("Model could not be trained from this dataset.")
    else:
        with st.form("risk_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                facility = st.selectbox("Facility type", facilities)
                user_type = st.selectbox("User type", users)
            with col2:
                time_category = st.selectbox("Time category", sorted(df["Time_Category"].dropna().unique()))
                day = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            with col3:
                lead_days = st.number_input("Days booked in advance", min_value=0, max_value=365, value=7)
                participants = st.number_input("Expected participants", min_value=1, max_value=1000, value=20)
            submitted = st.form_submit_button("Estimate risk")
        if submitted:
            new_row = pd.DataFrame([{
                "Facility_Type": facility,
                "User_Type": user_type,
                "Time_Category": time_category,
                "Reservation_Lead_Days": lead_days,
                "Participants": participants,
                "Day_of_Week": day
            }])
            encoded = pd.get_dummies(new_row, drop_first=True).reindex(columns=model_columns, fill_value=0)
            probabilities = model.predict_proba(encoded)[0]
            classes = list(model.classes_)
            not_completed_probability = float(probabilities[classes.index(0)]) if 0 in classes else 0.0
            completed_probability = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
            a, b = st.columns(2)
            a.metric("Estimated non-completion risk", f"{not_completed_probability * 100:.1f}%")
            b.metric("Estimated completion likelihood", f"{completed_probability * 100:.1f}%")
            if not_completed_probability >= 0.65:
                st.warning("Higher estimated risk: consider a confirmation reminder or administrator review.")
            else:
                st.success("Lower estimated risk according to this model. Normal reservation procedures still apply.")
            st.caption("This is a statistical estimate, not a guarantee. Model results depend on the historical dataset and may be inaccurate for new cases.")
