from __future__ import annotations
from datetime import datetime
from pathlib import Path
import io
import pandas as pd
import streamlit as st
import plotly.express as px

from modules.data_processing import REQUIRED_COLUMNS, PREDICTORS, validate_schema, clean_reservations, quality_summary
from modules.exploratory_analysis import make_facility_chart, make_outcome_chart, make_user_chart, make_lead_chart
from modules.prediction import load_model_artifact, predict_one, predict_frame
from modules.dashboard import get_kpis

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "facility_reservations.csv"

st.set_page_config(page_title="Campus Facilities | Decision Support", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --navy:#14283f; --teal:#287d8e; --muted:#64748b; --paper:#f5f8fb; --line:#e1e8ef; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--paper); }
[data-testid="stHeader"] { background:rgba(245,248,251,.92); }
.block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1440px; }
[data-testid="stSidebar"] { background:#14283f; }
[data-testid="stSidebar"] * { color:#f4f7fb !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .55rem .65rem; border-radius:9px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(255,255,255,.09); }
.hero { background:linear-gradient(120deg,#14283f 0%,#1d5065 100%); border-radius:18px; padding:1.6rem 1.8rem; color:#fff; margin-bottom:1.2rem; box-shadow:0 10px 28px rgba(20,40,63,.12); }
.hero h1 { font-family:'Manrope',sans-serif; font-size:2rem; line-height:1.2; margin:0 0 .45rem 0; color:#fff; }
.hero p { color:#d8e6ee; margin:0; font-size:1rem; }
.eyebrow { text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; color:#a7d7de; margin-bottom:.55rem; }
.section-title { font-family:'Manrope',sans-serif; font-weight:800; color:#14283f; font-size:1.25rem; margin:.2rem 0 .2rem 0; }
.section-note { color:#64748b; font-size:.91rem; margin-bottom:1rem; }
.card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; }
.kpi-label { color:#64748b; font-size:.82rem; font-weight:600; margin-bottom:.3rem; }
.kpi-value { color:#14283f; font-family:'Manrope',sans-serif; font-size:1.8rem; font-weight:800; line-height:1.2; }
.kpi-foot { color:#64748b; font-size:.76rem; margin-top:.35rem; }
.pill { display:inline-block; border-radius:999px; padding:.2rem .6rem; font-size:.75rem; font-weight:700; background:#e6f3f4; color:#1d6672; }
.small-muted { color:#64748b; font-size:.82rem; }
hr { border-color:#e1e8ef; }
[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; box-shadow:0 2px 8px rgba(20,40,63,.025); }
[data-testid="stMetricLabel"] { color:#64748b; }
.stButton>button, .stDownloadButton>button { border-radius:9px; font-weight:700; border:1px solid #287d8e; }
div[data-testid="stForm"] { background:#fff; padding:1.1rem; border:1px solid var(--line); border-radius:14px; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_default_data():
    raw = pd.read_csv(DEFAULT_DATA)
    return raw, clean_reservations(raw)

@st.cache_resource(show_spinner=False)
def get_artifact():
    return load_model_artifact()

try:
    raw_default, cleaned_default = load_default_data()
    artifact = get_artifact()
except Exception as exc:
    st.error(f"The application could not initialize: {exc}")
    st.info("Check that the CSV and models/trained_model.joblib are included in the GitHub repository.")
    st.stop()

with st.sidebar:
    st.markdown("<div style='padding:.3rem 0 1rem 0'><div style='font-size:1.45rem;font-weight:800;font-family:Manrope,sans-serif'>CAMPUS<span style='color:#8dd4dc'>/</span>OPS</div><div style='font-size:.78rem;color:#c5d5e1'>Facility intelligence workspace</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"], label_visibility="visible")
    st.markdown("---")
    st.caption("SYSTEM STATUS")
    st.markdown("<span class='pill'>● Model artifact loaded</span>", unsafe_allow_html=True)
    st.caption("For academic decision support. Predictions must be reviewed by an authorized person.")

# The uploaded data is kept in session state so all modules can use the same validated data.
if "raw_data" not in st.session_state:
    st.session_state.raw_data = raw_default.copy()
    st.session_state.cleaned_data = cleaned_default.copy()
    st.session_state.data_source = "Bundled synthetic dataset"
raw_df = st.session_state.raw_data
clean_df = st.session_state.cleaned_data


def hero(kicker, title, subtitle):
    st.markdown(f"<div class='hero'><div class='eyebrow'>{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def section(title, note=None):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if note:
        st.markdown(f"<div class='section-note'>{note}</div>", unsafe_allow_html=True)


def kpi(label, value, foot):
    st.markdown(f"<div class='card'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div><div class='kpi-foot'>{foot}</div></div>", unsafe_allow_html=True)

if page == "Home":
    hero("CAMPUS FACILITY RESERVATION & MONITORING", "A clearer view of campus facility activity.", "Explore reservation records, understand historical patterns, and review model-based signals before making an administrative decision.")
    k = get_kpis(clean_df)
    cols = st.columns(4)
    with cols[0]: kpi("TOTAL RESERVATIONS", f"{k['total']:,}", "Observed records in the current dataset")
    with cols[1]: kpi("COMPLETED", f"{k['completed']:,}", "Observed status: Used")
    with cols[2]: kpi("NOT COMPLETED", f"{k['not_completed']:,}", "Observed status: Cancelled or No-Show")
    with cols[3]: kpi("NON-COMPLETION RATE", f"{k['noncompletion_rate']:.1f}%", "Historical rate, not a prediction")
    st.write("")
    left, right = st.columns([1.25, .75], gap="large")
    with left:
        section("Workspace overview", "Use the left navigation to open each required module.")
        st.markdown("""
        <div class='card'>
        <b>01 · Dataset</b><br><span class='small-muted'>Upload a CSV, validate the required schema, inspect missing values and duplicates, and download cleaned data.</span><br><br>
        <b>02 · Data Analysis</b><br><span class='small-muted'>Explore descriptive summaries and interactive charts of facilities, outcomes, user groups, and booking lead time.</span><br><br>
        <b>03 · Prediction</b><br><span class='small-muted'>Use the saved Task 7 Decision Tree model to estimate the outcome for a proposed reservation.</span><br><br>
        <b>04 · Dashboard</b><br><span class='small-muted'>Review observed KPIs alongside clearly labeled model estimates and a human-review queue.</span><br><br>
        <b>05 · System Design</b><br><span class='small-muted'>Review requirements, actors, use cases, and the decision-support workflow.</span>
        </div>
        """, unsafe_allow_html=True)
    with right:
        section("Model information")
        st.markdown(f"<div class='card'><div class='kpi-label'>SAVED MODEL</div><b>{artifact.get('model_version','Task 7 Decision Tree')}</b><hr><div class='kpi-label'>FEATURES USED</div><span class='small-muted'>{', '.join(artifact['predictors'])}</span><hr><div class='kpi-label'>VALIDATION SETUP</div><span class='small-muted'>80/20 stratified split · random_state=42</span><hr><div class='kpi-label'>CURRENT DATA SOURCE</div><span class='small-muted'>{st.session_state.data_source}</span></div>", unsafe_allow_html=True)
        st.warning("Predictions are estimates. They must not automatically reject a reservation or deny access to a facility.")

elif page == "Dataset":
    hero("MODULE 01", "Dataset upload & validation", "Validate the input before using it for analysis or decision support.")
    st.markdown("Upload a CSV containing the reservation fields used by the midterm exercise. Your uploaded dataset will be used by the other modules for this session.")
    upload = st.file_uploader("Upload reservation dataset (.csv)", type=["csv"], help="Required fields are listed below.")
    if upload is not None:
        try:
            candidate = pd.read_csv(upload)
            valid, issues = validate_schema(candidate)
            if valid:
                st.session_state.raw_data = candidate.copy()
                st.session_state.cleaned_data = clean_reservations(candidate)
                st.session_state.data_source = f"Uploaded file: {upload.name}"
                st.success(f"Validated successfully. {len(candidate):,} rows loaded.")
                raw_df, clean_df = st.session_state.raw_data, st.session_state.cleaned_data
            else:
                st.error("The file was not accepted because required columns are missing or the dataset is empty.")
                st.write("Issues:", issues)
        except Exception as exc:
            st.error(f"Could not read or validate this CSV: {exc}")
    else:
        st.info(f"Currently using: {st.session_state.data_source}")
    section("Validation results", "Missing values and duplicate IDs are reported from the raw input, before cleaning.")
    q = quality_summary(raw_df)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{q['rows']:,}")
    c2.metric("Columns", f"{q['columns']:,}")
    c3.metric("Missing cells", f"{q['missing_cells']:,}")
    c4.metric("Duplicate reservation IDs", f"{q['duplicate_ids']:,}")
    missing = raw_df.isna().sum().rename("Missing values").reset_index().rename(columns={"index":"Column"})
    left,right=st.columns(2)
    with left:
        section("Required schema")
        schema_df=pd.DataFrame({"Required column":REQUIRED_COLUMNS,"Present": [c in raw_df.columns for c in REQUIRED_COLUMNS]})
        st.dataframe(schema_df,use_container_width=True,hide_index=True)
    with right:
        section("Missing-value profile")
        st.dataframe(missing[missing["Missing values"]>0] if missing["Missing values"].gt(0).any() else missing.head(0),use_container_width=True,hide_index=True)
        st.caption("No missing values is a valid result; the table will be empty in that case.")
    section("Raw data preview")
    st.dataframe(raw_df.head(25),use_container_width=True,hide_index=True)
    section("Cleaned data preview")
    st.dataframe(clean_df.head(25),use_container_width=True,hide_index=True)
    st.download_button("Download cleaned CSV", clean_df.to_csv(index=False).encode("utf-8"), file_name="facility_reservations_cleaned.csv", mime="text/csv", type="primary")
    st.caption("Cleaning follows the preparation rules documented in the midterm notebook. Review the data-quality profile before relying on any result.")

elif page == "Data Analysis":
    hero("MODULE 02", "Exploratory data analysis", "Describe the historical records before interpreting any model prediction.")
    k = get_kpis(clean_df)
    c1,c2,c3=st.columns(3)
    c1.metric("Records analyzed",f"{len(clean_df):,}")
    c2.metric("Facility types",f"{clean_df['Facility_Type'].nunique():,}")
    c3.metric("User categories",f"{clean_df['User_Type'].nunique():,}")
    st.caption("All charts on this page describe observed historical records, not future predictions.")
    c1,c2=st.columns(2,gap="large")
    with c1: st.plotly_chart(make_facility_chart(clean_df),use_container_width=True)
    with c2: st.plotly_chart(make_outcome_chart(clean_df),use_container_width=True)
    c3,c4=st.columns(2,gap="large")
    with c3: st.plotly_chart(make_user_chart(clean_df),use_container_width=True)
    with c4: st.plotly_chart(make_lead_chart(clean_df),use_container_width=True)
    section("Descriptive statistics")
    numeric=[c for c in ["Reservation_Lead_Days","Participants","Actual_Attendance","Cancellation_Days_Before"] if c in clean_df.columns]
    st.dataframe(clean_df[numeric].describe().T.round(2),use_container_width=True)
    section("Category distribution")
    selected_col=st.selectbox("Choose a category",["Facility_Type","User_Type","Time_Category","Day_of_Week","Usage_Status","Lead_Time_Category"])
    dist=clean_df[selected_col].fillna("Missing").value_counts().rename_axis(selected_col).reset_index(name="Count")
    st.dataframe(dist,use_container_width=True,hide_index=True)

elif page == "Prediction":
    hero("MODULE 03", "Reservation prediction", "Estimate the outcome using only fields available when a reservation is submitted.")
    st.info("The saved model follows the Task 7 configuration: Decision Tree, max_depth=4, random_state=42. Post-outcome fields such as Usage_Status, Actual_Attendance, and Cancellation_Days_Before are not model inputs.")
    st.markdown("#### Enter proposed reservation details")
    facilities=sorted(clean_df["Facility_Type"].dropna().astype(str).unique().tolist())
    user_types=sorted(clean_df["User_Type"].dropna().astype(str).unique().tolist())
    time_types=sorted(clean_df["Time_Category"].dropna().astype(str).unique().tolist())
    days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    with st.form("prediction_form"):
        a,b,c=st.columns(3)
        with a:
            facility=st.selectbox("Facility type",facilities)
            user_type=st.selectbox("User type",user_types)
        with b:
            time_category=st.selectbox("Time category",time_types)
            day=st.selectbox("Day of week",days)
        with c:
            lead_days=st.number_input("Days booked in advance",min_value=0,max_value=365,value=7,step=1)
            participants=st.number_input("Expected participants",min_value=1,max_value=1000,value=20,step=1)
        submitted=st.form_submit_button("Run prediction",type="primary",use_container_width=True)
    if submitted:
        values={"Facility_Type":facility,"User_Type":user_type,"Time_Category":time_category,"Reservation_Lead_Days":float(lead_days),"Participants":float(participants),"Day_of_Week":day}
        result=predict_one(artifact,values)
        st.session_state.last_prediction={**result,"inputs":values,"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if st.session_state.get("last_prediction"):
        result=st.session_state.last_prediction
        st.markdown("---")
        section("Prediction result",f"Generated at {result['timestamp']} · {result['model_version']}")
        c1,c2,c3=st.columns(3)
        c1.metric("Estimated non-completion risk",f"{result['risk_probability']*100:.1f}%")
        c2.metric("Estimated completion likelihood",f"{result['completion_probability']*100:.1f}%")
        c3.metric("Predicted class",result["predicted_class"])
        if result["risk_probability"]>=.65:
            st.warning("Review recommended: consider confirming details or sending a reminder. This is not a reason to automatically reject the request.")
        elif result["risk_probability"]>=.40:
            st.info("Additional check suggested: verify reservation details and consider a confirmation reminder.")
        else:
            st.success("Lower estimated risk according to this model. Continue the normal reservation process.")
        st.markdown("**Inputs used**")
        st.dataframe(pd.DataFrame([result["inputs"]]),use_container_width=True,hide_index=True)
        st.caption("Probabilities are model estimates based on historical synthetic records and may be inaccurate for new cases. They are not guarantees or final decisions.")

elif page == "Dashboard":
    hero("MODULE 04", "Administrator decision-support dashboard", "Separate what happened in the dataset from what the model estimates.")
    k=get_kpis(clean_df)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Observed reservations",f"{k['total']:,}")
    c2.metric("Observed non-completion",f"{k['not_completed']:,}")
    c3.metric("Observed non-completion rate",f"{k['noncompletion_rate']:.1f}%")
    c4.metric("Model test accuracy",f"{artifact['metrics']['accuracy']*100:.1f}%")
    st.caption("Observed KPIs come from historical outcomes. Model accuracy comes from the saved model's held-out test split. Risk estimates below are model outputs.")
    left,right=st.columns(2,gap="large")
    with left: st.plotly_chart(make_facility_chart(clean_df),use_container_width=True)
    with right: st.plotly_chart(make_user_chart(clean_df),use_container_width=True)
    section("Model evaluation", "Calculated from the held-out split used for the saved Task 7 model.")
    m=artifact["metrics"]
    a,b,c,d=st.columns(4)
    a.metric("Accuracy",f"{m['accuracy']*100:.1f}%")
    b.metric("Precision",f"{m['precision']*100:.1f}%")
    c.metric("Recall",f"{m['recall']*100:.1f}%")
    d.metric("F1 score",f"{m['f1']*100:.1f}%")
    cm=pd.DataFrame(m["confusion_matrix"],index=["Actual: Not Completed","Actual: Completed"],columns=["Predicted: Not Completed","Predicted: Completed"])
    st.markdown("**Confusion matrix**")
    st.dataframe(cm,use_container_width=True)
    st.warning("The saved model's measured results are modest. Treat the model as an early-warning aid only; an authorized administrator must review the context and make the final decision.")
    section("Risk review queue", "The model estimates are applied to the available records for demonstration. They are not pending reservations or verified future outcomes.")
    with st.spinner("Calculating model estimates for the current dataset..."):
        queue=predict_frame(artifact,clean_df)
    high=queue[queue["Predicted_Risk_Percent"]>=65].sort_values("Predicted_Risk_Percent",ascending=False)
    q1,q2,q3=st.columns(3)
    q1.metric("High estimated risk",f"{len(high):,}","Risk estimate ≥ 65%")
    q2.metric("Medium estimate",f"{int(queue['Predicted_Risk_Percent'].between(40,64.999).sum()):,}","40% to <65%")
    q3.metric("Lower estimate",f"{int((queue['Predicted_Risk_Percent']<40).sum()):,}","Below 40%")
    show_cols=[c for c in ["Reservation_ID","Facility_Type","User_Type","Time_Category","Reservation_Lead_Days","Participants","Predicted_Risk_Percent","Predicted_Outcome","Review_Flag"] if c in queue.columns]
    st.dataframe(queue[show_cols].sort_values("Predicted_Risk_Percent",ascending=False).head(30),use_container_width=True,hide_index=True)
    st.download_button("Download risk review queue",queue[show_cols].to_csv(index=False).encode("utf-8"),file_name="reservation_risk_review_queue.csv",mime="text/csv")
    st.caption("Model version: " + artifact.get("model_version","Task 7 Decision Tree"))

elif page == "System Design":
    hero("MODULE 05", "Systems analysis & design", "Connect evidence from the data and model to requirements, use cases, and a responsible workflow.")
    section("Functional requirements", "The first three requirements are carried forward from Task 9 of the midterm notebook. Additional requirements support the required dataset and decision-support modules.")
    reqs=pd.DataFrame([
        {"ID":"FR-DM-01","Evidence / rationale":"The midterm EDA identified a high non-completion rate for last-minute reservations.","Requirement":"The system shall flag last-minute reservation requests for confirmation and display the flag to an authorized administrator."},
        {"ID":"FR-DM-02","Evidence / rationale":"The midterm EDA reported differences in observed non-completion by user type.","Requirement":"The system shall display observed reservation outcomes by user type so administrators can review patterns without automatically penalizing a group."},
        {"ID":"FR-DM-03","Evidence / rationale":"The Task 7 model estimates reservation outcomes from submission-time predictors.","Requirement":"The system shall display the estimated non-completion risk and route elevated estimates for human review."},
        {"ID":"FR-DM-04","Evidence / rationale":"Dataset quality checks are required before analysis and prediction.","Requirement":"The system shall validate required CSV columns and report missing values and duplicate reservation IDs before processing an uploaded dataset."},
        {"ID":"FR-DM-05","Evidence / rationale":"University management needs facility utilization information.","Requirement":"The system shall provide historical summaries of reservations by facility type, user type, booking lead time, and observed outcome."},
    ])
    st.dataframe(reqs,use_container_width=True,hide_index=True)
    section("Non-functional requirements")
    nfrs=pd.DataFrame([
        {"ID":"NFR-01 · Reliability","Requirement":"The system shall show a clear validation message when a CSV is empty, unreadable, or missing required columns."},
        {"ID":"NFR-02 · Usability","Requirement":"The system shall label historical facts and model predictions separately and provide readable charts, tables, and metric definitions."},
        {"ID":"NFR-03 · Responsible use","Requirement":"The system shall present risk estimates as decision support and shall not automatically reject a reservation based on a prediction."},
    ])
    st.dataframe(nfrs,use_container_width=True,hide_index=True)
    section("Actors & use case diagram")
    st.image(str(BASE_DIR/"assets"/"use_case_diagram.png"),use_container_width=True,caption="Campus Facility Reservation and Monitoring System — updated use case diagram")
    section("Decision-support workflow")
    st.markdown("""
    <div class='card' style='text-align:center;line-height:2.1'>
    <span class='pill'>1 · Validate dataset</span>　→　<span class='pill'>2 · Clean & analyze</span>　→　<span class='pill'>3 · Estimate outcome</span><br>
    ↓<br>
    <span class='pill'>4 · Flag for review when appropriate</span>　→　<span class='pill'>5 · Authorized administrator reviews context</span><br>
    ↓<br>
    <span class='pill'>6 · Human decision & follow-up</span>
    </div>
    """,unsafe_allow_html=True)
    st.caption("This demo does not authenticate user roles or persist a formal decision log. Those are limitations for a production campus system and should be documented in the deployment report.")

st.markdown("<hr><div class='small-muted' style='text-align:center'>Campus Facility Reservation & Monitoring System · SAD laboratory project · Synthetic academic dataset</div>",unsafe_allow_html=True)
