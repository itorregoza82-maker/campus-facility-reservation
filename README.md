# Campus Facility Reservation & Monitoring System

A Streamlit decision-support prototype for the SAD laboratory exercise. The interface separates **observed historical data** from **model estimates** and keeps a human administrator responsible for final decisions.

## Required modules

1. **Dataset** — CSV upload, required-column checks, missing-value and duplicate-ID summary, data preview, cleaned CSV download.
2. **Data Analysis** — descriptive statistics and interactive charts for facility counts, observed outcomes, user categories, and booking lead time.
3. **Prediction** — uses the saved Task 7 Decision Tree artifact and only submission-time fields.
4. **Dashboard** — observed KPIs, held-out model metrics, risk estimate queue, and download.
5. **System Design** — functional/non-functional requirements, actors, UML use case diagram, and decision-support workflow.

The **Home** page is the landing page for navigation; it is not counted as one of the five required modules.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — navigation and page layout
- `modules/data_processing.py` — CSV validation, cleaning, feature engineering
- `modules/exploratory_analysis.py` — EDA chart helpers
- `modules/prediction.py` — saved-model loading and inference
- `modules/dashboard.py` — observed KPI helpers
- `models/trained_model.joblib` — saved Task 7 Decision Tree artifact
- `assets/use_case_diagram.png` — updated UML use case diagram
- `facility_reservations.csv` — supplied synthetic dataset
- `facility_reservations_cleaned.csv` — cleaned dataset output

## Model and data notes

The saved model uses the Task 7 settings documented in `LABORATORY_EXAMINATION.ipynb`: `DecisionTreeClassifier(max_depth=4, random_state=42)`, an 80/20 stratified split, and the six pre-submission predictors `Facility_Type`, `User_Type`, `Time_Category`, `Reservation_Lead_Days`, `Participants`, and `Day_of_Week`. Post-outcome fields such as `Usage_Status`, `Actual_Attendance`, and `Cancellation_Days_Before` are excluded from model inputs.

The displayed metrics are calculated from the saved artifact's held-out split; they are not manually typed or assumed. Model results can differ if the source dataset or preprocessing changes. Re-run the original midterm notebook and save the fitted model if your instructor expects the exact serialized artifact generated in your own Colab session.

## Responsible use and limitations

- Predictions are estimates, not guarantees or authorizations to reject reservations.
- A risk flag should prompt confirmation or authorized human review.
- This is a demonstration using synthetic academic data, not a production booking system.
- The app does not implement authentication, role-based access, persistent decision logs, or live database synchronization. Document these limitations in the deployment report.
- Do not upload personal or sensitive real-student data to a public app.

## Deployment checklist

- Commit all project files to GitHub, including the `modules`, `models`, and `assets` folders.
- In Streamlit Community Cloud, set `app.py` as the entry point.
- Test valid and invalid CSV uploads, prediction inputs, charts, and all five modules.
- Capture screenshots of each module and five sample predictions for submission.
