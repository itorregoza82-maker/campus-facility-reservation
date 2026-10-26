# Campus Facility Reservation & Monitoring Dashboard

A Streamlit dashboard built from the facility reservation laboratory dataset.

## Features
- Reservation overview and outcome charts
- Filterable reservation records and CSV export
- Decision Tree evaluation using a reproducible stratified 80/20 split
- Risk estimate for a proposed reservation
- Uses only submission-time fields for model predictions to reduce data leakage

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `facility_reservations.csv`.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
4. Click **Deploy** and wait for the build to finish.

## Important
This is an academic decision-support prototype. A risk estimate must not be used as an automatic reason to reject or cancel a reservation. Review predictions alongside human judgment and applicable campus policy.
