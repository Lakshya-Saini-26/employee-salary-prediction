from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ---------- Paths (App/ sits next to Data/ and Models/) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "Models", "RF_employees.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "Models", "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "..", "Models", "columns.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "Data", "Salary Data.csv")

# ---------- Load model artifacts (saved with joblib in the notebook) ----------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
model_columns = joblib.load(COLUMNS_PATH)  # exact column order used in training

# ---------- Load original data to build dropdown options ----------
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)

genders = sorted(df["Gender"].unique())                # ['Female', 'Male']
education_levels = sorted(df["Education Level"].unique())  # ["Bachelor's","Master's","PhD"]
job_titles = sorted(df["Job Title"].unique())           # all 174 job titles

# ---------- Recreate the SAME LabelEncoder mapping used in training ----------
# LabelEncoder().fit_transform() encodes classes alphabetically -> matches sorted()
gender_map = {label: idx for idx, label in enumerate(genders)}
education_map = {label: idx for idx, label in enumerate(education_levels)}


@app.route("/")
def home():
    return render_template(
        "index.html",
        genders=genders,
        education_levels=education_levels,
        job_titles=job_titles,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form.get("age"))
        gender = request.form.get("gender")
        education = request.form.get("education")
        years_experience = float(request.form.get("years_experience"))
        job_title = request.form.get("job_title")

        # ---- Build a single-row input matching training feature order/logic ----
        # Start with all model columns at 0
        input_row = {col: 0 for col in model_columns}

        # Numeric features
        input_row["Age"] = age
        input_row["Years of Experience"] = years_experience

        # Label-encoded categoricals (same alphabetical mapping as training)
        input_row["Gender"] = gender_map.get(gender, 0)
        input_row["Education Level"] = education_map.get(education, 0)

        # One-hot encoded Job Title (drop_first=True was used during training,
        # so if the selected title was the dropped baseline category, every
        # Job Title_* column correctly stays 0)
        job_col = f"Job Title_{job_title}"
        if job_col in input_row:
            input_row[job_col] = 1

        input_df = pd.DataFrame([input_row])
        input_df = input_df[model_columns]     # enforce exact training column order
        input_df = input_df.astype(int)        # matches X.astype(int) in the notebook

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        return render_template(
            "index.html",
            genders=genders,
            education_levels=education_levels,
            job_titles=job_titles,
            prediction_text=f"Predicted Salary: ${prediction:,.2f}",
            selected_age=age,
            selected_gender=gender,
            selected_education=education,
            selected_experience=years_experience,
            selected_job=job_title,
        )

    except Exception as e:
        return render_template(
            "index.html",
            genders=genders,
            education_levels=education_levels,
            job_titles=job_titles,
            prediction_text=f"Error: {e}",
        )


if __name__ == "__main__":
    app.run(debug=True)
