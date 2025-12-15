import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from io import StringIO
from pandas.api.types import is_numeric_dtype

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    auc,
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import random
import subprocess
import sys
def install_packages():
    """Install required packages if not already installed."""
    packages = [
        "streamlit",
        "pandas",
        "numpy",
        "plotly",
        "seaborn",
        "matplotlib",
        "scikit-learn",
    ]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()
# Initialize session state for model persistence
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {"svm": None, "logreg": None, "rf": None, "gb": None}
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "random_data" not in st.session_state:
    st.session_state.random_data = None
if "select_gender" not in st.session_state:
    st.session_state.select_gender = 0
if "select_senior" not in st.session_state:
    st.session_state.select_senior = 0
if "select_partner" not in st.session_state:
    st.session_state.select_partner = 0
if "select_dependents" not in st.session_state:
    st.session_state.select_dependents = 0
if "slider_tenure" not in st.session_state:
    st.session_state.slider_tenure = 12
if "select_phone" not in st.session_state:
    st.session_state.select_phone = 0
if "select_multiple" not in st.session_state:
    st.session_state.select_multiple = 0
if "select_internet" not in st.session_state:
    st.session_state.select_internet = "DSL"
if "select_security" not in st.session_state:
    st.session_state.select_security = 0
if "select_backup" not in st.session_state:
    st.session_state.select_backup = 0
if "select_device" not in st.session_state:
    st.session_state.select_device = 0
if "select_tech" not in st.session_state:
    st.session_state.select_tech = 0
if "select_tv" not in st.session_state:
    st.session_state.select_tv = 0
if "select_movies" not in st.session_state:
    st.session_state.select_movies = 0
if "select_contract" not in st.session_state:
    st.session_state.select_contract = "Month-to-month"
if "select_paperless" not in st.session_state:
    st.session_state.select_paperless = 0
if "select_payment" not in st.session_state:
    st.session_state.select_payment = "Electronic check"
if "input_monthly" not in st.session_state:
    st.session_state.input_monthly = 70.0
if "input_total" not in st.session_state:
    st.session_state.input_total = 1000.0

# Bank churn (churn.csv) model persistence
if "bank_trained_models" not in st.session_state:
    st.session_state.bank_trained_models = {"svm": None, "logreg": None, "rf": None, "gb": None}
if "bank_X_train" not in st.session_state:
    st.session_state.bank_X_train = None

# Bank churn prediction widget defaults
if "bank_gender" not in st.session_state:
    st.session_state.bank_gender = "Female"
if "bank_geography" not in st.session_state:
    st.session_state.bank_geography = "France"
if "bank_credit_score" not in st.session_state:
    st.session_state.bank_credit_score = 650
if "bank_age" not in st.session_state:
    st.session_state.bank_age = 40
if "bank_tenure" not in st.session_state:
    st.session_state.bank_tenure = 5
if "bank_balance" not in st.session_state:
    st.session_state.bank_balance = 50000.0
if "bank_num_products" not in st.session_state:
    st.session_state.bank_num_products = 1
if "bank_has_card" not in st.session_state:
    st.session_state.bank_has_card = 1
if "bank_is_active" not in st.session_state:
    st.session_state.bank_is_active = 1
if "bank_est_salary" not in st.session_state:
    st.session_state.bank_est_salary = 100000.0


def generate_random_customer():
    """Generate random customer data for prediction testing."""
    import random
    random_customer = {
        "gender": random.choice([0, 1]),
        "senior_citizen": random.choice([0, 1]),
        "partner": random.choice([0, 1]),
        "dependents": random.choice([0, 1]),
        "tenure": random.randint(0, 72),
        "phone_service": random.choice([0, 1]),
        "multiple_lines": random.choice([0, 1]),
        "internet_service": random.choice(["DSL", "Fiber optic", "No"]),
        "online_security": random.choice([0, 1]),
        "online_backup": random.choice([0, 1]),
        "device_protection": random.choice([0, 1]),
        "tech_support": random.choice([0, 1]),
        "streaming_tv": random.choice([0, 1]),
        "streaming_movies": random.choice([0, 1]),
        "contract": random.choice(["Month-to-month", "One year", "Two year"]),
        "paperless_billing": random.choice([0, 1]),
        "payment_method": random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
        "monthly_charges": round(random.uniform(20, 150), 2),
        "total_charges": round(random.uniform(100, 8000), 2),
    }
    return random_customer


def generate_random_bank_customer():
    """Generate random bank customer data for prediction testing."""

    return {
        "CreditScore": random.randint(350, 850),
        "Geography": random.choice(["France", "Spain", "Germany"]),
        "Gender": random.choice(["Female", "Male"]),
        "Age": random.randint(18, 92),
        "Tenure": random.randint(0, 10),
        "Balance": round(random.uniform(0, 250000), 2),
        "NumOfProducts": random.choice([1, 2, 3, 4]),
        "HasCrCard": random.choice([0, 1]),
        "IsActiveMember": random.choice([0, 1]),
        "EstimatedSalary": round(random.uniform(10000, 200000), 2),
    }


def preprocess_bank(df: pd.DataFrame):
    """Preprocess + feature engineer the bank churn dataset (churn.csv)."""
    df = df.copy()

    # Drop IDs
    df = df.drop(columns=[c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns])

    # Basic missing handling (numeric -> median)
    for col in df.columns:
        if col in ["Exited"]:
            continue
        if is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    # Encode Gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0).astype(int)

    # Geography one-hot
    if "Geography" in df.columns:
        geo_dummies = pd.get_dummies(df["Geography"], drop_first=True, dtype=int)
        df = pd.concat([df.drop(columns=["Geography"]), geo_dummies], axis=1)

    # Feature engineering
    # Customer Value Score (Balance + EstimatedSalary)
    if "Balance" in df.columns and "EstimatedSalary" in df.columns:
        df["Balance_to_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
        df["Is_Wealthy"] = (df["Balance"] > df["Balance"].quantile(0.75)).astype(int)

    # Age group binning
    if "Age" in df.columns:
        age_group = pd.cut(
            df["Age"],
            bins=[0, 30, 40, 50, 60, 100],
            labels=["18-30", "31-40", "41-50", "51-60", "60+"],
        )
        df["Age_Group"] = pd.Categorical(age_group).codes
        df["Is_Senior"] = (df["Age"] > 60).astype(int)

    # Tenure features
    if "Tenure" in df.columns:
        tenure_group = pd.cut(
            df["Tenure"],
            bins=[0, 1, 3, 5, 10, float("inf")],
            labels=["New", "Developing", "Mature", "Loyal", "Very_Loyal"],
            include_lowest=True,
        )
        df["Tenure_Group"] = pd.Categorical(tenure_group).codes
        df["Is_New_Customer"] = (df["Tenure"] <= 1).astype(int)

    # Credit risk
    if "CreditScore" in df.columns:
        credit_risk = pd.cut(
            df["CreditScore"],
            bins=[0, 400, 600, 750, 850, 1000],
            labels=["Very_High", "High", "Medium", "Low", "Very_Low"],
        )
        df["Credit_Risk"] = pd.Categorical(credit_risk).codes

    # Product engagement
    if "NumOfProducts" in df.columns:
        df["Num_Products"] = df["NumOfProducts"]
        df["Has_Multiple_Products"] = (df["Num_Products"] > 1).astype(int)

    # Activity * Tenure
    if "IsActiveMember" in df.columns and "Tenure" in df.columns:
        df["Activity_Tenure_Score"] = df["IsActiveMember"] * df["Tenure"]

    # Age * Products
    if "Age" in df.columns and "Num_Products" in df.columns:
        df["Age_Product_Interaction"] = df["Age"] * df["Num_Products"]

    # Balance risk
    if "Balance" in df.columns:
        df["Zero_Balance"] = (df["Balance"] == 0).astype(int)
        df["Low_Balance"] = (df["Balance"] < df["Balance"].quantile(0.25)).astype(int)

    # Ensure numeric where possible
    for col in df.columns:
        if col == "Exited":
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if "Exited" in df.columns:
        df["Exited"] = pd.to_numeric(df["Exited"], errors="coerce").fillna(0).astype(int)

    return df


def encode_bank_prediction_input(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_of_products,
    has_cr_card,
    is_active_member,
    estimated_salary,
    X_train,
):
    """Encode a single bank-customer row to match training features."""
    input_df = pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Geography": [geography],
            "Gender": [gender],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
        }
    )

    input_processed = preprocess_bank(input_df)

    for col in X_train.columns:
        if col not in input_processed.columns:
            input_processed[col] = 0
    input_processed = input_processed[X_train.columns]
    return input_processed
st.set_page_config(page_title="Churn BI Dashboard", layout="wide")

st.title("📊 Business Intelligence: Churn Prediction")

# 2. Sidebar - Upload & Settings
st.sidebar.header("User Input / Data")
uploaded_file = st.sidebar.file_uploader("Upload Telco CSV (optional)", type=["csv"])

# Controls
st.sidebar.header("Training Settings")
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", min_value=0, value=42, step=1)
run_svm = st.sidebar.checkbox("Run SVM", value=True)
run_logreg = st.sidebar.checkbox("Run Logistic Regression (with scaling)", value=True)
run_rf = st.sidebar.checkbox("Run Random Forest", value=True)
run_gb = st.sidebar.checkbox("Run Gradient Boosting", value=True)

st.sidebar.header("Actions")
train_button = st.sidebar.button("Run Training")

@st.fragment(run_every="1s")
@st.cache_data(show_spinner=False)
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    # Fallback to repo dataset if no upload
    return pd.read_csv("./telco_customer_churn.csv")


def dataframe_info_textframe(df: pd.DataFrame):
    buf = StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def preprocess(df: pd.DataFrame):
    df = df.copy()
    # Coerce TotalCharges -> numeric and impute median
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(value=df["TotalCharges"].median())

    # Consolidate "No internet service" into "No"
    for col in df.columns:
        try:
            uniques = df[col].dropna().unique().tolist()
        except Exception:
            continue
        if set(uniques) == set(["No internet service", "No", "Yes"]):
            df[col] = df[col].replace({"No internet service": "No"})

    # Consolidate "No phone service" into "No" and create Phoneservice from MultipleLines (per snippet)
    if "MultipleLines" in df.columns:
        df["Phoneservice"] = df["MultipleLines"].replace({"Yes": 1, "No": 1, "No phone service": 0})
    for col in df.columns:
        try:
            uniques = df[col].dropna().unique().tolist()
        except Exception:
            continue
        if set(uniques) == set(["No phone service", "No", "Yes"]):
            df[col] = df[col].replace({"No phone service": "No"})

    # Binary mappings
    for col in df.columns:
        try:
            uniques = sorted(pd.Series(df[col].dropna().unique()).astype(str).tolist())
        except Exception:
            continue
        if uniques == ["No", "Yes"]:
            df[col] = df[col].map({"No": 0, "Yes": 1})
        if uniques == ["Female", "Male"]:
            df[col] = df[col].map({"Male": 0, "Female": 1})

    # One-hot encoding for selected multi-class categories
    for needed in ["InternetService", "Contract", "PaymentMethod"]:
        if needed not in df.columns:
            # If missing, create a placeholder to keep pipeline consistent
            df[needed] = "Unknown"

    dummies = pd.get_dummies(
        df,
        dtype=int,
        columns=["InternetService", "Contract", "PaymentMethod"],
        drop_first=True,
    )

    # Ensure target exists and is numeric 0/1
    if "Churn" in dummies.columns and not is_numeric_dtype(dummies["Churn"]):
        # After earlier mapping, Churn may already be numeric; else map string
        if dummies["Churn"].dtype == object:
            dummies["Churn"] = dummies["Churn"].map({"No": 0, "Yes": 1})
    return dummies


def encode_prediction_input(
    gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines,
    internet_service, online_security, online_backup, device_protection, tech_support,
    streaming_tv, streaming_movies, contract, paperless_billing, payment_method,
    monthly_charges, total_charges, X_train
):
    """Encode prediction input to match training features."""
    # Create a single-row dataframe with the same structure as training data
    input_df = pd.DataFrame({
        "Gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "Tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
    })
    
    # Preprocess using the same function
    input_processed = preprocess(input_df)
    
    # Align columns with training features
    for col in X_train.columns:
        if col not in input_processed.columns:
            input_processed[col] = 0
    
    # Select only the columns used in training, in the same order
    input_processed = input_processed[X_train.columns]
    
    return input_processed


dataset_telco_tab, dataset_bank_tab = st.tabs(["Telco churn (telco_customer_churn.csv)", "Bank churn (churn.csv)"])

with dataset_telco_tab:
    st.subheader("Telco Customer Churn")

    df = load_data(uploaded_file)

    # Show raw data
    with st.expander("Data Preview"):
        st.dataframe(df.head())
        st.caption("Top 5 rows of the dataset.")

    with st.expander("DataFrame Info"):
        st.text(dataframe_info_textframe(df))

    df_processed = preprocess(df)

    missing_cols = []
    required_cols = ["Churn", "TotalCharges"]
    for c in required_cols:
        if c not in df_processed.columns:
            missing_cols.append(c)

    target_missing = "Churn" not in df_processed.columns

    # Tabs
    tab1, tab2, tab3 = st.tabs(["EDA Visuals", "Model Training", "Prediction"])

    with tab1:
        st.subheader("Exploratory Data Analysis")
        st.text("Select a coliumn for histogram visualization:")
        selected_col = st.selectbox("Numeric Columns", df.select_dtypes(include=[np.number]).columns.tolist())
        color_col = "Churn" if "Churn" in df.columns else None
        if selected_col is not None and color_col is not None:
            fig = px.histogram(df, x=selected_col, color=color_col, title=f"Churn Distribution by {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Histogram not available: required columns not found.")

    with tab2:
        st.subheader("Model Performance")

        if target_missing:
            st.error("Target column 'Churn' not found after preprocessing. Please check your input data.")
        else:
            # Define features/target
            X = df_processed.drop(columns=["Churn"]) if "Churn" in df_processed.columns else df_processed.copy()
            y = df_processed["Churn"] if "Churn" in df_processed.columns else None

            st.write(f"Shape after preprocessing — Features: {X.shape}, Target: {y.shape if y is not None else (0,)}")

            if train_button:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

                # SVM
                if run_svm:
                    with st.spinner("Training SVM..."):
                        svm = SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=random_state)
                        svm.fit(X_train, y_train)
                        st.session_state.trained_models["svm"] = svm
                        st.session_state.X_train = X_train
                        y_pred = svm.predict(X_test)
                        y_proba = svm.predict_proba(X_test)[:, 1]
                        st.markdown("### SVM Classification Report")
                        st.text(classification_report(y_test, y_pred))

                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_auc = auc(fpr, tpr)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot([0, 1], [0, 1], "k--", label="Random")
                        ax.plot(fpr, tpr, label=f"SVM (AUC = {roc_auc:.3f})")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve — SVM")
                        ax.legend()
                        st.pyplot(fig)

                # Logistic Regression with scaling + GridSearchCV
                if run_logreg:
                    with st.spinner("Training Logistic Regression (with scaling and CV)..."):
                        steps = [("scaler", StandardScaler()), ("logreg", LogisticRegression())]
                        pipeline = Pipeline(steps)
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)
                        st.markdown("### Logistic Regression — Baseline")
                        st.text(classification_report(y_test, y_pred))

                        conf = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(4.5, 3.5))
                        sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title("Confusion Matrix — Logistic Regression")
                        ax.set_xlabel("Predicted Label")
                        ax.set_ylabel("True Label")
                        st.pyplot(fig)

                        param_grid = {"logreg__max_iter": [500, 600, 700, 800, 900, 1000]}
                        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
                        pipeline_cv = GridSearchCV(pipeline, param_grid, cv=kf)
                        pipeline_cv.fit(X_train, y_train)

                        st.session_state.trained_models["logreg"] = pipeline_cv
                        st.session_state.X_train = X_train

                        y_pred = pipeline_cv.predict(X_test)
                        y_pred_proba = pipeline_cv.predict_proba(X_test)[:, 1]
                        st.markdown("### Logistic Regression — GridSearchCV Results")
                        st.write(f"Best Params: {pipeline_cv.best_params_}")
                        st.write(f"Accuracy: {pipeline_cv.score(X_test, y_test):.4f}")
                        st.write(f"Precision (Churn): {precision_score(y_test, y_pred):.4f}")
                        st.write(f"Recall (Churn): {recall_score(y_test, y_pred):.4f}")
                        st.write(f"F1-Score (Churn): {f1_score(y_test, y_pred):.4f}")

                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                        roc_auc = auc(fpr, tpr)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
                        ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve — Logistic Regression")
                        ax.legend()
                        st.pyplot(fig)

                # Random Forest
                if run_rf:
                    with st.spinner("Training Random Forest..."):
                        rf = RandomForestClassifier(random_state=random_state, n_estimators=100, max_depth=None)
                        rf.fit(X_train, y_train)
                        st.session_state.trained_models["rf"] = rf
                        st.session_state.X_train = X_train
                        y_pred_rf = rf.predict(X_test)
                        y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
                        st.markdown("### Random Forest Classifier Evaluation")
                        st.write(f"Accuracy: {rf.score(X_test, y_test):.4f}")
                        st.write(f"Precision (Churn): {precision_score(y_test, y_pred_rf):.4f}")
                        st.write(f"Recall (Churn): {recall_score(y_test, y_pred_rf):.4f}")
                        st.write(f"F1-Score (Churn): {f1_score(y_test, y_pred_rf):.4f}")

                        conf_rf = confusion_matrix(y_test, y_pred_rf)
                        fig, ax = plt.subplots(figsize=(4.5, 3.5))
                        sns.heatmap(conf_rf, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title("Random Forest Confusion Matrix")
                        ax.set_xlabel("Predicted Label")
                        ax.set_ylabel("True Label")
                        st.pyplot(fig)

                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_rf)
                        roc_auc = auc(fpr, tpr)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot([0, 1], [0, 1], "k--", label="Random")
                        ax.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.3f})")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve — Random Forest")
                        ax.legend()
                        st.pyplot(fig)

                # Gradient Boosting
                if run_gb:
                    with st.spinner("Training Gradient Boosting..."):
                        gb = GradientBoostingClassifier(random_state=random_state)
                        gb.fit(X_train, y_train)
                        st.session_state.trained_models["gb"] = gb
                        st.session_state.X_train = X_train
                        y_pred_gb = gb.predict(X_test)
                        y_pred_proba_gb = gb.predict_proba(X_test)[:, 1]
                        st.markdown("### Gradient Boosting Classifier Evaluation")
                        st.write(f"Accuracy: {gb.score(X_test, y_test):.4f}")
                        st.write(f"Precision (Churn): {precision_score(y_test, y_pred_gb):.4f}")
                        st.write(f"Recall (Churn): {recall_score(y_test, y_pred_gb):.4f}")
                        st.write(f"F1-Score (Churn): {f1_score(y_test, y_pred_gb):.4f}")

                        conf_gb = confusion_matrix(y_test, y_pred_gb)
                        fig, ax = plt.subplots(figsize=(4.5, 3.5))
                        sns.heatmap(conf_gb, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title("Gradient Boosting Confusion Matrix")
                        ax.set_xlabel("Predicted Label")
                        ax.set_ylabel("True Label")
                        st.pyplot(fig)

                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_gb)
                        roc_auc = auc(fpr, tpr)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot([0, 1], [0, 1], "k--", label="Random")
                        ax.plot(fpr, tpr, label=f"Gradient Boosting (AUC = {roc_auc:.3f})")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve — Gradient Boosting")
                        ax.legend()
                        st.pyplot(fig)

    with tab3:
        st.subheader("Real-time Prediction (Demo)")
        st.info("This section can be wired to the trained pipeline to predict individual customers once feature inputs are standardized to match training features.")
        # Random data button
        col_button = st.columns([1, 5])
        with col_button[0]:
            if st.button("🎲 Generate Random", use_container_width=True):
                random_data = generate_random_customer()
                st.session_state.select_gender = random_data["gender"]
                st.session_state.select_senior = random_data["senior_citizen"]
                st.session_state.select_partner = random_data["partner"]
                st.session_state.select_dependents = random_data["dependents"]
                st.session_state.slider_tenure = random_data["tenure"]
                st.session_state.select_phone = random_data["phone_service"]
                st.session_state.select_multiple = random_data["multiple_lines"]
                st.session_state.select_internet = random_data["internet_service"]
                st.session_state.select_security = random_data["online_security"]
                st.session_state.select_backup = random_data["online_backup"]
                st.session_state.select_device = random_data["device_protection"]
                st.session_state.select_tech = random_data["tech_support"]
                st.session_state.select_tv = random_data["streaming_tv"]
                st.session_state.select_movies = random_data["streaming_movies"]
                st.session_state.select_contract = random_data["contract"]
                st.session_state.select_paperless = random_data["paperless_billing"]
                st.session_state.select_payment = random_data["payment_method"]
                st.session_state.input_monthly = random_data["monthly_charges"]
                st.session_state.input_total = random_data["total_charges"]
                st.session_state.random_data = random_data

        with col_button[1]:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.select_gender = 0
                st.session_state.select_senior = 0
                st.session_state.select_partner = 0
                st.session_state.select_dependents = 0
                st.session_state.slider_tenure = 12
                st.session_state.select_phone = 0
                st.session_state.select_multiple = 0
                st.session_state.select_internet = "DSL"
                st.session_state.select_security = 0
                st.session_state.select_backup = 0
                st.session_state.select_device = 0
                st.session_state.select_tech = 0
                st.session_state.select_tv = 0
                st.session_state.select_movies = 0
                st.session_state.select_contract = "Month-to-month"
                st.session_state.select_paperless = 0
                st.session_state.select_payment = "Electronic check"
                st.session_state.input_monthly = 70.0
                st.session_state.input_total = 1000.0
                st.session_state.random_data = None

        # Example inputs (non-functional placeholder without a persisted model)
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender",
                [0, 1],
                format_func=lambda x: "Male" if x == 1 else "Female",
                key="select_gender",
            )
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], key="select_senior")
            partner = st.selectbox(
                "Partner",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="select_partner",
            )
            dependents = st.selectbox(
                "Dependents",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="select_dependents",
            )
            tenure = st.slider("Tenure (Months)", 0, 72, key="slider_tenure")
            phone_service = st.selectbox(
                "Phone Service",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="select_phone",
            )
            multiple_lines = st.selectbox(
                "Multiple Lines",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="select_multiple",
            )

        with col2:
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
                key="select_internet",
            )
            online_security = st.selectbox(
                "Online Security",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="select_security",
            )
            online_backup = st.selectbox(
                "Online Backup",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="select_backup",
            )
            device_protection = st.selectbox(
                "Device Protection",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="select_device",
            )
            tech_support = st.selectbox(
                "Tech Support",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="select_tech",
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="select_tv",
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="select_movies",
            )

        with col3:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
                key="select_contract",
            )
            paperless_billing = st.selectbox(
                "Paperless Billing",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="select_paperless",
            )
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                key="select_payment",
            )
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, key="input_monthly")
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, key="input_total")

        if st.button("Predict Churn"):
            if st.session_state.X_train is None:
                st.error("❌ No trained model found. Please train a model in the 'Model Training' tab first.")
            else:
                try:
                    input_encoded = encode_prediction_input(
                        gender,
                        senior_citizen,
                        partner,
                        dependents,
                        tenure,
                        phone_service,
                        multiple_lines,
                        internet_service,
                        online_security,
                        online_backup,
                        device_protection,
                        tech_support,
                        streaming_tv,
                        streaming_movies,
                        contract,
                        paperless_billing,
                        payment_method,
                        monthly_charges,
                        total_charges,
                        st.session_state.X_train,
                    )

                    st.subheader("📊 Model Predictions")
                    st.write("**Threshold: 50% (≥50% = Stay, <50% = Churn)**")

                    col_models = st.columns(2)
                    predictions_dict = {}

                    if st.session_state.trained_models["svm"] is not None:
                        try:
                            svm_model = st.session_state.trained_models["svm"]
                            svm_proba = svm_model.predict_proba(input_encoded)[0][1] * 100
                            svm_prediction = "🟢 STAY" if svm_proba >= 50 else "🔴 CHURN"
                            predictions_dict["SVM"] = (svm_proba, svm_prediction)

                            with col_models[0]:
                                st.write("**SVM Model**")
                                st.metric("Churn Probability", f"{svm_proba:.2f}%", delta=f"{svm_prediction}")
                        except Exception as e:
                            st.warning(f"SVM prediction error: {str(e)}")

                    if st.session_state.trained_models["logreg"] is not None:
                        try:
                            logreg_model = st.session_state.trained_models["logreg"]
                            logreg_proba = logreg_model.predict_proba(input_encoded)[0][1] * 100
                            logreg_prediction = "🟢 STAY" if logreg_proba >= 50 else "🔴 CHURN"
                            predictions_dict["Logistic Regression"] = (logreg_proba, logreg_prediction)

                            with col_models[1]:
                                st.write("**Logistic Regression**")
                                st.metric(
                                    "Churn Probability",
                                    f"{logreg_proba:.2f}%",
                                    delta=f"{logreg_prediction}",
                                )
                        except Exception as e:
                            st.warning(f"Logistic Regression prediction error: {str(e)}")

                    if st.session_state.trained_models["rf"] is not None:
                        try:
                            rf_model = st.session_state.trained_models["rf"]
                            rf_proba = rf_model.predict_proba(input_encoded)[0][1] * 100
                            rf_prediction = "🟢 STAY" if rf_proba >= 50 else "🔴 CHURN"
                            predictions_dict["Random Forest"] = (rf_proba, rf_prediction)

                            with col_models[0]:
                                st.write("**Random Forest**")
                                st.metric("Churn Probability", f"{rf_proba:.2f}%", delta=f"{rf_prediction}")
                        except Exception as e:
                            st.warning(f"Random Forest prediction error: {str(e)}")

                    if st.session_state.trained_models["gb"] is not None:
                        try:
                            gb_model = st.session_state.trained_models["gb"]
                            gb_proba = gb_model.predict_proba(input_encoded)[0][1] * 100
                            gb_prediction = "🟢 STAY" if gb_proba >= 50 else "🔴 CHURN"
                            predictions_dict["Gradient Boosting"] = (gb_proba, gb_prediction)

                            with col_models[1]:
                                st.write("**Gradient Boosting**")
                                st.metric("Churn Probability", f"{gb_proba:.2f}%", delta=f"{gb_prediction}")
                        except Exception as e:
                            st.warning(f"Gradient Boosting prediction error: {str(e)}")

                    if predictions_dict:
                        st.divider()
                        st.write("**Summary**")
                        for model_name, (proba, pred) in predictions_dict.items():
                            st.write(f"- **{model_name}**: {pred} ({proba:.2f}%)")
                    else:
                        st.warning("No trained models available. Please train models in the 'Model Training' tab.")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.info("Make sure all input features are properly formatted and match the training data.")


with dataset_bank_tab:
    st.subheader("Bank Customer Churn (churn.csv)")
    try:
        bank_df_raw = pd.read_csv("./churn.csv")
    except Exception as e:
        st.error(f"Failed to load churn.csv: {str(e)}")
        st.stop()

    with st.expander("Data Preview"):
        st.dataframe(bank_df_raw.head())
        st.caption("Top 5 rows of churn.csv")

    with st.expander("DataFrame Info"):
        st.text(dataframe_info_textframe(bank_df_raw))

    with st.expander("Categorical Unique Values"):
        obj_cols = bank_df_raw.select_dtypes(include=["object"]).columns.tolist()
        if not obj_cols:
            st.write("No object columns found.")
        else:
            for col in obj_cols:
                st.write(f"{col}: {bank_df_raw[col].dropna().unique().tolist()}")

    with st.expander("Missing / Duplicates"):
        st.write("Missing values per column")
        st.dataframe(bank_df_raw.isna().sum())
        st.write(f"Duplicate rows: {int(bank_df_raw.duplicated().sum())}")

    bank_df = preprocess_bank(bank_df_raw)
    bank_target_missing = "Exited" not in bank_df.columns

    bank_tab1, bank_tab2, bank_tab3 = st.tabs(["EDA Visuals", "Model Training", "Prediction"])

    with bank_tab1:
        st.subheader("Exploratory Data Analysis")
        numeric_cols = bank_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "Exited"]
        if numeric_cols:
            selected_col = st.selectbox("Numeric Columns", numeric_cols, key="bank_hist_col")
            if selected_col:
                fig = px.histogram(
                    bank_df,
                    x=selected_col,
                    color="Exited" if "Exited" in bank_df.columns else None,
                    title=f"Exited Distribution by {selected_col}",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for histogram.")

    with bank_tab2:
        st.subheader("Model Performance")
        if bank_target_missing:
            st.error("Target column 'Exited' not found after preprocessing.")
        else:
            Xb = bank_df.drop(columns=["Exited"])
            yb = bank_df["Exited"]
            st.write(f"Shape after preprocessing — Features: {Xb.shape}, Target: {yb.shape}")

            train_bank_button = st.button("Run Bank Training", key="train_bank")
            if train_bank_button:
                X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
                    Xb, yb, test_size=test_size, random_state=random_state
                )

                st.session_state.bank_X_train = X_train_b

                # SVM
                if run_svm:
                    with st.spinner("Training Bank SVM..."):
                        svm_b = Pipeline(
                            [("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, random_state=random_state))]
                        )
                        svm_b.fit(X_train_b, y_train_b)
                        st.session_state.bank_trained_models["svm"] = svm_b
                        y_pred = svm_b.predict(X_test_b)
                        y_proba = svm_b.predict_proba(X_test_b)[:, 1]
                        st.markdown("### Bank SVM Classification Report")
                        st.text(classification_report(y_test_b, y_pred))

                        fpr, tpr, _ = roc_curve(y_test_b, y_proba)
                        roc_auc = auc(fpr, tpr)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot([0, 1], [0, 1], "k--", label="Random")
                        ax.plot(fpr, tpr, label=f"SVM (AUC = {roc_auc:.3f})")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve — Bank SVM")
                        ax.legend()
                        st.pyplot(fig)

                # Logistic Regression
                if run_logreg:
                    with st.spinner("Training Bank Logistic Regression (with scaling and CV)..."):
                        pipeline = Pipeline(
                            [("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000))]
                        )
                        param_grid = {"logreg__max_iter": [500, 600, 700, 800, 900, 1000]}
                        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
                        pipeline_cv = GridSearchCV(pipeline, param_grid, cv=kf)
                        pipeline_cv.fit(X_train_b, y_train_b)

                        st.session_state.bank_trained_models["logreg"] = pipeline_cv
                        y_pred = pipeline_cv.predict(X_test_b)
                        y_proba = pipeline_cv.predict_proba(X_test_b)[:, 1]
                        st.markdown("### Bank Logistic Regression — GridSearchCV")
                        st.write(f"Best Params: {pipeline_cv.best_params_}")
                        st.text(classification_report(y_test_b, y_pred))

                        conf = confusion_matrix(y_test_b, y_pred)
                        fig, ax = plt.subplots(figsize=(4.5, 3.5))
                        sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title("Confusion Matrix — Bank Logistic Regression")
                        ax.set_xlabel("Predicted Label")
                        ax.set_ylabel("True Label")
                        st.pyplot(fig)

                        fpr, tpr, _ = roc_curve(y_test_b, y_proba)
                        roc_auc = auc(fpr, tpr)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot([0, 1], [0, 1], "k--", label="Random")
                        ax.plot(fpr, tpr, label=f"LogReg (AUC = {roc_auc:.3f})")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve — Bank Logistic Regression")
                        ax.legend()
                        st.pyplot(fig)

                # Random Forest
                if run_rf:
                    with st.spinner("Training Bank Random Forest..."):
                        rf_b = RandomForestClassifier(random_state=random_state, n_estimators=200)
                        rf_b.fit(X_train_b, y_train_b)
                        st.session_state.bank_trained_models["rf"] = rf_b
                        y_pred = rf_b.predict(X_test_b)
                        y_proba = rf_b.predict_proba(X_test_b)[:, 1]
                        st.markdown("### Bank Random Forest")
                        st.text(classification_report(y_test_b, y_pred))

                        conf = confusion_matrix(y_test_b, y_pred)
                        fig, ax = plt.subplots(figsize=(4.5, 3.5))
                        sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title("Confusion Matrix — Bank Random Forest")
                        ax.set_xlabel("Predicted Label")
                        ax.set_ylabel("True Label")
                        st.pyplot(fig)

                        fpr, tpr, _ = roc_curve(y_test_b, y_proba)
                        roc_auc = auc(fpr, tpr)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot([0, 1], [0, 1], "k--", label="Random")
                        ax.plot(fpr, tpr, label=f"RF (AUC = {roc_auc:.3f})")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve — Bank Random Forest")
                        ax.legend()
                        st.pyplot(fig)

                # Gradient Boosting
                if run_gb:
                    with st.spinner("Training Bank Gradient Boosting..."):
                        gb_b = GradientBoostingClassifier(random_state=random_state)
                        gb_b.fit(X_train_b, y_train_b)
                        st.session_state.bank_trained_models["gb"] = gb_b
                        y_pred = gb_b.predict(X_test_b)
                        y_proba = gb_b.predict_proba(X_test_b)[:, 1]
                        st.markdown("### Bank Gradient Boosting")
                        st.text(classification_report(y_test_b, y_pred))

                        conf = confusion_matrix(y_test_b, y_pred)
                        fig, ax = plt.subplots(figsize=(4.5, 3.5))
                        sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title("Confusion Matrix — Bank Gradient Boosting")
                        ax.set_xlabel("Predicted Label")
                        ax.set_ylabel("True Label")
                        st.pyplot(fig)

                        fpr, tpr, _ = roc_curve(y_test_b, y_proba)
                        roc_auc = auc(fpr, tpr)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot([0, 1], [0, 1], "k--", label="Random")
                        ax.plot(fpr, tpr, label=f"GB (AUC = {roc_auc:.3f})")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve — Bank Gradient Boosting")
                        ax.legend()
                        st.pyplot(fig)

    with bank_tab3:
        st.subheader("Real-time Prediction (Demo)")

        col_button = st.columns([1, 5])
        with col_button[0]:
            if st.button("🎲 Generate Random", use_container_width=True, key="bank_random"):
                rd = generate_random_bank_customer()
                st.session_state.bank_credit_score = rd["CreditScore"]
                st.session_state.bank_geography = rd["Geography"]
                st.session_state.bank_gender = rd["Gender"]
                st.session_state.bank_age = rd["Age"]
                st.session_state.bank_tenure = rd["Tenure"]
                st.session_state.bank_balance = rd["Balance"]
                st.session_state.bank_num_products = rd["NumOfProducts"]
                st.session_state.bank_has_card = rd["HasCrCard"]
                st.session_state.bank_is_active = rd["IsActiveMember"]
                st.session_state.bank_est_salary = rd["EstimatedSalary"]

        with col_button[1]:
            if st.button("🔄 Clear", use_container_width=True, key="bank_clear"):
                st.session_state.bank_gender = "Female"
                st.session_state.bank_geography = "France"
                st.session_state.bank_credit_score = 650
                st.session_state.bank_age = 40
                st.session_state.bank_tenure = 5
                st.session_state.bank_balance = 50000.0
                st.session_state.bank_num_products = 1
                st.session_state.bank_has_card = 1
                st.session_state.bank_is_active = 1
                st.session_state.bank_est_salary = 100000.0

        c1, c2 = st.columns(2)
        with c1:
            credit_score = st.number_input("Credit Score", 300, 1000, key="bank_credit_score")
            geography = st.selectbox("Geography", ["France", "Spain", "Germany"], key="bank_geography")
            gender = st.selectbox("Gender", ["Female", "Male"], key="bank_gender")
            age = st.number_input("Age", 18, 100, key="bank_age")
            tenure = st.slider("Tenure (Years)", 0, 10, key="bank_tenure")

        with c2:
            balance = st.number_input("Balance", 0.0, 300000.0, step=100.0, key="bank_balance")
            num_products = st.slider("Num Of Products", 1, 4, key="bank_num_products")
            has_card = st.selectbox("Has Credit Card", [0, 1], key="bank_has_card")
            is_active = st.selectbox("Is Active Member", [0, 1], key="bank_is_active")
            est_salary = st.number_input("Estimated Salary", 0.0, 300000.0, step=100.0, key="bank_est_salary")

        if st.button("Predict Exited", key="bank_predict"):
            if st.session_state.bank_X_train is None:
                st.error("❌ No trained bank model found. Train a model in the 'Model Training' tab first.")
            else:
                try:
                    input_encoded = encode_bank_prediction_input(
                        credit_score,
                        geography,
                        gender,
                        age,
                        tenure,
                        balance,
                        num_products,
                        has_card,
                        is_active,
                        est_salary,
                        st.session_state.bank_X_train,
                    )

                    st.subheader("📊 Model Predictions")
                    st.write("**Threshold: 50% (≥50% = Exited, <50% = Stayed)**")

                    col_models = st.columns(2)
                    preds = {}

                    for name, model in st.session_state.bank_trained_models.items():
                        if model is None:
                            continue
                        try:
                            proba = model.predict_proba(input_encoded)[0][1] * 100
                            pred = "🔴 EXITED" if proba >= 50 else "🟢 STAYED"
                            preds[name.upper()] = (proba, pred)
                        except Exception as e:
                            st.warning(f"{name} prediction error: {str(e)}")

                    if preds:
                        items = list(preds.items())
                        for idx, (model_name, (proba, pred)) in enumerate(items):
                            with col_models[idx % 2]:
                                st.write(f"**{model_name}**")
                                st.metric("Exit Probability", f"{proba:.2f}%", delta=pred)

                        st.divider()
                        st.write("**Summary**")
                        for model_name, (proba, pred) in preds.items():
                            st.write(f"- **{model_name}**: {pred} ({proba:.2f}%)")
                    else:
                        st.warning("No trained bank models available. Please train models in the 'Model Training' tab.")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.info("Make sure all input features are properly formatted and match the training data.")
