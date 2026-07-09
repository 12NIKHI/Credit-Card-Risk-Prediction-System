# 💳 Credit Card Customer Behaviour Prediction
End‑to‑end machine learning project to predict whether a credit card customer is Good or Bad, with a full MLOps‑style pipeline and Flask deployment.

📌 Project Overview
This project builds a production‑ready ML pipeline that classifies credit card customers into Good and Bad risk categories based on their demographic and financial attributes.
It covers everything from raw CSV data to a deployed web app with real‑time predictions.

🎯 Business Objective
Identify risky credit card customers before default.
Reduce financial losses due to bad debt and fraud.
Improve credit approval and limit decisions using data‑driven insights.

🗂 Project Structure



creditcard.csv          # Raw dataset
main.py                 # End-to-end ML pipeline entry script
random_sample.py        # Missing value imputation (Random Sample)
var_out.py              # Variable transformation & outlier handling
feature_selection.py    # Constant/quasi-constant & correlation-based feature selection
data_balance.py         # Class imbalance handling (SMOTE)
model_training.py       # Model training & evaluation utilities
log_code.py             # Centralized logging configuration
app.py                  # Flask application for deployment
index.html              # Frontend UI (Bootstrap form)
requirements.txt        # Python dependencies
Procfile                # Process definition for production hosting
credit_card.pkl         # Saved final ML model
scalar.pkl              # Saved StandardScaler
🔄 Machine Learning Pipeline


1️⃣ Data Cleaning
Removed invalid or inconsistent target labels (non Good/Bad entries).
Fixed incorrect data types (e.g., numeric columns stored as object).
Dropped unused or duplicate columns such as derived income fields.

2️⃣ Missing Value Treatment
Applied Random Sample Imputation per column:
Each missing value is replaced by a random sample from existing non‑missing values in the same feature, preserving the original distribution.

3️⃣ Feature Transformation & Outliers
Yeo–Johnson transformation on skewed numeric features to stabilize variance.
IQR‑based trimming to cap extreme outliers while keeping most observations.

4️⃣ Feature Selection
Removed constant and quasi‑constant features using VarianceThreshold.
Applied hypothesis‑test / Pearson correlation filtering to drop features with weak relationship to the target.

5️⃣ Encoding & Scaling
One‑Hot Encoding for nominal categorical variables (e.g., Gender, Region).
Ordinal Encoding for ordered categories (e.g., Education, Occupation levels).
Applied StandardScaler on the final numeric feature matrix; the fitted scaler is saved as scalar.pkl.

6️⃣ Class Imbalance Handling
The Good/Bad classes are imbalanced; applied SMOTE to oversample the minority class and create a balanced training set before model training.

7️⃣ Model Training
Trained and evaluated multiple classifiers:
K‑Nearest Neighbors (KNN)
Logistic Regression
Naive Bayes
Decision Tree
Random Forest
AdaBoost
Gradient Boosting – selected as final model
XGBoost
The best‑performing model (based on ROC‑AUC and classification metrics) is saved as:
credit_card.pkl



🚀 Deployment
Backend: Flask application (app.py) exposing prediction endpoint.
Frontend: index.html with Bootstrap form for user inputs and result display.
The app loads credit_card.pkl and scalar.pkl at startup and returns real‑time predictions (Good / Bad) for new customers.

⚙️ How to Run Locally

 1. Install dependencies
pip install -r requirements.txt

 2. Run the ML pipeline (optional if model already trained)
python main.py

 3. Start the Flask app
python app.py
Open your browser at:
http://127.0.0.1:5000
Enter customer details in the form and submit to see the Good/Bad prediction.




🧠 Technologies Used
Python, NumPy, Pandas
Scikit‑learn (preprocessing, models, evaluation)
SMOTE from imbalanced-learn for class imbalance
XGBoost, GradientBoostingClassifier
Flask for deployment
HTML / Bootstrap for UI




👤 Author
Topalle Nikhila
Data Scientist
