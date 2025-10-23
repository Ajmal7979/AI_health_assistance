🩺 Disease Prediction Using Machine Learning
✅ 1. Project Overview

This project predicts diseases based on user symptoms using Machine Learning algorithms. Three models were used:

Logistic Regression (LR)

Multinomial Naive Bayes (MNB)

Random Forest (RF)
These models are combined using a Voting Classifier (Hard Voting) to increase overall prediction accuracy and reliability.

🎯 2. Objective

To build a system that predicts the most probable disease from symptoms.

To assist users or doctors with early diagnosis.

To evaluate multiple ML models and choose the best-performing one.

To deploy this using a simple interface (Flask/Streamlit).

🧠 3. Methodology
Step-by-step Workflow:

Data Collection – Dataset with symptoms as features and disease as target.

Data Preprocessing

Handle missing values

Convert symptoms into numerical format (Label Encoding / One-Hot Encoding)

Model Training

Train LR, MNB, and RF models separately.

Model Evaluation

Accuracy, Precision, Recall, F1-score.

Voting Classifier

Combine all 3 models → Final prediction = Majority vote.

Deployment

Save model using joblib and integrate with Flask/Streamlit frontend.

🤖 4. Machine Learning Models Used
Model	Why Used?	Strengths
Logistic Regression (LR)	Good for binary/multiclass classification	Fast, interpretable
Multinomial Naive Bayes (MNB)	Works well for text-like symptom frequency data	High speed, good for high-dimensional data
Random Forest (RF)	Handles non-linear data + gives high accuracy	Reduces overfitting, works with complex data
Voting Classifier	Combines all 3 models	Improves overall reliability


📁 6. Project Structure
Disease_Prediction/
│
├── dataset.csv
├── disease_model.pkl          # Saved voting model
├── tfidf_vectorizer.pkl       # If using text symptom input
├── label_encoder.pkl
├── app.py                     # Flask/Streamlit app
├── model_training.py
├── README.md
└── requirements.txt
