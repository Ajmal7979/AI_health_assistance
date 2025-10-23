import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
from fpdf import FPDF
import difflib

# -----------------------------------------------------------
# 🧠 Page Setup
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown(
    "<h1 style='text-align: center; color: #1E90FF;'>💬 AI Health Assistant</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size:18px;'>Enter your symptoms to get a preliminary health assessment</p>",
    unsafe_allow_html=True
)

# -----------------------------------------------------------
# 🧩 Load Model Components
# -----------------------------------------------------------
voting_model = joblib.load("disease_modelv3.pkl")
vectorizer = joblib.load("tfidf_vectorizerv3.pkl")
encoder = joblib.load("label_encoderv3.pkl")

# -----------------------------------------------------------
# 🩹 Load Precautions Dataset
# -----------------------------------------------------------
precaution_df = pd.read_csv("Disease precaution.csv")
precaution_df.columns = [c.strip().lower().replace(" ", "_") for c in precaution_df.columns]

# Detect disease column
if "disease" in precaution_df.columns:
    disease_col = "disease"
elif "disease_name" in precaution_df.columns:
    disease_col = "disease_name"
else:
    st.error("❌ No valid 'disease' column found.")
    st.stop()

# -----------------------------------------------------------
# 🏥 Department Mapping
# -----------------------------------------------------------
department_map = {
    'drug reaction': 'General Physician',
    'malaria': 'Infectious Disease Specialist',
    'allergy': 'Dermatologist',
    'hypothyroidism': 'Endocrinologist',
    'psoriasis': 'Dermatologist',
    'gerd': 'Gastroenterologist',
    'chronic cholestasis': 'Gastroenterologist',
    'hepatitis a': 'Gastroenterologist',
    'osteoarthristis': 'Orthopedic',
    '(vertigo) paroymsal  positional vertigo': 'Neurologist',
    'hypoglycemia': 'Endocrinologist',
    'acne': 'Dermatologist',
    'diabetes': 'Endocrinologist',
    'impetigo': 'Dermatologist',
    'hypertension': 'Cardiologist',
    'peptic ulcer diseae': 'Gastroenterologist',
    'dimorphic hemmorhoids(piles)': 'Gastroenterologist',
    'common cold': 'General Physician',
    'chicken pox': 'Dermatologist',
    'cervical spondylosis': 'Orthopedic',
    'hyperthyroidism': 'Endocrinologist',
    'urinary tract infection': 'Urologist',
    'aids': 'Infectious Disease Specialist',
    'paralysis (brain hemorrhage)': 'Neurologist',
    'typhoid': 'General Physician',
    'hepatitis b': 'Gastroenterologist',
    'fungal infection': 'Dermatologist',
    'hepatitis c': 'Gastroenterologist',
    'migraine': 'Neurologist',
    'bronchial asthma': 'Pulmonologist',
    'alcoholic hepatitis': 'Gastroenterologist',
    'jaundice': 'Gastroenterologist',
    'hepatitis e': 'Gastroenterologist',
    'dengue': 'General Physician',
    'hepatitis d': 'Gastroenterologist',
    'heart attack': 'Cardiologist',
    'pneumonia': 'Pulmonologist',
    'arthritis': 'Orthopedic',
    'gastroenteritis': 'Gastroenterologist',
    'tuberculosis': 'Pulmonologist'
}

# -----------------------------------------------------------
# 🧠 Helper Functions
# -----------------------------------------------------------
def predict_disease(symptoms_input):
    cleaned = re.sub(r'[^a-zA-Z, ]', '', symptoms_input.lower()).strip()
    X_input = vectorizer.transform([cleaned])
    pred_encoded = voting_model.predict(X_input)[0]
    return encoder.inverse_transform([pred_encoded])[0]

def get_precautions(disease_name):
    row = precaution_df[precaution_df[disease_col].str.lower() == disease_name.lower()]
    if not row.empty:
        return [row.iloc[0][col] for col in precaution_df.columns if "precaution" in col.lower() and pd.notna(row.iloc[0][col])]
    return ["No specific precautions available."]

def get_severity_message(days):
    if days <= 2:
        return "🟢 Mild — likely temporary. Monitor and rest."
    elif 3 <= days <= 5:
        return "🟠 Moderate — consult a doctor if symptoms persist."
    else:
        return "🔴 Severe — seek medical attention immediately."

# ---------- New: symptom matching + safe prediction ----------
def _get_vectorizer_features():
    try:
        return set(vectorizer.get_feature_names_out())
    except Exception:
        return set(vectorizer.vocabulary_.keys())

def match_known_symptoms(symptoms_input, cutoff_multi=0.7, cutoff_word=0.85):
    """
    Return a list of matched tokens from the vectorizer vocabulary.
    Uses exact matching and difflib.get_close_matches as fallback.
    """
    features = _get_vectorizer_features()
    # split on commas and semicolons, also keep space-separated phrases
    tokens = [t.strip().lower() for t in re.split(r'[,\n;]+', symptoms_input) if t.strip()]
    matched = []
    for tok in tokens:
        if tok in features:
            matched.append(tok)
            continue
        # try close multi-word matches
        close = difflib.get_close_matches(tok, features, n=1, cutoff=cutoff_multi)
        if close:
            matched.append(close[0])
            continue
        # fallback: try matching component words
        for w in tok.split():
            if w in features:
                matched.append(w)
            else:
                c = difflib.get_close_matches(w, features, n=1, cutoff=cutoff_word)
                if c:
                    matched.append(c[0])
    # preserve order, remove duplicates
    seen = set()
    result = []
    for m in matched:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result

def predict_with_matches(symptoms_input):
    """
    Use matched keywords to build a cleaned input for prediction.
    If the model supports predict_proba, return top-3 suggestions with probabilities.
    """
    matches = match_known_symptoms(symptoms_input)
    if not matches:
        return None, []  # signal "no match"
    cleaned = ", ".join(matches)
    # primary prediction
    primary = predict_disease(cleaned)
    suggestions = []
    if hasattr(voting_model, "predict_proba"):
        X_input = vectorizer.transform([re.sub(r'[^a-zA-Z, ]', '', cleaned.lower()).strip()])
        probs = voting_model.predict_proba(X_input)[0]
        top_idx = np.argsort(probs)[::-1][:3]
        top_diseases = encoder.inverse_transform(top_idx)
        suggestions = list(zip([d.title() for d in top_diseases], probs[top_idx]))
    return primary, suggestions

# ------------------------ 🧾 Generate PDF Report ------------------------
def generate_report(disease, dept, severity, precautions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="AI Health Assistant Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Disease Predicted: {disease.title()}")
    pdf.multi_cell(0, 10, txt=f"Recommended Department: {dept}")
    pdf.multi_cell(0, 10, txt=f"Severity: {severity}")
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Recommended Precautions:", ln=True)
    pdf.set_font("Arial", size=12)
    for p in precautions:
        pdf.multi_cell(0, 8, f"- {p}")

    file_name = f"Health_Report_{disease.replace(' ', '_')}.pdf"
    pdf.output(file_name)
    return file_name

# -----------------------------------------------------------
# 💬 Streamlit Chatbot UI
# -----------------------------------------------------------
chat_container = st.container()

with chat_container:
    symptoms_input = st.text_area(
        "🤒 Describe your symptoms (comma-separated)",
        height=100,
        placeholder="e.g., fever, cough, headache"
    )

    if symptoms_input.strip() != "":
        days = st.number_input(
            "📅 How many days have you experienced these symptoms?",
            min_value=1,
            max_value=30,
            value=1
        )

        if st.button("🔍 Analyze Symptoms"):
            with st.spinner("Analyzing your symptoms..."):
                primary, suggestions = predict_with_matches(symptoms_input)

                if primary is None:
                    # No recognizable symptoms found
                    st.error("⚠️ No recognizable symptom keywords found in your input.")
                    st.info("Try describing symptoms (e.g., 'fever, cough, headache') or consult a doctor if unsure.")
                    google_help = "https://www.google.com/search?q=when+to+see+a+doctor"
                    st.markdown(f"[📘 Learn when to see a doctor]({google_help})")
                else:
                    precautions = get_precautions(primary)
                    dept = department_map.get(primary.lower(), "General Physician")
                    severity_message = get_severity_message(days)

                    # 🩺 Display Results
                    st.markdown("### 🩺 Prediction Result")
                    st.info(f"**Predicted Disease:** {primary.title()}")
                    st.success(f"**Department to Visit:** {dept}")
                    st.warning(f"**Severity Level:** {severity_message}")

                    if suggestions:
                        st.markdown("#### 🔎 Top suggestions (with confidence)")
                        for d, p in suggestions:
                            st.write(f"- {d}: {round(float(p) * 100, 2)}%")

                    st.markdown("### 💊 Recommended Precautions")
                    for i, p in enumerate(precautions, 1):
                        st.write(f"{i}. {p}")

                    # Doctor finder
                    st.markdown("---")
                    st.markdown("### 🩻 Find Nearby Specialists")
                    st.write(f"Click below to find **{dept}** near you 👇")
                    google_search_url = f"https://www.google.com/maps/search/{dept.replace(' ', '+')}+near+me"
                    st.markdown(f"[🔗 Find Nearby Doctors]({google_search_url})")
                    st.markdown("---")
                    st.markdown(
                        "<p style='font-size:18px; text-align:center; color:green;'>"
                        "💡 Remember: Early detection saves lives! Stay positive and take care of your health. 🌟"
                        "</p>",
                        unsafe_allow_html=True
                    )
                    st.caption("⚠️ This is a preliminary AI-based prediction. Always consult a certified doctor.")
