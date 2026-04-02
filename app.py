import streamlit as st
import pickle
import re

# --------------------------
# Load model and vectorizer
# --------------------------
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

# --------------------------
# Helper function
# --------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# --------------------------
# Page setup
# --------------------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="✉️",
    layout="centered",
)

# --------------------------
# Custom CSS
# --------------------------
st.markdown("""
<style>
body {
    background-color: #f2f2f7;
    color: #111;
    font-family: 'Helvetica', sans-serif;
}
.main-card {
    background-color: #ffffff;
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    max-width: 700px;
    margin: auto;
}
h1, h2, h3 {
    font-weight: 600;
}
button {
    background: linear-gradient(90deg,#0072ff,#00c6ff);
    color: white;
    border-radius: 10px;
    padding: 10px 25px;
    font-weight: 600;
}
textarea {
    border-radius: 12px !important;
    padding: 15px !important;
}
.result {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-weight: 600;
    margin-top: 20px;
}
.spam {
    background-color: #ffe5e5;
    color: #c00000;
}
.notspam {
    background-color: #e5ffe5;
    color: #006400;
}
.sample-btn {
    background-color: #ddd;
    color: #111;
    border-radius: 8px;
    padding: 8px 15px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Main card container
# --------------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;'>✉️ Spam Email Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Enter any message below to detect spam using ML</p>", unsafe_allow_html=True)

# --------------------------
# Input area
# --------------------------
user_input = st.text_area("Enter your message here:", height=120)

# --------------------------
# Sample messages
# --------------------------
st.markdown("<h4>Sample messages:</h4>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Prize Alert", key="sample1"):
        user_input = "Congratulations! You won a prize, click now"
with col2:
    if st.button("Money Transfer", key="sample2"):
        user_input = "You received a payment click here to accept"
with col3:
    if st.button("Class Reminder", key="sample3"):
        user_input = "Are you coming to class today?"
with col4:
    if st.button("Free Gift", key="sample4"):
        user_input = "Claim your free gift card now click here"

# --------------------------
# Predict button
# --------------------------
if st.button("Predict"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prob = model.predict_proba(vector)[:,1][0]

        if prob >= 0.3:
            st.markdown(f"<div class='result spam'>❌ Spam Detected<br>Probability: {round(prob*100,2)}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result notspam'>✅ Not Spam<br>Probability: {round(prob*100,2)}%</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a message to predict.")

# --------------------------
# Footer
# --------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888; font-size:12px;'>Developed with Python, Streamlit & ML | Spam Email Detector</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)