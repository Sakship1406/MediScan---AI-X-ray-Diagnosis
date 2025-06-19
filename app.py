import streamlit as st
import json
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from fpdf import FPDF
import base64

# -------------------------------
# App Configuration & Theme
# -------------------------------
st.set_page_config(page_title="MediScan - AI X-ray Diagnosis", layout="centered", page_icon="🩺")

LOGO_URL = "https://cdn-icons-png.flaticon.com/512/4149/4149653.png"

APP_NAME = "MediScan AI - Pneumonia Detection"

# -------------------------------
# User Data Management
# -------------------------------
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# -------------------------------
# Login Page
# -------------------------------
def login_page():
    st.image(LOGO_URL, width=80)
    st.markdown(f"<h2 style='color:#2c3e50;'>{APP_NAME}</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#4a4a4a;'>🔐 Login</h3>", unsafe_allow_html=True)
    users = load_users()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.user = username
            st.session_state.page = "dashboard"
            st.success("Logged in successfully ✅")
            st.rerun()
        else:
            st.error("Invalid credentials ❌")

    if st.button("Don't have an account? Sign up"):
        st.session_state.page = "signup"
        st.rerun()

# -------------------------------
# Signup Page
# -------------------------------
def signup_page():
    st.image(LOGO_URL, width=80)
    st.markdown(f"<h2 style='color:#2c3e50;'>{APP_NAME}</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#4a4a4a;'>📝 Sign Up</h3>", unsafe_allow_html=True)

    users = load_users()

    new_username = st.text_input("Choose a Username")
    new_password = st.text_input("Choose a Password", type="password")

    if st.button("Create Account"):
        if new_username in users:
            st.warning("Username already exists ⚠️")
        elif not new_username or not new_password:
            st.warning("Please fill all fields ⚠️")
        else:
            users[new_username] = new_password
            save_users(users)
            st.success("Signup successful 🎉 Please login.")
            st.session_state.page = "login"
            st.rerun()

# -------------------------------
# Model Loader
# -------------------------------
@st.cache_resource
def load_model_cached():
    return load_model("cnn_model.h5")

# -------------------------------
# Diagnosis Dashboard
# -------------------------------
def diagnosis_dashboard(user):
    model = load_model_cached()

    st.sidebar.image(LOGO_URL, width=60)
    st.sidebar.markdown(f"**👤 Welcome:** `{user}`")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.page = "login"
        st.session_state.user = None
        st.rerun()

    st.markdown(f"<h2 style='color:#2c3e50;'>🩺 {APP_NAME} Dashboard</h2>", unsafe_allow_html=True)
    st.write("Upload a chest X-ray image to begin diagnosis.")

    uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.markdown("### 🖼️ Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="X-ray Image", use_container_width=True)

        img = image.convert('L').resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 150, 150, 1)

        prediction = model.predict(img_array)[0][0]
        label = "Pneumonia" if prediction > 0.5 else "Normal"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.success(f"🧪 Prediction: **{label}**")
        st.info(f"📊 Confidence: **{confidence:.2f}**")

        if st.button("📥 Download PDF Report"):
            report_name = f"diagnosis_report_{user}.pdf"
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="Medical Diagnosis Report", ln=True, align='C')
            pdf.cell(200, 10, txt=f"User: {user}", ln=True)
            pdf.cell(200, 10, txt=f"Prediction: {label}", ln=True)
            pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}", ln=True)
            pdf.output(report_name)

            with open(report_name, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="{report_name}">📄 Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)

# -------------------------------
# App Controller
# -------------------------------
def main():
    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "user" not in st.session_state:
        st.session_state.user = None

    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "signup":
        signup_page()
    elif st.session_state.page == "dashboard":
        if st.session_state.user:
            diagnosis_dashboard(st.session_state.user)
        else:
            st.warning("Session expired. Please log in again.")
            st.session_state.page = "login"
            st.rerun()

if __name__ == "__main__":
    main()
