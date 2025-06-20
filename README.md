# MediScan AI X-ray Diagnosis

**MediScan AI** is a medical image diagnosis web application built with **Streamlit** and **TensorFlow**. It allows users to upload chest X-ray images and get predictions powered by a deep learning model for **binary classification** (e.g., Pneumonia vs Normal).

## ✅ Features

- 🩻 Upload and classify chest X-ray images
- 🤖 Pre-trained CNN model for **2-class prediction**
- 🔐 Basic user login system using JSON
- 📊 Clean and interactive Streamlit interface
- ☁️ Ready for deployment on Streamlit Cloud

## 📁 Project Structure

├── .streamlit/ # Streamlit configuration
├── app.py # Main application script
├── cnn_model.h5 # Trained 2-class CNN model
├── requirements.txt # Project dependencies
├── runtime.txt # Python version for Streamlit Cloud
├── users.json # Simple user login credentials
└── .gitignore # Git ignore rules


## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sakship1406/mediscan-ai.git
   cd mediscan-ai

2. **Install the required packages:**
   pip install -r requirements.txt

3  **Start the Streamlit app:**
   streamlit run app.py



