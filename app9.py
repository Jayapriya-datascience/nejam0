import streamlit as st
import pickle
import numpy as np
import os
import joblib

# CSS for Multi-Color Blinking Effect
multi_color_blink_css = """
<style>
@keyframes colorBlink {
    0% {color: red;}
    25% {color: green;}
    50% {color: blue;}
    75% {color: orange;}
}
.color-blink {
    animation: colorBlink 1s infinite;
    text-align: center;
    font-weight: bold;
}
</style>
"""
st.markdown(multi_color_blink_css, unsafe_allow_html=True)

# Load Model and Scaler
model_path = os.path.join("model9", "trained_model9.pkl")
scaler_path = os.path.join("model9", "scaler9.pkl")

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found! Expected at: {model_path}")
    st.stop()
if not os.path.exists(scaler_path):
    st.error(f"❌ Scaler file not found! Expected at: {scaler_path}")
    st.stop()

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

try:
    scaler = joblib.load(scaler_path)
    if not hasattr(scaler, "transform"):
        raise TypeError("Loaded scaler is not valid.")
except Exception as e:
    st.error(f"❌ Error loading scaler: {e}")
    st.stop()

# App Title
st.markdown("""
<div style="background-color:#e6f2ff; padding:10px; border-radius:10px;">
<h2 class="color-blink">💤 Sleep Disorder Prediction App</h2>
</div>
""", unsafe_allow_html=True)

st.write("🌙 Enter your details to check for sleep disorder risk.")

# Input Fields
st.title("Personal Information")
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
gender = {"Male": 0, "Female": 1, "Other": 2}[gender]

occupation = st.selectbox("Occupation", ["Nurse", "Doctor", "Engineer", "Lawyer", "Teacher", "Accountant", "Salesperson", "Student", "Others"])
occupation = {"Nurse": 0, "Doctor": 1, "Engineer": 2, "Lawyer": 3, "Teacher": 4, "Accountant": 5, "Salesperson": 6, "Student": 7, "Others": 8}[occupation]

height = st.number_input("Height (cm)", 100, 250, 170)
weight = st.number_input("Weight (kg)", 30, 200, 70)

if height > 0 and weight > 0:
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        bmi_category = 0
    elif 18.5 <= bmi < 24.9:
        bmi_category = 1
    elif 25 <= bmi < 29.9:
        bmi_category = 2
    else:
        bmi_category = 3
    st.text_input("BMI Category", value=["Underweight", "Normal", "Overweight", "Obese"][bmi_category], disabled=True)

# Sleep Details
st.title("Sleep Details")
sleep_duration = st.slider("Sleep Duration (hours)", 1.0, 12.0, 7.0)
quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 5)
st.session_state.sleep_details = {'quality_of_sleep': quality_of_sleep}

physical_activity = st.number_input("Physical Activity Level", 0, 100, value=30, step=10)
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
heart_rate = st.number_input("Heart Rate (bpm)", 40, 120, 70)
daily_steps = st.number_input("Daily Steps (0-10000)", 0, 10000, 5000, step=100)
systolic = st.number_input("Systolic Blood Pressure", 80, 200, 120)
diastolic = st.number_input("Diastolic Blood Pressure", 50, 130, 80)

# Prepare input
input_features = np.array([[age, gender, occupation, sleep_duration, quality_of_sleep, physical_activity,
                            stress_level, bmi_category, heart_rate, daily_steps, systolic, diastolic]])

try:
    input_features_scaled = scaler.transform(input_features)
except Exception as e:
    st.error(f"Error scaling input: {e}")
    st.stop()

# Prediction
disorder_info = {
    "Insomnia": ("Difficulty falling or staying asleep.", "Reduce caffeine, maintain a sleep schedule, try relaxation techniques."),
    "Sleep Anxiety": ("Anxiety-related sleep disturbances.", "Practice meditation, avoid screens before bed."),
    "Obstructive Sleep Apnea": ("Airway blockage during sleep.", "Lose weight, avoid alcohol, consider CPAP."),
    "Hypertension-related Sleep Issues": ("Linked to high blood pressure.", "Monitor BP, reduce salt."),
    "Restless Leg Syndrome": ("Uncontrollable urge to move legs.", "Exercise, avoid caffeine."),
    "Narcolepsy": ("Sudden sleep attacks.", "Maintain sleep schedule."),
    "General Sleep Disorder": ("Mild sleep disturbances.", "Improve sleep hygiene.")
}

if st.button("🔍 Predict"):
    prediction = model.predict(input_features_scaled)
    st.session_state.prediction = prediction
    possible_disorders = []

    if prediction[0] == 1:
        st.error("⚠️ High risk of sleep disorder detected!")

        if sleep_duration < 5 or quality_of_sleep < 3:
            possible_disorders.append("Insomnia")
        if stress_level > 5:
            possible_disorders.append("Sleep Anxiety")
        if bmi_category == 3 and heart_rate > 90:
            possible_disorders.append("Obstructive Sleep Apnea")
        if systolic > 140 or diastolic > 90:
            possible_disorders.append("Hypertension-related Sleep Issues")
        if daily_steps < 3000 and physical_activity < 20:
            possible_disorders.append("Restless Leg Syndrome")
        if sleep_duration > 9 or stress_level > 3:
            possible_disorders.append("Narcolepsy")
        if not possible_disorders:
            possible_disorders.append("General Sleep Disorder")

        st.session_state.possible_disorders = possible_disorders
        st.warning(f"🛏️ **Possible Sleep Disorders:** {', '.join(possible_disorders)}")

        for disorder in possible_disorders:
            st.subheader(f"🩺 {disorder}")
            st.write(f"🔹 **Definition:** {disorder_info[disorder][0]}")
            st.write(f"💡 **Tips:** {disorder_info[disorder][1]}")

    else:
        st.success("✅ Low risk of sleep disorder detected. Keep up the good habits!")
        st.session_state.possible_disorders = []
        st.subheader("🛌 Tips for Healthy Sleep")
        st.write("1. Maintain a consistent sleep schedule.")
        st.write("2. Create a relaxing bedtime routine.")
        st.write("3. Exercise regularly.")
        st.write("4. Keep your room cool, dark, and quiet.")
        st.write("5. Manage stress before sleep.")

# PDF Download
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def generate_pdf(disorders):
    path = "Sleep_Disorder_Report.pdf"
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 50, "Sleep Disorder Report")
    y = height - 100
    c.setFont("Helvetica", 12)
    c.drawString(50, y, "Detected Disorders and Suggestions:")
    y -= 30
    for d in disorders:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, d)
        y -= 20
        c.setFont("Helvetica", 12)
        c.drawString(50, y, f"Definition: {disorder_info[d][0]}")
        y -= 20
        c.drawString(50, y, f"Tips: {disorder_info[d][1]}")
        y -= 40
    c.save()
    return path

if "possible_disorders" in st.session_state and st.session_state.possible_disorders:
    pdf_file = generate_pdf(st.session_state.possible_disorders)
    with open(pdf_file, "rb") as file:
        st.download_button("📄 Download Report", data=file, file_name="Sleep_Disorder_Report.pdf", mime="application/pdf")
