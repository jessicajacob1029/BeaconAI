import tkinter as tk
from tkinter import messagebox

# Rule-based Disease Prediction System
def predict_disease(symptoms):
    if "Chest pain" in symptoms and "Fever" in symptoms:
        return "Heart Disease"
    elif "Cough" in symptoms and "Fever" in symptoms:
        return "Flu"
    elif "Fatigue" in symptoms and "Cough" in symptoms:
        return "Pneumonia"
    elif "Dizziness" in symptoms and "Fatigue" in symptoms:
        return "Migraine"
    else:
        return "Unknown"

# Treatment Suggestions based on Disease
def suggest_treatment(disease):
    treatments = {
        "Heart Disease": "Treatment: Surgery, Lifestyle Changes, Medication",
        "Flu": "Treatment: Antiviral Drugs, Hydration, Rest",
        "Pneumonia": "Treatment: Antibiotics, Rest, Fluids",
        "Migraine": "Treatment: Pain Relievers, Avoid Triggers",
        "Unknown": "Treatment: Consult a doctor for diagnosis"
    }
    return treatments.get(disease, "Consult a doctor for treatment options")

# Severity Prediction based on Symptoms
def predict_severity(symptoms):
    severity = "Mild"
    if "Chest pain" in symptoms or "Dizziness" in symptoms:
        severity = "Severe"
    elif "Cough" in symptoms or "Fatigue" in symptoms:
        severity = "Moderate"
    return severity

# Doctor Recommendation based on Disease and Severity
def recommend_doctor(disease, severity):
    doctors = {
        "Heart Disease": "Cardiologist",
        "Flu": "General Physician",
        "Pneumonia": "Pulmonologist",
        "Migraine": "Neurologist",
        "Unknown": "General Physician"
    }
    doctor = doctors.get(disease, "General Physician")
    if severity == "Severe":
        doctor += " (Emergency)"
    return doctor

# AI-based Appointment Scheduling
def schedule_appointment(doctor):
    available_times = {
        "Cardiologist": ["9:00 AM", "11:00 AM", "2:00 PM"],
        "General Physician": ["10:00 AM", "12:00 PM", "3:00 PM"],
        "Pulmonologist": ["9:30 AM", "1:00 PM", "4:00 PM"],
        "Neurologist": ["8:00 AM", "10:30 AM", "3:30 PM"],
    }
    available_slots = available_times.get(doctor, ["9:00 AM", "2:00 PM"])
    return available_slots

# Function to handle the result and show in UI
def show_prediction_result():
    symptoms = []
    if chest_pain_var.get():
        symptoms.append("Chest pain")
    if cough_var.get():
        symptoms.append("Cough")
    if fatigue_var.get():
        symptoms.append("Fatigue")
    if dizziness_var.get():
        symptoms.append("Dizziness")
    
    # Predict the disease
    disease = predict_disease(symptoms)
    
    # Predict severity
    severity = predict_severity(symptoms)
    
    # Recommend a doctor based on disease and severity
    doctor = recommend_doctor(disease, severity)
    
    # Suggest treatment based on disease
    treatment = suggest_treatment(disease)
    
    # Show results in a message box
    messagebox.showinfo("Prediction Result", f"Disease: {disease}\nSeverity: {severity}\nDoctor: {doctor}\nTreatment: {treatment}")

    # Schedule appointment based on doctor recommendation
    available_slots = schedule_appointment(doctor)
    appointment_time = tk.simpledialog.askstring("Schedule Appointment", f"Available slots for {doctor}: {', '.join(available_slots)}\nSelect a time: ")
    if appointment_time in available_slots:
        messagebox.showinfo("Appointment Scheduled", f"Your appointment with {doctor} has been scheduled at {appointment_time}.")
    else:
        messagebox.showerror("Invalid Slot", "Please choose a valid time slot.")

# Function for the login page
def check_login():
    username = username_entry.get()
    password = password_entry.get()
    if username == "admin" and password == "admin123":
        login_window.destroy()
        hospital_dashboard()
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

# Hospital Dashboard
def hospital_dashboard():
    dashboard_window = tk.Tk()
    dashboard_window.title("Hospital Management System")

    dashboard_window.geometry("400x400")
    dashboard_window.configure(bg="lightblue")

    # Create buttons for disease prediction, doctor recommendation, and appointment scheduling
    disease_prediction_button = tk.Button(dashboard_window, text="Predict Disease", command=show_prediction_result, width=25, height=2, bg="lightgreen")
    disease_prediction_button.pack(pady=20)

    doctor_recommendation_button = tk.Button(dashboard_window, text="Doctor Recommendation", command=show_prediction_result, width=25, height=2, bg="lightgreen")
    doctor_recommendation_button.pack(pady=20)

    appointment_button = tk.Button(dashboard_window, text="Schedule Appointment", command=show_prediction_result, width=25, height=2, bg="lightgreen")
    appointment_button.pack(pady=20)
    
    dashboard_window.mainloop()

# Function to create the login page
def create_login_page():
    global login_window, username_entry, password_entry, chest_pain_var, cough_var, fatigue_var, dizziness_var
    login_window = tk.Tk()
    login_window.title("Hospital Login")

    login_window.geometry("400x400")
    login_window.configure(bg="lightblue")

    # Username label and entry
    username_label = tk.Label(login_window, text="Username:", bg="lightblue")
    username_label.pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    # Password label and entry
    password_label = tk.Label(login_window, text="Password:", bg="lightblue")
    password_label.pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    # Login button
    login_button = tk.Button(login_window, text="Login", command=check_login, width=20, height=2, bg="lightgreen")
    login_button.pack(pady=20)

    # Create the symptoms checkboxes for disease prediction
    chest_pain_var = tk.BooleanVar()
    cough_var = tk.BooleanVar()
    fatigue_var = tk.BooleanVar()
    dizziness_var = tk.BooleanVar()

    # Symptoms checkboxes
    chest_pain_check = tk.Checkbutton(login_window, text="Chest pain", variable=chest_pain_var, bg="lightblue")
    chest_pain_check.pack(pady=5)
    cough_check = tk.Checkbutton(login_window, text="Cough", variable=cough_var, bg="lightblue")
    cough_check.pack(pady=5)
    fatigue_check = tk.Checkbutton(login_window, text="Fatigue", variable=fatigue_var, bg="lightblue")
    fatigue_check.pack(pady=5)
    dizziness_check = tk.Checkbutton(login_window, text="Dizziness", variable=dizziness_var, bg="lightblue")
    dizziness_check.pack(pady=5)

    login_window.mainloop()

# Start the login page
create_login_page()


