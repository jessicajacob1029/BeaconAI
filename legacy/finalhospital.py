import tkinter as tk
from tkinter import messagebox, ttk
import pickle
import numpy as np
import datetime

try:
    with open("disease_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None

def check_login():
    username = user_entry.get()
    password = pass_entry.get()
    if username == "admin" and password == "1234":
        messagebox.showinfo("Login", "Login Successful")
        hospital_dashboard()
    else:
        messagebox.showerror("Login", "Invalid Credentials")

def hospital_dashboard():
    dashboard_win = tk.Toplevel(root)
    dashboard_win.title("Hospital Management Dashboard")
    dashboard_win.geometry("700x600")
    dashboard_win.configure(bg='lightgreen')

    buttons = [
        ("Book Appointment", book_appointment_page, "green"),
        ("Pharmacy System", pharmacy_system_page, "blue"),
        ("Predict Disease", predict_disease_page, "red"),
        ("Doctor Details", doctor_details_page, "purple"),
        ("Emergency Bed Allotment", bed_allotment_page, "orange")
    ]

    for text, command, color in buttons:
        tk.Button(dashboard_win, text=text, command=command, bg=color, fg="white",
                  font=("Arial", 18, "bold"), width=30, height=3).pack(pady=15)

def book_appointment_page():
    appointment_win = tk.Toplevel(root)
    appointment_win.title("Book Appointment")
    appointment_win.geometry("550x450")
    appointment_win.configure(bg='yellow')

    tk.Label(appointment_win, text="Select Doctor", font=("Arial", 18, "bold")).pack(pady=10)
    doctors = ["Dr. Smith - Cardiologist", "Dr. Brown - Neurologist", "Dr. Johnson - Pediatrician"]
    doctor_var = tk.StringVar()
    ttk.Combobox(appointment_win, textvariable=doctor_var, values=doctors, font=("Arial", 16), width=35).pack(pady=10)

    tk.Label(appointment_win, text="Select Date", font=("Arial", 18, "bold")).pack(pady=10)
    date_picker = tk.Entry(appointment_win, font=("Arial", 16), width=25)
    date_picker.insert(0, datetime.date.today().strftime('%Y-%m-%d'))
    date_picker.pack(pady=10)

    tk.Label(appointment_win, text="Select Time", font=("Arial", 18, "bold")).pack(pady=10)
    time_picker = tk.Entry(appointment_win, font=("Arial", 16), width=25)
    time_picker.pack(pady=10)

    tk.Label(appointment_win, text="Or Ask for a Random Time", font=("Arial", 16)).pack(pady=10)
    tk.Button(appointment_win, text="Random Time", command=lambda: suggest_random_time(time_picker),
              bg="purple", fg="white", font=("Arial", 18, "bold"), width=25, height=2).pack(pady=15)

    tk.Button(appointment_win, text="Book", command=lambda: messagebox.showinfo("Success", "Appointment Booked"),
              bg="green", fg="white", font=("Arial", 18, "bold"), width=25, height=2).pack(pady=15)

def suggest_random_time(time_picker):
    import random
    hours = random.randint(9, 17)  
    minutes = random.choice([0, 15, 30, 45])  
    time_picker.delete(0, tk.END)
    time_picker.insert(0, f"{hours:02d}:{minutes:02d}")

def pharmacy_system_page():
    pharmacy_win = tk.Toplevel(root)
    pharmacy_win.title("Pharmacy System")
    pharmacy_win.geometry("550x550")
    pharmacy_win.configure(bg='lightpink')

    medicines = ["Paracetamol", "Ibuprofen", "Cough Syrup", "Antibiotics", "Pain Relief Gel", "Vitamin C", "Antiseptic Cream"]
    selected_meds = []
    
    for med in medicines:
        var = tk.BooleanVar()
        tk.Checkbutton(pharmacy_win, text=med, variable=var, font=("Arial", 16)).pack(anchor="w", pady=5)
        selected_meds.append(var)

    tk.Label(pharmacy_win, text="Select Collection Date", font=("Arial", 18, "bold")).pack(pady=10)
    date_entry = tk.Entry(pharmacy_win, font=("Arial", 16), width=25)
    date_entry.pack(pady=10)

    tk.Label(pharmacy_win, text="Select Collection Time", font=("Arial", 18, "bold")).pack(pady=10)
    time_entry = tk.Entry(pharmacy_win, font=("Arial", 16), width=25)
    time_entry.pack(pady=10)

    tk.Button(pharmacy_win, text="Book Medicines", command=lambda: messagebox.showinfo("Success", "Medicines Booked"),
              bg="blue", fg="white", font=("Arial", 18, "bold"), width=25, height=2).pack(pady=15)

def predict_disease_page():
    predict_win = tk.Toplevel(root)
    predict_win.title("Disease Prediction")
    predict_win.geometry("550x600")
    predict_win.configure(bg='lavender')

    tk.Label(predict_win, text="Select Symptoms", font=("Arial", 18, "bold")).pack(pady=10)
    symptoms = ["Fever", "Cough", "Headache", "Fatigue", "Shortness of breath", "Sore throat", "Runny nose", "Chest pain", "Muscle pain", "Nausea", "Dizziness", "Loss of taste"]
    selected_symptoms = []

    for symptom in symptoms:
        var = tk.BooleanVar()
        tk.Checkbutton(predict_win, text=symptom, variable=var, font=("Arial", 16)).pack(anchor="w", pady=5)
        selected_symptoms.append(var)

    tk.Button(predict_win, text="Predict Disease", command=lambda: predict_disease(selected_symptoms),
              bg="red", fg="white", font=("Arial", 18, "bold"), width=25, height=2).pack(pady=15)

def predict_disease(selected_symptoms):
    if model is None:
        messagebox.showerror("Error", "Model not trained")
        return

    feature_names = ["Fever", "Cough", "Headache", "Fatigue", "Shortness of breath", "Sore throat", 
                     "Runny nose", "Chest pain", "Muscle pain", "Nausea", "Dizziness", "Loss of taste"]
    
    input_data = np.zeros(len(feature_names))

    for i, symptom in enumerate(feature_names):
        if symptom in selected_symptoms:
            input_data[i] = 1  

    input_data = input_data.reshape(1, -1)  
    disease = model.predict(input_data)[0]

    messagebox.showinfo("Prediction", f"Possible Disease: {disease}")

def doctor_details_page():
    doctor_win = tk.Toplevel(root)
    doctor_win.title("Doctor Details")
    doctor_win.geometry("550x450")
    doctor_win.configure(bg='lavender')

    details = """Dr. Smith - Cardiologist\nMBBS, MD (Cardiology)\nExperience: 15 years\n\nDr. Brown - Neurologist\nMBBS, DM (Neurology)\nExperience: 10 years\n\nDr. Johnson - Pediatrician\nMBBS, MD (Pediatrics)\nExperience: 8 years"""
    tk.Label(doctor_win, text=details, font=("Arial", 18), justify="left").pack(pady=20)

def bed_allotment_page():
    bed_win = tk.Toplevel(root)
    bed_win.title("Emergency Bed Allotment")
    bed_win.geometry("550x350")
    bed_win.configure(bg='lightblue')

    tk.Label(bed_win, text="Allotting bed for emergency case...", font=("Arial", 18, "bold")).pack(pady=20)
    tk.Button(bed_win, text="Allot Bed", command=lambda: messagebox.showinfo("Success", "Bed Allotted"),
              bg="orange", fg="white", font=("Arial", 18, "bold"), width=25, height=2).pack(pady=20)

# Main Login Window
root = tk.Tk()
root.configure(bg='lightblue')
root.title("Hospital Management System")
root.geometry("550x450")

tk.Label(root, text="Username", font=("Arial", 18, "bold")).pack(pady=10)
user_entry = tk.Entry(root, font=("Arial", 16), width=30)
user_entry.pack(pady=10)

tk.Label(root, text="Password", font=("Arial", 18, "bold")).pack(pady=10)
pass_entry = tk.Entry(root, show="*", font=("Arial", 16), width=30)
pass_entry.pack(pady=10)

tk.Button(root, text="Login", command=check_login, bg="black", fg="white",
          font=("Arial", 18, "bold"), width=25, height=2).pack(pady=20)

root.mainloop()
