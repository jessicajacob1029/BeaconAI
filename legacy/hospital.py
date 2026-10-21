import tkinter as tk
from tkinter import messagebox
import datetime

# Login page
def show_login_page():
    def login():
        username = username_entry.get()
        password = password_entry.get()

        if username == "admin" and password == "1234":
            messagebox.showinfo("Login", "Logged in successfully!")
            show_main_page()
        else:
            messagebox.showerror("Login", "Invalid credentials. Try again.")

    login_window = tk.Toplevel(root)
    login_window.title("Login")
    login_window.geometry("300x300")
    login_window.config(bg="lightblue")

    tk.Label(login_window, text="Username", font=("Arial", 14), bg="lightblue").pack(pady=10)
    username_entry = tk.Entry(login_window, font=("Arial", 14))
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password", font=("Arial", 14), bg="lightblue").pack(pady=10)
    password_entry = tk.Entry(login_window, font=("Arial", 14), show="*")
    password_entry.pack(pady=5)

    login_button = tk.Button(login_window, text="Login", font=("Arial", 14), bg="orange", fg="white", command=login)
    login_button.pack(pady=20, ipadx=20, ipady=5)

# Main page
def show_main_page():
    main_window = tk.Toplevel(root)
    main_window.title("Main Page")
    main_window.geometry("400x400")
    main_window.config(bg="lightgreen")

    def open_book_appointment():
        book_appointment_page()

    def open_predict_disease():
        predict_disease_page()

    def open_pharmacy_system():
        pharmacy_system_page()

    def open_doctor_details():
        doctor_details_page()

    # Buttons for various options
    book_button = tk.Button(main_window, text="Book Appointment", font=("Arial", 14), bg="orange", fg="white", command=open_book_appointment)
    book_button.pack(pady=20)

    predict_button = tk.Button(main_window, text="Predict Disease", font=("Arial", 14), bg="orange", fg="white", command=open_predict_disease)
    predict_button.pack(pady=20)

    pharmacy_button = tk.Button(main_window, text="Pharmacy System", font=("Arial", 14), bg="orange", fg="white", command=open_pharmacy_system)
    pharmacy_button.pack(pady=20)

    doctor_button = tk.Button(main_window, text="Doctor Details", font=("Arial", 14), bg="orange", fg="white", command=open_doctor_details)
    doctor_button.pack(pady=20)

# Appointment page
def book_appointment_page():
    book_window = tk.Toplevel(root)
    book_window.title("Book Appointment")
    book_window.geometry("400x400")
    book_window.config(bg="lightyellow")

    tk.Label(book_window, text="Select Doctor", font=("Arial", 14), bg="lightyellow").pack(pady=10)
    
    doctors = ["Dr. John", "Dr. Mary", "Dr. Paul", "Dr. Sarah"]
    doctor_var = tk.StringVar()
    doctor_var.set(doctors[0])

    doctor_menu = tk.OptionMenu(book_window, doctor_var, *doctors)
    doctor_menu.pack(pady=10)

    tk.Label(book_window, text="Select Date", font=("Arial", 14), bg="lightyellow").pack(pady=10)
    date_picker = tk.Entry(book_window, font=("Arial", 14))
    date_picker.insert(0, datetime.date.today().strftime('%Y-%m-%d'))
    date_picker.pack(pady=10)

    tk.Label(book_window, text="Select Time", font=("Arial", 14), bg="lightyellow").pack(pady=10)
    time_picker = tk.Entry(book_window, font=("Arial", 14))
    time_picker.pack(pady=10)

    def book_appointment():
        doctor = doctor_var.get()
        date = date_picker.get()
        time = time_picker.get()
        messagebox.showinfo("Appointment Booked", f"Your appointment with {doctor} is booked on {date} at {time}.")

    book_button = tk.Button(book_window, text="Book Appointment", font=("Arial", 14), bg="orange", fg="white", command=book_appointment)
    book_button.pack(pady=20)

# Disease prediction page
def predict_disease_page():
    predict_window = tk.Toplevel(root)
    predict_window.title("Disease Prediction")
    predict_window.geometry("600x400")
    predict_window.config(bg="lightpink")

    symptoms = [
        "Fever", "Cough", "Fatigue", "Headache", "Body Aches", "Loss of Smell/Taste", "Difficulty Breathing"
    ]

    selected_symptoms = []

    def predict_disease():
        selected = [symptom for symptom, var in selected_symptoms if var.get()]
        
        if "Fever" in selected and "Cough" in selected and "Fatigue" in selected:
            disease = "Common Cold / Flu"
        elif "Fever" in selected and "Body Aches" in selected and "Loss of Smell/Taste" in selected:
            disease = "COVID-19"
        elif "Headache" in selected and "Fatigue" in selected:
            disease = "Migraine"
        elif "Cough" in selected and "Difficulty Breathing" in selected:
            disease = "Asthma"
        else:
            disease = "Not enough symptoms selected for prediction"

        if selected:
            messagebox.showinfo("Prediction Result", f"Possible disease based on symptoms: {disease}")
        else:
            messagebox.showerror("Error", "Please select at least one symptom.")

    for symptom in symptoms:
        var = tk.BooleanVar()
        selected_symptoms.append((symptom, var))
        symptom_cb = tk.Checkbutton(predict_window, text=symptom, variable=var, font=("Arial", 12), bg="lightpink")
        symptom_cb.pack(anchor="w", padx=10, pady=5)

    predict_button = tk.Button(predict_window, text="Predict Disease", font=("Arial", 14), bg="orange", fg="white", command=predict_disease)
    predict_button.pack(pady=20)

# Pharmacy system page
def pharmacy_system_page():
    pharmacy_window = tk.Toplevel(root)
    pharmacy_window.title("Pharmacy System")
    pharmacy_window.geometry("600x400")
    pharmacy_window.config(bg="lightblue")

    medicines = ["Paracetamol", "Ibuprofen", "Amoxicillin", "Aspirin", "Cough Syrup", "Vitamin C", "Antiseptic Cream"]
    selected_medicines = []

    def book_medicines():
        selected = [med for med, var in selected_medicines if var.get()]
        if selected:
            messagebox.showinfo("Booking Successful", f"You have booked: {', '.join(selected)}")
        else:
            messagebox.showerror("Error", "Please select at least one medicine.")

    for i, med in enumerate(medicines):
        var = tk.BooleanVar()
        selected_medicines.append((med, var))
        med_cb = tk.Checkbutton(pharmacy_window, text=med, variable=var, font=("Arial", 12), bg="lightblue")
        med_cb.grid(row=i//2, column=i%2, padx=10, pady=5, sticky="w")

    tk.Label(pharmacy_window, text="Select Collection Date and Time", font=("Arial", 14), bg="lightblue").grid(row=4, column=0, columnspan=2, pady=10)

    collection_date = tk.Entry(pharmacy_window, font=("Arial", 14))
    collection_date.insert(0, datetime.date.today().strftime('%Y-%m-%d'))
    collection_date.grid(row=5, column=0, pady=5)

    collection_time = tk.Entry(pharmacy_window, font=("Arial", 14))
    collection_time.grid(row=5, column=1, pady=5)

    book_button = tk.Button(pharmacy_window, text="Book Selected Medicines", font=("Arial", 14), bg="orange", fg="white", command=book_medicines)
    book_button.grid(row=6, column=0, columnspan=2, pady=20)

# Doctor details page
def doctor_details_page():
    doctor_window = tk.Toplevel(root)
    doctor_window.title("Doctor Details")
    doctor_window.geometry("600x400")
    doctor_window.config(bg="lightyellow")

    doctor_info = {
        "Dr. John": "Specialization: Cardiology\nExperience: 10 years\nEducation: MBBS, Harvard University\nContact: 123-456-7890",
        "Dr. Mary": "Specialization: Neurology\nExperience: 8 years\nEducation: MD, Stanford University\nContact: 234-567-8901",
        "Dr. Paul": "Specialization: Orthopedics\nExperience: 15 years\nEducation: MD in Orthopedics, Johns Hopkins University\nContact: 345-678-9012",
        "Dr. Sarah": "Specialization: Pediatrics\nExperience: 5 years\nEducation: MD, University of California\nContact: 456-789-0123"
    }

    doctor_name = tk.Label(doctor_window, text="Select Doctor", font=("Arial", 14), bg="lightyellow")
    doctor_name.pack(pady=10)

    doctors = list(doctor_info.keys())
    doctor_var = tk.StringVar()
    doctor_var.set(doctors[0])

    doctor_menu = tk.OptionMenu(doctor_window, doctor_var, *doctors)
    doctor_menu.pack(pady=10)

    def show_doctor_details():
        selected_doctor = doctor_var.get()
        doctor_details = doctor_info[selected_doctor]
        messagebox.showinfo(f"Details for {selected_doctor}", doctor_details)

    details_button = tk.Button(doctor_window, text="Show Doctor Details", font=("Arial", 14), bg="orange", fg="white", command=show_doctor_details)
    details_button.pack(pady=20)

# Main window
root = tk.Tk()
root.title("AI-based Hospital Management System")
root.geometry("500x500")
root.config(bg="lightgreen")

login_button = tk.Button(root, text="Login", font=("Arial", 14), bg="orange", fg="white", command=show_login_page)
login_button.pack(pady=100, ipadx=20, ipady=5)

root.mainloop()


