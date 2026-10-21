import sqlite3
import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Create database tables
def create_tables():
    conn = sqlite3.connect('hospital.db')
    cur = conn.cursor()

    # Create users table for login
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        password TEXT
    );
    """)

    # Create appointments table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        doctor TEXT,
        date TEXT,
        time TEXT
    );
    """)

    # Create pharmacy table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pharmacy (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        medicines TEXT,
        date TEXT,
        time TEXT
    );
    """)

    conn.commit()
    conn.close()

# Call the function to create the tables
create_tables()

# Train disease prediction model (example)
def train_model():
    # Example data for model training (replace with real data)
    data = {'Fever': [1, 0, 1, 1],
            'Cough': [1, 1, 0, 0],
            'Headache': [0, 1, 1, 0],
            'Fatigue': [1, 1, 0, 1],
            'Disease': ['Flu', 'Cold', 'Covid', 'Covid']}
    
    df = pd.DataFrame(data)
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    
    # Train the model
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model()

# Function to make a prediction
def predict_disease(symptoms):
    feature_names = ['Fever', 'Cough', 'Headache', 'Fatigue']
    input_data = np.array([symptoms])
    input_data_df = pd.DataFrame(input_data, columns=feature_names)
    prediction = model.predict(input_data_df)
    return prediction[0]

# Register User (for login functionality)
def register_user(username, password):
    conn = sqlite3.connect('hospital.db')
    cur = conn.cursor()
    cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

# Book appointment
def book(username, doctor_var, date_entry, time_entry):
    conn = sqlite3.connect('hospital.db')
    cur = conn.cursor()
    cur.execute("INSERT INTO appointments (username, doctor, date, time) VALUES (?, ?, ?, ?)", 
                (username, doctor_var.get(), date_entry.get(), time_entry.get()))
    conn.commit()
    conn.close()

# View bookings for the user
def view_bookings(username):
    conn = sqlite3.connect('hospital.db')
    cur = conn.cursor()
    cur.execute("SELECT doctor, date, time FROM appointments WHERE username=?", (username,))
    rows = cur.fetchall()
    
    # Example: print the bookings to console, update this to show in the UI as needed
    for row in rows:
        print(row)
    
    conn.close()

# View pharmacy orders for the user
def view_pharmacy_orders(username):
    conn = sqlite3.connect('hospital.db')
    cur = conn.cursor()
    cur.execute("SELECT medicines, date, time FROM pharmacy WHERE username=?", (username,))
    rows = cur.fetchall()
    
    # Example: print the orders to console, update this to show in the UI as needed
    for row in rows:
        print(row)
    
    conn.close()

# Tkinter UI for user interactions

# Login window
def login_window():
    def login_action():
        username = username_entry.get()
        password = password_entry.get()
        
        conn = sqlite3.connect('hospital.db')
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cur.fetchone()
        conn.close()

        if user:
            messagebox.showinfo("Login", "Login Successful!")
            main_ui(username)  # Call the main UI after login
            login.destroy()    # Close login window
        else:
            messagebox.showerror("Login Error", "Invalid username or password!")

    login = tk.Tk()
    login.title("Login")

    tk.Label(login, text="Username:").grid(row=0, column=0)
    username_entry = tk.Entry(login)
    username_entry.grid(row=0, column=1)

    tk.Label(login, text="Password:").grid(row=1, column=0)
    password_entry = tk.Entry(login, show="*")
    password_entry.grid(row=1, column=1)

    login_button = tk.Button(login, text="Login", command=login_action)
    login_button.grid(row=2, columnspan=2)

    register_button = tk.Button(login, text="Register", command=lambda: register_window())
    register_button.grid(row=3, columnspan=2)

    login.mainloop()

# Register window
def register_window():
    def register_action():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            register_user(username, password)
            messagebox.showinfo("Registration", "User Registered Successfully!")
            register.destroy()
        else:
            messagebox.showerror("Registration Error", "Please fill out both fields!")

    register = tk.Tk()
    register.title("Register")

    tk.Label(register, text="Username:").grid(row=0, column=0)
    username_entry = tk.Entry(register)
    username_entry.grid(row=0, column=1)

    tk.Label(register, text="Password:").grid(row=1, column=0)
    password_entry = tk.Entry(register, show="*")
    password_entry.grid(row=1, column=1)

    register_button = tk.Button(register, text="Register", command=register_action)
    register_button.grid(row=2, columnspan=2)

    register.mainloop()

# Main UI window (after login)
def main_ui(username):
    def book_appointment():
        doctor_var = doctor_choice.get()
        date = date_entry.get()
        time = time_entry.get()
        book(username, doctor_var, date, time)
        messagebox.showinfo("Appointment", "Appointment booked successfully!")

    def view_bookings_ui():
        view_bookings(username)

    def view_pharmacy_ui():
        view_pharmacy_orders(username)

    main = tk.Tk()
    main.title("Hospital Management System")

    # Welcome label
    tk.Label(main, text=f"Welcome, {username}!", font=("Arial", 20)).grid(row=0, columnspan=2)

    # Appointment booking
    tk.Label(main, text="Doctor:").grid(row=1, column=0)
    doctor_choice = tk.StringVar(main)
    doctor_choice.set("Select Doctor")
    doctor_menu = tk.OptionMenu(main, doctor_choice, "Cardiologist", "Neurologist", "Pulmonologist", "ENT Specialist", "General Physician", "Gastroenterologist", "Orthopedic")
    doctor_menu.grid(row=1, column=1)

    tk.Label(main, text="Date (YYYY-MM-DD):").grid(row=2, column=0)
    date_entry = tk.Entry(main)
    date_entry.grid(row=2, column=1)

    tk.Label(main, text="Time (HH:MM):").grid(row=3, column=0)
    time_entry = tk.Entry(main)
    time_entry.grid(row=3, column=1)

    book_button = tk.Button(main, text="Book Appointment", command=book_appointment)
    book_button.grid(row=4, columnspan=2)

    # View Bookings
    view_button = tk.Button(main, text="View Bookings", command=view_bookings_ui)
    view_button.grid(row=5, columnspan=2)

    # View Pharmacy Orders
    pharmacy_button = tk.Button(main, text="View Pharmacy Orders", command=view_pharmacy_ui)
    pharmacy_button.grid(row=6, columnspan=2)

    main.mainloop()

# Start the login window
login_window()




