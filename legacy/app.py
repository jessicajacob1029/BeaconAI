from flask import Flask, render_template, request, redirect, url_for
import sqlite3

app = Flask(__name__)

# Connect to the SQLite database
def get_db_connection():
    conn = sqlite3.connect('hospital_management.db')
    conn.row_factory = sqlite3.Row
    return conn

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Register route (for registering new users)
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return redirect(url_for('index'))
    return render_template('register.html')

# Login route (for logging in users)
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password)).fetchone()
    conn.close()
    
    if user:
        return redirect(url_for('dashboard', username=username))
    else:
        return 'Invalid Credentials'

# Dashboard route
@app.route('/dashboard')
def dashboard():
    username = request.args.get('username')
    return render_template('dashboard.html', username=username)

# Book appointment route
@app.route('/book_appointment', methods=['GET', 'POST'])
def book_appointment():
    if request.method == 'POST':
        username = request.form['username']
        doctor = request.form['doctor']
        date = request.form['date']
        time = request.form['time']
        conn = get_db_connection()
        conn.execute("INSERT INTO appointments (username, doctor, date, time) VALUES (?, ?, ?, ?)",
                     (username, doctor, date, time))
        conn.commit()
        conn.close()
        return redirect(url_for('dashboard', username=username))
    return render_template('book_appointment.html')

if __name__ == "__main__":
    app.run(debug=True)
