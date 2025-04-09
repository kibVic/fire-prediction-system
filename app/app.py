import joblib
import pandas as pd
import json
import os
from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'kelvin' 

# Load model and encoders
model = joblib.load('../model_training/models/random_forest_model.pkl')
confidence_encoder = joblib.load('../model_training/encoders/confidence_encoder.pkl')
daynight_encoder = joblib.load('../model_training/encoders/daynight_encoder.pkl')

# Load users from users.json
def load_users():
    if not os.path.exists('users.json'):
        return {}
    with open('users.json') as f:
        return json.load(f)

# Check credentials
def validate_user(username, password):
    users = load_users()
    return users.get(username) == password

# Preprocess input data
def preprocess_data(df):
    df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
    df['modis_timestamp'] = pd.to_datetime(df['modis_timestamp'])

    df['fire_lat'].fillna(df['fire_lat'].median(), inplace=True)
    df['fire_long'].fillna(df['fire_long'].median(), inplace=True)
    df['bright_ti4'].fillna(df['bright_ti4'].median(), inplace=True)
    df['fire_radiative_power'].fillna(df['fire_radiative_power'].median(), inplace=True)
    df['confidence'].fillna(df['confidence'].mode()[0], inplace=True)
    df['daynight'].fillna(df['daynight'].mode()[0], inplace=True)

    df['sensor_hour'] = df['sensor_timestamp'].dt.hour
    df['sensor_dayofweek'] = df['sensor_timestamp'].dt.dayofweek
    df['sensor_day'] = df['sensor_timestamp'].dt.day
    df['timestamp_diff'] = (df['sensor_timestamp'] - df['modis_timestamp']).dt.total_seconds() / 60
    df['timestamp_diff'].fillna(0, inplace=True)

    df['confidence_encoded'] = confidence_encoder.transform(df['confidence'])
    df['daynight_encoded'] = daynight_encoder.transform(df['daynight'])

    df.drop(columns=['sensor_timestamp', 'modis_timestamp', 'confidence', 'daynight'], inplace=True)

    return df

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if validate_user(username, password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.')
    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Home route - only for logged-in users
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

# Prediction route - must be authenticated
@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    try:
        df = pd.DataFrame([data])
        df_preprocessed = preprocess_data(df)
        prediction = model.predict(df_preprocessed)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
