# from flask import Flask, request, render_template, jsonify
# import joblib
# import pandas as pd
# import json

# # Initialize the Flask app
# app = Flask(__name__)

# # Load the trained model and encoders
# model = joblib.load('random_forest_model.pkl')
# confidence_encoder = joblib.load('confidence_encoder.pkl')
# daynight_encoder = joblib.load('daynight_encoder.pkl')

# # Load users from the users.json file
# def load_users():
#     with open('users.json') as f:
#         return json.load(f)

# # Preprocess the input data
# def preprocess_data(df):
#     # Ensure that sensor_timestamp and modis_timestamp are in datetime format
#     df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
#     df['modis_timestamp'] = pd.to_datetime(df['modis_timestamp'])

#     # Handle missing data
#     df['fire_lat'].fillna(df['fire_lat'].median(), inplace=True)
#     df['fire_long'].fillna(df['fire_long'].median(), inplace=True)
#     df['bright_ti4'].fillna(df['bright_ti4'].median(), inplace=True)
#     df['fire_radiative_power'].fillna(df['fire_radiative_power'].median(), inplace=True)
#     df['confidence'].fillna(df['confidence'].mode()[0], inplace=True)
#     df['daynight'].fillna(df['daynight'].mode()[0], inplace=True)

#     # Feature Engineering
#     df['sensor_hour'] = df['sensor_timestamp'].dt.hour
#     df['sensor_dayofweek'] = df['sensor_timestamp'].dt.dayofweek
#     df['sensor_day'] = df['sensor_timestamp'].dt.day
#     df['timestamp_diff'] = (df['sensor_timestamp'] - df['modis_timestamp']).dt.total_seconds() / 60
#     df['timestamp_diff'].fillna(0, inplace=True)

#     # Encode categorical features
#     df['confidence_encoded'] = confidence_encoder.transform(df['confidence'])
#     df['daynight_encoded'] = daynight_encoder.transform(df['daynight'])

#     # Drop unnecessary columns
#     df = df.drop(columns=['sensor_timestamp', 'modis_timestamp', 'confidence', 'daynight'])

#     return df

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the data from the POST request
#     data = request.get_json()

#     # Convert the data to a DataFrame
#     df = pd.DataFrame([data])

#     # Preprocess the input data
#     df_preprocessed = preprocess_data(df)

#     # Make the prediction
#     prediction = model.predict(df_preprocessed)

#     # Return the result as a JSON response
#     return jsonify({'prediction': int(prediction[0])})

# # Home route - to render the page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
import joblib
import pandas as pd
import json
import os

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace this with a secure key!

# Load model and encoders
model = joblib.load('random_forest_model.pkl')
confidence_encoder = joblib.load('confidence_encoder.pkl')
daynight_encoder = joblib.load('daynight_encoder.pkl')

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
