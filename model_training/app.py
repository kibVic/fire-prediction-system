import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load('random_forest_model.pkl')
confidence_encoder = joblib.load('confidence_encoder.pkl')
daynight_encoder = joblib.load('daynight_encoder.pkl')

# Preprocess the input data
def preprocess_data(df):
    # Ensure that sensor_timestamp and modis_timestamp are in datetime format
    df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
    df['modis_timestamp'] = pd.to_datetime(df['modis_timestamp'])

    # Handle missing data
    df['fire_lat'].fillna(df['fire_lat'].median(), inplace=True)
    df['fire_long'].fillna(df['fire_long'].median(), inplace=True)
    df['bright_ti4'].fillna(df['bright_ti4'].median(), inplace=True)
    df['fire_radiative_power'].fillna(df['fire_radiative_power'].median(), inplace=True)
    df['confidence'].fillna(df['confidence'].mode()[0], inplace=True)
    df['daynight'].fillna(df['daynight'].mode()[0], inplace=True)

    # Feature Engineering
    df['sensor_hour'] = df['sensor_timestamp'].dt.hour
    df['sensor_dayofweek'] = df['sensor_timestamp'].dt.dayofweek
    df['sensor_day'] = df['sensor_timestamp'].dt.day
    df['timestamp_diff'] = (df['sensor_timestamp'] - df['modis_timestamp']).dt.total_seconds() / 60
    df['timestamp_diff'].fillna(0, inplace=True)

    # Encode categorical features
    df['confidence_encoded'] = confidence_encoder.transform(df['confidence'])
    df['daynight_encoded'] = daynight_encoder.transform(df['daynight'])

    # Drop unnecessary columns
    df = df.drop(columns=['sensor_timestamp', 'modis_timestamp', 'confidence', 'daynight'])

    return df

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()

    # Convert the data to a DataFrame
    df = pd.DataFrame([data])

    # Preprocess the input data
    df_preprocessed = preprocess_data(df)

    # Make the prediction
    prediction = model.predict(df_preprocessed)

    # Return the result as a JSON response
    return jsonify({'prediction': int(prediction[0])})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
