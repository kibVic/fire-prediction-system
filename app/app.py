import os
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app and database
app = Flask(__name__)

# Use environment variables to configure the database URI
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5432)  # Default to 5432 if None
POSTGRES_DB = os.getenv("POSTGRES_DB")

# Construct the database URI
DATABASE_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
print(
    f"DATABASE_URI: {DATABASE_URI}"
)  # Debugging log to check if the URI is correctly formed

# Set up app configuration
app.secret_key = os.getenv("SECRET_KEY")  # Load secret key from .env

# Load model and encoders (Make sure the file path is correct)
model_path = "model_training/models/random_forest_model.pkl"  # Adjust the path to where your model is located
confidence_encoder_path = (
    "model_training/encoders/confidence_encoder.pkl"  # Same for encoders
)
daynight_encoder_path = "model_training/encoders/daynight_encoder.pkl"

# Check if the model file exists before loading
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    print(f"Error: Model file not found at {model_path}")

# Check if the encoder files exist before loading
if os.path.exists(confidence_encoder_path):
    confidence_encoder = joblib.load(confidence_encoder_path)
else:
    print(f"Error: Confidence encoder file not found at {confidence_encoder_path}")

if os.path.exists(daynight_encoder_path):
    daynight_encoder = joblib.load(daynight_encoder_path)
else:
    print(f"Error: Daynight encoder file not found at {daynight_encoder_path}")


# Preprocess input data
def preprocess_data(df):
    df["sensor_timestamp"] = pd.to_datetime(df["sensor_timestamp"])
    df["modis_timestamp"] = pd.to_datetime(df["modis_timestamp"])

    df["fire_lat"].fillna(df["fire_lat"].median(), inplace=True)
    df["fire_long"].fillna(df["fire_long"].median(), inplace=True)
    df["bright_ti4"].fillna(df["bright_ti4"].median(), inplace=True)
    df["fire_radiative_power"].fillna(df["fire_radiative_power"].median(), inplace=True)
    df["confidence"].fillna(df["confidence"].mode()[0], inplace=True)
    df["daynight"].fillna(df["daynight"].mode()[0], inplace=True)

    df["sensor_hour"] = df["sensor_timestamp"].dt.hour
    df["sensor_dayofweek"] = df["sensor_timestamp"].dt.dayofweek
    df["sensor_day"] = df["sensor_timestamp"].dt.day
    df["timestamp_diff"] = (
        df["sensor_timestamp"] - df["modis_timestamp"]
    ).dt.total_seconds() / 60
    df["timestamp_diff"].fillna(0, inplace=True)

    df["confidence_encoded"] = confidence_encoder.transform(df["confidence"])
    df["daynight_encoded"] = daynight_encoder.transform(df["daynight"])

    df.drop(
        columns=["sensor_timestamp", "modis_timestamp", "confidence", "daynight"],
        inplace=True,
    )

    return df


# Home route - direct access to the application
@app.route("/")
def index():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        df = pd.DataFrame([data])
        df_preprocessed = preprocess_data(df)
        prediction = model.predict(df_preprocessed)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
