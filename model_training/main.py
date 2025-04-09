import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Function to load data from PostgreSQL
def load_data_from_postgres(user, password, host, port, database, table_name):
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
    df = pd.read_sql(f'SELECT * FROM {table_name}', engine)
    return df

# Function to preprocess the DataFrame
def preprocess_data(df):
    # Step 1: Handle missing data
    df['fire_lat'].fillna(df['fire_lat'].median(), inplace=True)
    df['fire_long'].fillna(df['fire_long'].median(), inplace=True)
    df['bright_ti4'].fillna(df['bright_ti4'].median(), inplace=True)
    df['fire_radiative_power'].fillna(df['fire_radiative_power'].median(), inplace=True)
    df['confidence'].fillna(df['confidence'].mode()[0], inplace=True)
    df['daynight'].fillna(df['daynight'].mode()[0], inplace=True)

    # Step 2: Feature Engineering
    df['sensor_hour'] = df['sensor_timestamp'].dt.hour
    df['sensor_dayofweek'] = df['sensor_timestamp'].dt.dayofweek
    df['sensor_day'] = df['sensor_timestamp'].dt.day
    df['timestamp_diff'] = (df['sensor_timestamp'] - df['modis_timestamp']).dt.total_seconds() / 60
    df['timestamp_diff'].fillna(0, inplace=True)

    # Step 3: Encode categorical features using separate encoders
    confidence_encoder = LabelEncoder()
    df['confidence_encoded'] = confidence_encoder.fit_transform(df['confidence'])

    daynight_encoder = LabelEncoder()
    df['daynight_encoded'] = daynight_encoder.fit_transform(df['daynight'])

    # Save the encoders for use in the prediction app
    joblib.dump(confidence_encoder, 'confidence_encoder.pkl')
    joblib.dump(daynight_encoder, 'daynight_encoder.pkl')

    # Return the preprocessed dataframe
    return df

# Function to apply SMOTE to the training data
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    if y_train.value_counts().min() > 1:
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print("SMOTE applied successfully.")
        return X_train_smote, y_train_smote
    else:
        print("SMOTE cannot be applied. Insufficient samples in the minority class.")
        return X_train, y_train

# Function to train a Random Forest classifier
def train_random_forest(X_train, y_train):
    rf_classifier_weighted = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_classifier_weighted.fit(X_train, y_train)
    return rf_classifier_weighted

# Function to evaluate the model with adjusted threshold
def evaluate_model_with_threshold(rf_classifier, X_test, y_test, threshold=0.1):
    y_pred_prob = rf_classifier.predict_proba(X_test)[:, 1]
    y_pred_adjusted = (y_pred_prob >= threshold).astype(int)
    
    accuracy_adjusted = accuracy_score(y_test, y_pred_adjusted)
    print(f"Accuracy with adjusted threshold: {accuracy_adjusted:.4f}")
    print("Classification Report with adjusted threshold:")
    print(classification_report(y_test, y_pred_adjusted))

# Function to save the model
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# Function to load the saved model
def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

# Function to make predictions using the loaded model
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# Main function
def main():
    # Step 1: Load the data
    user = 'root'
    password = 'root'
    host = '172.20.0.3'
    port = '5432'
    database = 'magic_db'
    table_name = 'prediction_data'
    
    df = load_data_from_postgres(user, password, host, port, database, table_name)
    
    print(df.columns)
    # Step 2: Preprocess the data
    df = preprocess_data(df)

    # Step 3: Prepare the final DataFrame
    final_df = df.drop(columns=['sensor_timestamp', 'modis_timestamp', 'confidence', 'daynight'])

    # Step 4: Split the data
    X = final_df.drop(columns=['fire_detected'])
    y = final_df['fire_detected']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Apply SMOTE
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # Step 6: Train the model
    rf_classifier = train_random_forest(X_train_smote, y_train_smote)

    # Step 7: Save the model
    save_model(rf_classifier, 'random_forest_model.pkl')

    # Step 8: Evaluate the model
    evaluate_model_with_threshold(rf_classifier, X_test, y_test, threshold=0.1)

    # Step 9: Load and predict
    loaded_model = load_model('random_forest_model.pkl')
    predictions = make_predictions(loaded_model, X_test)
    print(f"Predictions: {predictions}")

# Run the main function
if __name__ == '__main__':
    main()
