import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Function to load data from PostgreSQL
def load_data_from_postgres(user, password, host, port, database, table_name):
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
    df = pd.read_sql(f'SELECT * FROM {table_name}', engine)
    return df

# Function to handle missing data
def handle_missing_data(df):
    df['fire_lat'].fillna(df['fire_lat'].median(), inplace=True)
    df['fire_long'].fillna(df['fire_long'].median(), inplace=True)
    df['bright_ti4'].fillna(df['bright_ti4'].median(), inplace=True)
    df['fire_radiative_power'].fillna(df['fire_radiative_power'].median(), inplace=True)
    
    df['confidence'].fillna(df['confidence'].mode()[0], inplace=True)
    df['daynight'].fillna(df['daynight'].mode()[0], inplace=True)
    return df

# Function for feature engineering
def feature_engineering(df):
    # Extract time features from sensor_timestamp
    df['sensor_hour'] = df['sensor_timestamp'].dt.hour
    df['sensor_dayofweek'] = df['sensor_timestamp'].dt.dayofweek
    df['sensor_day'] = df['sensor_timestamp'].dt.day
    
    # Calculate the time difference between sensor_timestamp and modis_timestamp
    df['timestamp_diff'] = (df['sensor_timestamp'] - df['modis_timestamp']).dt.total_seconds() / 60
    df['timestamp_diff'].fillna(0, inplace=True)
    
    return df

# Function to encode categorical features
def encode_categorical_features(df):
    label_encoder = LabelEncoder()
    df['confidence_encoded'] = label_encoder.fit_transform(df['confidence'])
    df['daynight_encoded'] = label_encoder.fit_transform(df['daynight'])
    
    # Save the label encoder after training
    joblib.dump(label_encoder, 'label_encoder.pkl')
    return df

# Function for scaling numerical features
def scale_numerical_features(df):
    scaler = StandardScaler()
    df[['sensor_value', 'fire_lat', 'fire_long', 'fire_radiative_power']] = scaler.fit_transform(
        df[['sensor_value', 'fire_lat', 'fire_long', 'fire_radiative_power']])
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
        return X_train, y_train  # Use the original data if SMOTE can't be applied

# Function to train a Random Forest classifier with class weights
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


# Main function to load, preprocess, train, evaluate, and save the model
def main():
    # Step 1: Load the data
    user = 'root'
    password = 'root'
    host = '172.20.0.2'
    port = '5432'
    database = 'magic_db'
    table_name = 'prediction_data'
    
    df = load_data_from_postgres(user, password, host, port, database, table_name)
    
    # Step 2: Data Preprocessing
    df = handle_missing_data(df)
    df = feature_engineering(df)
    df = encode_categorical_features(df)
    df = scale_numerical_features(df)

    # Step 3: Prepare the final DataFrame
    final_df = df.drop(columns=['sensor_timestamp', 'modis_timestamp', 'confidence', 'daynight'])
    
    # Step 4: Split the data into features (X) and target (y)
    X = final_df.drop(columns=['fire_detected'])
    y = final_df['fire_detected']

    print(X.columns)

    # Step 5: Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #print x test
    print(X_test)

    #print y test
    print(y_test)
    
    # Step 6: Apply SMOTE if applicable
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    
    # Step 7: Train the Random Forest classifier
    rf_classifier = train_random_forest(X_train_smote, y_train_smote)
    
    # Step 8: Save the trained model
    save_model(rf_classifier, 'random_forest_model.pkl')
    
    # Step 9: Evaluate the model with adjusted threshold
    evaluate_model_with_threshold(rf_classifier, X_test, y_test, threshold=0.1)

    # Step 10: Load the saved model and make predictions
    loaded_model = load_model('random_forest_model.pkl')
    predictions = make_predictions(loaded_model, X_test)
    print(f"Predictions: {predictions}")

# Run the main function
if __name__ == '__main__':
    main()
