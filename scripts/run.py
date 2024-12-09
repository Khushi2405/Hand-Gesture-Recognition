import serial
import time
import csv
import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress specific UserWarnings related to feature names
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have valid feature names.*")

# Set up the serial connection (replace 'COM_PORT' with your ESP32 port)
esp32_port = 'COM10'  # Change this to your ESP32 port
baud_rate = 115200
ser = serial.Serial(esp32_port, baud_rate, timeout=1)
# Load the saved model and feature columns
svm_model = joblib.load('../models/gesture_svm_model.pkl')
feature_columns = joblib.load('../models/gesture_feature_columns.pkl')
time.sleep(1)  # Wait for the connection to establish
print("All connections established")

# Function to collect sensor data and store it in the buffer
def collect_data(count=10):
    i = 0
    while i <= count :
        with open('C:/Users/Khush/Desktop/Masters_Academics/First_Semester/Mobile_Assignments/Assignment_3/Hand_Gesture_Recognition/data/{}.csv'.format(i), 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"])  # Write header
            # Start recording
            if not ser.is_open:
                ser.open()
            ser.write(b'START\n')
            print("Perform gesture: ", i)
            time.sleep(0.1)
            start_time = time.time()
            while time.time() - start_time < 2.5:
                if ser.in_waiting > 0:
                    data = ser.readline().decode('utf-8').strip()  # Read data from ESP32
                    data = data.split(": ")[-1]
                    # Split the data string into separate values
                    values = data.split(',')
                    if len(values) == 6:  # Ensure we have the right number of values
                        csv_writer.writerow(values)  # Write to file
        
        ser.close()
        input_row = combine_data('C:/Users/Khush/Desktop/Masters_Academics/First_Semester/Mobile_Assignments/Assignment_3/Hand_Gesture_Recognition/data/{}.csv'.format(i))
        gesture = predict_gesture_with_undetected(input_row, svm_model, feature_columns)
        print("Gesture performed is", gesture)
        i = i + 1
        time.sleep(1)


def combine_data(filepath):

    try:
        with open(filepath, 'r') as file:
            # Process the file (example: reading the content)
            data = pd.read_csv(file)
    
            # Normalize the IMU data (Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z)
            imu_data = data[['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']]
            # Normalize by subtracting the mean and dividing by the standard deviation
            normalized_imu_data = (imu_data - imu_data.mean(axis=0)) / imu_data.std(axis=0)
            
            # Replace original data with normalized data
            data[['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']] = imu_data
            
            # Compute summary statistics for each IMU axis
            features = {
                'Accel_X_mean': data['Accel_X'].mean(),
                'Accel_X_std': data['Accel_X'].std(),
                'Accel_X_min': data['Accel_X'].min(),
                'Accel_X_max': data['Accel_X'].max(),
                'Accel_Y_mean': data['Accel_Y'].mean(),
                'Accel_Y_std': data['Accel_Y'].std(),
                'Accel_Y_min': data['Accel_Y'].min(),
                'Accel_Y_max': data['Accel_Y'].max(),
                'Accel_Z_mean': data['Accel_Z'].mean(),
                'Accel_Z_std': data['Accel_Z'].std(),
                'Accel_Z_min': data['Accel_Z'].min(),
                'Accel_Z_max': data['Accel_Z'].max(),
                'Gyro_X_mean': data['Gyro_X'].mean(),
                'Gyro_X_std': data['Gyro_X'].std(),
                'Gyro_X_min': data['Gyro_X'].min(),
                'Gyro_X_max': data['Gyro_X'].max(),
                'Gyro_Y_mean': data['Gyro_Y'].mean(),
                'Gyro_Y_std': data['Gyro_Y'].std(),
                'Gyro_Y_min': data['Gyro_Y'].min(),
                'Gyro_Y_max': data['Gyro_Y'].max(),
                'Gyro_Z_mean': data['Gyro_Z'].mean(),
                'Gyro_Z_std': data['Gyro_Z'].std(),
                'Gyro_Z_min': data['Gyro_Z'].min(),
                'Gyro_Z_max': data['Gyro_Z'].max()
            }

            return features

    except FileNotFoundError:
        print("The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

# Define function with "undetected" category for low confidence
def predict_gesture_with_undetected(input_row, model, feature_columns, confidence_threshold=0.6):
    """
    Predicts the gesture from a single input row with "undetected" for low confidence.

    Parameters:
        input_row (list or np.array): A single row containing feature values.
        model (sklearn.svm.SVC): The trained SVM model.
        feature_columns (list): The list of feature column names.
        confidence_threshold (float): Threshold below which the prediction is "undetected."

    Returns:
        str: The predicted gesture label or "undetected".
    """
    # Convert the input row to a NumPy array if it's not already
    input_data = np.array(list(input_row.values())).reshape(1, -1)
    # Ensure the number of features matches the training data
    if input_data.shape[1] != len(feature_columns):
        raise ValueError(f"Input data must have {len(feature_columns)} features. Got {input_data.shape[1]}.")
    probabilities = model.predict_proba(input_data)  # Shape: [n_samples, n_classes]
    max_confidence = np.max(probabilities, axis=1)
    # Check if confidence is below the threshold
    if max_confidence < confidence_threshold:
        return "undetected"

    # Return the predicted label
    prediction = model.predict(input_data)
    return prediction[0]

collect_data()
