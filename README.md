# Hand Gesture Recognition Using ESP32S3 and SVM model

## Overview

This project utilizes sensor data from an ESP32S3 microcontroller and MPU6050 IMU sensor to recognize hand gestures using a pre-trained Support Vector Machine (SVM) model. The system collects accelerometer and gyroscope data, processes it to extract features, and predicts the gesture performed using a trained machine-learning model.

## Features

- **IMU Data Collection**: Collect motion data (accelerometer and gyroscope) from ESP32 and MPU6050 IMU sensors.
- **Data Normalization and Processing**: Normalize and preprocess the collected data to ensure consistency.
- **Feature Extraction**: Extract statistical features such as mean, standard deviation, min, and max for gesture recognition.
- **Gesture Prediction**: Classify gestures using a trained Support Vector Machine (SVM) model, including a confidence-based "undetected" category for low-confidence predictions.
- **Data Logging**: Log collected data to CSV files for further analysis and model improvement.

## 📁 Project Structure

- **`data`**  
  Contains input data collected for real time gestures which is then picked up and for feature extraction before predicting

- **`main`**  
  Core logic for the ESP32, including:
  - `i2c_simple_main.c`: Main code for handling sensors and data collection.
  - `idf_component.yml`: ESP-IDF configuration file.

- **`managed_components`**  
  Includes reusable libraries or components for the project.

- **`models`**  
  Stores machine learning artifacts:
  - `gesture_feature_columns.pkl`: Metadata for the feature set.
  - `gesture_svm_model.pkl`: Trained SVM model for gesture classification.

- **`notebook`**  
  - `svm.ipynb`: Notebook for SVM training, evaluation, and visualization.

- **`scripts`**  
  Utility scripts for automation:
  - `run.py`: Main script to run.

## 🔧 Training the Model

1. **Data Collection**:  
   Each gesture's data is collected using ESP32 sensors. The `combine_data` function is used to merge data from different gestures into a single dataset.

2. **Feature Extraction**:  
   Statistical features such as mean, standard deviation, and variance are extracted from the collected data.

3. **Model Training**:  
   - The aggregated dataset (`combined_gesture_data.csv`) is loaded.
   - Features (`X`) and labels (`y`) are separated.
   - An SVM model is trained using the Scikit-learn `SVC` class with a linear kernel.
   - The trained model is saved as `gesture_svm_model.pkl`.

4. **Evaluation**:  
   - The model is evaluated on the training data, and a confusion matrix is generated to visualize performance.


## 🔧 Setup Instructions

### 🖥️ Hardware Setup
1. **Connect the ESP32S3 Microcontroller**:
   - To run this example, you should have one ESP32, ESP32-S, ESP32-C or ESP32-H based development board as well as a MPU6050. MPU6050 is a inertial measurement unit, which contains a accelerometer, gyroscope as well as a magnetometer.
   - Connect your ESP32S3 and IMU sensor to your computer using a USB cable.
   - Ensure that the ESP32 is programmed to send IMU (Inertial Measurement Unit) data in the required format.
   
#### Pin Assignment:
|                  | SDA             | SCL           |
| ---------------- | -------------- | -------------- |
| ESP I2C Master   | I2C_MASTER_SDA | I2C_MASTER_SCL |
| MPU9250 Sensor   | SDA            | SCL            |


### Install Required Libraries

To install the necessary libraries for the project, run the following command:

```bash
pip install numpy pandas joblib pyserial scikit-learn
```

### ⚙️ Update Serial Port
1. Locate the `esp32_port` variable in the relevant script.  
2. Update the variable to match your ESP32's serial port:  
   - For **Windows**, use `COMx` (e.g., `COM10`). This can be found in device manager under ports  
   - For **Linux/Mac**, use `/dev/ttyUSBx` or `/dev/ttyACMx` (e.g., `/dev/ttyUSB0`).
3. To establish a serial connection with the ESP32 for collecting sensor data:

```python
#### Update the ESP32 serial port
esp32_port = 'COM10'  # Replace 'COM10' with your ESP32's port
baud_rate = 115200

#### Initialize serial connection
ser = serial.Serial(esp32_port, baud_rate, timeout=1)
```

### Build and Flash

Enter `idf.py -p PORT flash monitor` to build, flash and monitor the project.

(To exit the serial monitor, type ``Ctrl-]``.)

See the [Getting Started Guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html) for full steps to configure and use ESP-IDF to build projects.

## 📊 Visualization Example

The `svm.ipynb` notebook generates a confusion matrix for gesture classification with 100 % accuracy: 
![Confusion Matrix Example](https://drive.google.com/uc?id=1Ea7dtWFG_5feM92Y4wrROAV9xHWgKTLH)


## Troubleshooting

(For any technical queries, please open an [issue](https://github.com/espressif/esp-idf/issues) on GitHub. We will get back to you as soon as possible.) -->

## 💡 Future Work
- Extend support for additional gestures.
- Optimize for deployment on resource-constrained devices.


## 🤝 Contributing
Contributions are welcome! Please submit a pull request or raise an issue for any suggestions or bugs.


## 🛡️ License
This project is licensed under the [MIT License](LICENSE).
