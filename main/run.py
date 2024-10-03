import serial
import time
import csv

def main():
    # Set up the serial connection (replace 'COM_PORT' with your ESP32 port)
    esp32_port = 'COM9'  # Change this to your ESP32 port
    baud_rate = 115200
    ser = serial.Serial(esp32_port, baud_rate, timeout=1)

    time.sleep(2)  # Wait for the connection to establish

    # Open a file to save the data
    with open('gesture_LEFT.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"])  # Write header

        # Start recording
        ser.write(b'START\n')
        print("Recording started...")

        try:
            while True:
                if ser.in_waiting > 0:
                    data = ser.readline().decode('utf-8').strip()  # Read data from ESP32
                    print(data)  # Print to console
                    data = data.split(": ")[-1]
                    # Split the data string into separate values
                    values = data.split(',')
                    if len(values) == 6:  # Ensure we have the right number of values
                        csv_writer.writerow(values)  # Write to file

        except KeyboardInterrupt:
            # Stop recording
            ser.write(b'STOP\n')
            print("Recording stopped.")
    
    ser.close()

if __name__ == "__main__":
    main()
