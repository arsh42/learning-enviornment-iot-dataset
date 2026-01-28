import sounddevice as sd
import numpy as np
import tflite_runtime.interpreter as tflite
import csv
import time
from datetime import datetime
import smbus2
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import board
import busio
import bmp280

# ---------------- CONFIG ----------------
MODEL_PATH = "1.tflite"
LABELS_CSV = "yamnet_class_map.csv"
LOG_FILE = "classroom_log.csv"
AUDIO_DURATION = 5  # seconds
I2C_BUS = 1
MQ135_CHANNEL = 0
# -----------------------------------------

# ---------- Load Labels from CSV ----------
def load_labels(csv_path):
    labels = []
    with open(csv_path, "r", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["display_name"])
    return labels

labels_list = load_labels(LABELS_CSV)

# ---------- Setup I2C and Sensors ----------
bus = smbus2.SMBus(I2C_BUS)

# BMP280 Sensor
bmp280_sensor = bmp280.BMP280(i2c_dev=bus)

# ADS1115 for MQ135
i2c_ads = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c_ads)
mq135_channel = AnalogIn(ads, ADS.P0)

# ---------- Load TFLite Model ----------
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Model input shape: {input_details[0]['shape']}")

# ---------- Audio Recording ----------
def record_audio(duration_sec):
    audio = sd.rec(int(duration_sec * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

# ---------- Audio Classification ----------
def classify_audio(duration_sec=AUDIO_DURATION):
    waveform = record_audio(duration_sec)
    expected_shape = input_details[0]['shape']

    # Match waveform to model input
    if len(expected_shape) == 1:  # 1D input
        target_len = expected_shape[0]
        if len(waveform) > target_len:
            waveform = waveform[:target_len]
        elif len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)), mode='constant')
        interpreter.set_tensor(input_details[0]['index'], waveform.astype(np.float32))
    elif len(expected_shape) == 2:  # 2D input [1, N]
        target_len = expected_shape[1]
        if len(waveform) > target_len:
            waveform = waveform[:target_len]
        elif len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)), mode='constant')
        waveform = np.expand_dims(waveform, axis=0)
        interpreter.set_tensor(input_details[0]['index'], waveform.astype(np.float32))
    else:
        raise ValueError(f"Unsupported input shape: {expected_shape}")

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_index = int(np.argmax(output_data))
    confidence = float(output_data[predicted_index])
    predicted_label = labels_list[predicted_index] if predicted_index < len(labels_list) else f"Unknown_{predicted_index}"

    return predicted_label, confidence

# ---------- Sensor Readings ----------
def read_temperature():
    return bmp280_sensor.get_temperature()

def read_pressure():
    return bmp280_sensor.get_pressure()

def read_air_quality():
    return mq135_channel.voltage

# ---------- Status Evaluation ----------
def evaluate_temperature(temp):
    if temp < 18:
        return "Too Cold"
    elif temp > 30:
        return "Too Hot"
    return "Comfortable"

def evaluate_pressure(pressure):
    if pressure < 1000:
        return "Low Pressure"
    elif pressure > 1020:
        return "High Pressure"
    return "Normal"

def evaluate_air_quality(voltage):
    if voltage < 1.0:
        return "Good"
    elif voltage < 2.0:
        return "Moderate"
    return "Poor"

# ---------- Logging ----------
def log_data(data):
    file_exists = False
    try:
        with open(LOG_FILE, 'r'):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Temperature (°C)", "Temp Status",
                "Pressure (hPa)", "Pressure Status",
                "Air Quality (V)", "Air Quality Status",
                "Noise Type", "Noise Confidence"
            ])
        writer.writerow(data)

# ---------- Main Loop ----------
def update_data():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    temp = read_temperature()
    pressure = read_pressure()
    air_quality = read_air_quality()

    temp_status = evaluate_temperature(temp)
    pressure_status = evaluate_pressure(pressure)
    air_quality_status = evaluate_air_quality(air_quality)

    noise_type, noise_confidence = classify_audio()

    # ---- Multi-line formatted print ----
    print(f"[{timestamp}]")
    print(f"Temperature: {temp:.2f}°C ({temp_status})")
    print(f"Pressure:    {pressure:.2f} hPa ({pressure_status})")
    print(f"Air Quality: {air_quality:.2f} V ({air_quality_status})")
    print(f"Noise Type:  {noise_type} ({noise_confidence:.2f})")
    print("-" * 50)  # separator line

    # ---- Log data to CSV ----
    log_data([
        timestamp, f"{temp:.2f}", temp_status,
        f"{pressure:.2f}", pressure_status,
        f"{air_quality:.2f}", air_quality_status,
        noise_type, f"{noise_confidence:.2f}"
    ])

# ---------- Run Forever ----------
if __name__ == "__main__":
    while True:
        update_data()
        time.sleep(5)  # wait between readings
