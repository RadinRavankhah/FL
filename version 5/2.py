import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
# from sklearn.model_selection import train_test_split

# Device Class
class Device:
    def __init__(self, device_id, ram, storage, cpu, bandwidth, battery, charging):
        self.device_id = device_id
        self.ram = ram
        self.storage = storage
        self.cpu = cpu
        self.bandwidth = bandwidth
        self.battery = battery
        self.charging = charging
        self.energy_consumption = ram + storage + cpu + bandwidth
        self.model = self.create_model()
        self.data = None  # Placeholder for dataset partition

    def create_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), 
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    

# Load dataset from CSV
csv_file = 'version 5/data/devices.csv'
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip().str.lower()


# Convert CSV rows into device objects
devices = []
for _, row in df.iterrows():
    device = Device(
        row['id'], row['ram'], row['storage'], row['cpu'], row['bandwidth'], row['battery'],
        row.get('charging', 0)
    )
    devices.append(device)

# Global Model
