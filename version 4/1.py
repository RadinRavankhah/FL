import pandas as pd
import numpy as np

# Set the number of devices
num_devices = 1000

# Generate random values for each attribute
np.random.seed(42)
devices = {
    'id': np.arange(num_devices),
    'ram': np.random.uniform(0.1, 1.0, num_devices),
    'storage': np.random.uniform(0.1, 1.0, num_devices),
    'cpu': np.random.uniform(0.1, 1.0, num_devices),
    'bandwidth': np.random.uniform(0.1, 1.0, num_devices),
    'charging_status': np.random.uniform(0.0, 1.0, num_devices)
}

# Create a DataFrame
df = pd.DataFrame(devices)

# Save to CSV
output_file = "version 4/data/federated_devices_1000.csv"
df.to_csv(output_file, index=False)
