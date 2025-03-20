import pandas as pd
import numpy as np

# Set the number of devices
num_devices = 30

# Generate random values for each attribute
np.random.seed(42)
devices = {
    'id': np.arange(num_devices),
    'ram': np.random.uniform(0.1, 1.0, num_devices),
    'storage': np.random.uniform(0.1, 1.0, num_devices),
    'cpu': np.random.uniform(0.1, 1.0, num_devices),
    'bandwidth': np.random.uniform(0.1, 1.0, num_devices),
    'battery': np.random.uniform(0.0, 1.0, num_devices),
    'charging': np.random.randint(0, 2, num_devices)
}

# Create a DataFrame
df = pd.DataFrame(devices)

# Save to CSV
output_file = "version 5/data/devices.csv"
df.to_csv(output_file, index=False)




output_bitstring_text_file = "version 5/data/bitstring.txt"

a = []
for i in range(num_devices):
    # for the first time, it should be 1 for every device in the bit string
    a.append(np.random.randint(1, 2))
print(a)
print("count of 1s:" + str(a.count(1)))

output_bitstring_text = ""
for bit in a:
    output_bitstring_text += str(bit) + ','

output_bitstring_text = output_bitstring_text[:-1]

with open(output_bitstring_text_file, 'w') as file:
    file.write(output_bitstring_text)