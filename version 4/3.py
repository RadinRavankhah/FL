import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = "version 4/data/pareto_front_results.csv"
df = pd.read_csv(file_path)

# Compute objective values for visualization
x_values = df['ram'] + df['storage'] + df['cpu'] + df['bandwidth']
y_values = df['charging_status']
pareto_ranks = df['pareto_rank']

# Create scatter plot
plt.figure(figsize=(10, 6))
sc = plt.scatter(x_values, y_values, c=pareto_ranks, cmap='viridis', edgecolors='k', alpha=0.75)
plt.colorbar(sc, label='Pareto Rank')
plt.xlabel('Total Resource (RAM + Storage + CPU + Bandwidth)')
plt.ylabel('Charging Status')
plt.title('Pareto Front Visualization')
plt.grid(True)
plt.show()
