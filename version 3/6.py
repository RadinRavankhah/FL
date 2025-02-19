import pandas as pd
import numpy as np

def dominates(row, candidate):
    """Check if a row dominates a candidate in a multi-objective sense."""
    return all(row >= candidate) and any(row > candidate)

def fast_non_dominated_sort(values):
    """Perform non-dominated sorting to determine Pareto fronts."""
    num_points = values.shape[0]
    domination_count = np.zeros(num_points, dtype=int)
    dominated_solutions = [[] for _ in range(num_points)]
    pareto_fronts = []

    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                if dominates(values[i], values[j]):
                    dominated_solutions[i].append(j)
                elif dominates(values[j], values[i]):
                    domination_count[i] += 1

        if domination_count[i] == 0:
            if len(pareto_fronts) == 0:
                pareto_fronts.append([])
            pareto_fronts[0].append(i)

    current_front = 0
    while current_front < len(pareto_fronts):
        next_front = []
        for i in pareto_fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        if next_front:
            pareto_fronts.append(next_front)
        current_front += 1

    return pareto_fronts

# Load data
file_path = "data/federated_devices_corrected.csv"
df = pd.read_csv(file_path)

# Extract the objective values
objectives = np.column_stack([
    -(df['ram'] + df['storage'] + df['cpu'] + df['bandwidth']),  # Maximization converted to minimization
    -df['charging_status']  # Maximization converted to minimization
])

# Compute Pareto fronts
pareto_fronts = fast_non_dominated_sort(objectives)

# Assign Pareto ranks
pareto_rank = np.zeros(len(df), dtype=int)
for rank, front in enumerate(pareto_fronts, start=1):
    for idx in front:
        pareto_rank[idx] = rank

df['pareto_rank'] = pareto_rank

# Save results to CSV
output_file = "data/pareto_front_results.csv"
df.to_csv(output_file, index=False)
