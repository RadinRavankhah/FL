import pandas as pd
import numpy as np
import random

# Function to check if one solution dominates another
def dominates(row, candidate):
    return all(row <= candidate) and any(row < candidate)

# Fast Non-Dominated Sorting
def fast_non_dominated_sort(values):
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

# Crowding Distance Calculation
def calculate_crowding_distance(values, front):
    num_objectives = values.shape[1]
    distances = np.zeros(len(front))
    sorted_indices = [sorted(front, key=lambda x: values[x, m]) for m in range(num_objectives)]
    
    for m in range(num_objectives):
        distances[0] = distances[-1] = np.inf  # Edge solutions get max distance
        min_value = values[sorted_indices[0][0], m]
        max_value = values[sorted_indices[0][-1], m]
        
        if max_value - min_value == 0:
            continue  # Avoid division by zero
        
        for i in range(1, len(front) - 1):
            distances[i] += (values[sorted_indices[m][i + 1], m] - values[sorted_indices[m][i - 1], m]) / (max_value - min_value)
    return distances

# Tournament Selection
def tournament_selection(population, objectives, pareto_fronts, tournament_size=2):
    selected = []
    while len(selected) < len(population):
        # contenders = random.sample(range(len(population)), tournament_size)
        tournament_size = min(tournament_size, len(population))  # Ensure tournament size is valid
        contenders = random.sample(range(len(population)), tournament_size) if len(population) > 1 else [0]
        
        best = sorted(contenders, key=lambda x: (pareto_rank[x], -crowding_dist[x]))[0]
        selected.append(best)
    return selected

# Crossover and Mutation
def crossover_and_mutate(parent1, parent2, mutation_rate=0.1):
    child = (parent1 + parent2) / 2  # Simple arithmetic crossover
    if random.random() < mutation_rate:
        mutation = np.random.normal(0, 0.1, size=child.shape)
        child += mutation
    return child

# Load dataset
file_path = "version 4/data/federated_devices_1000.csv"
df = pd.read_csv(file_path)

# Extract objectives (converted for minimization)
population = df[['ram', 'storage', 'cpu', 'bandwidth', 'charging_status']].values
objectives = np.column_stack([-population[:, :-1].sum(axis=1), -population[:, -1]])

# NSGA-II Main Loop
num_generations = 50
population_size = len(population)

for generation in range(num_generations):
    pareto_fronts = fast_non_dominated_sort(objectives)
    pareto_rank = np.zeros(population_size, dtype=int)
    
    for rank, front in enumerate(pareto_fronts):
        for idx in front:
            pareto_rank[idx] = rank
    
    crowding_dist = np.zeros(population_size)
    for front in pareto_fronts:
        if len(front) > 1:
            crowding_dist[front] = calculate_crowding_distance(objectives, front)
    
    # Select parents
    selected_indices = tournament_selection(population, objectives, pareto_fronts)
    parents = population[selected_indices]
    
    # Create next generation
    new_population = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            child = crossover_and_mutate(parents[i], parents[i + 1])
            new_population.append(child)
    
    # population = np.array(new_population)
    
    if len(new_population) == 0:
        print("Warning: No new individuals were generated! Reusing previous population.")
        new_population = population  # Keep the last valid population instead of breaking the loop
    population = np.array(new_population)
    # Ensure the population remains 2D
    if population.ndim == 1:
        population = population.reshape(1, -1)
        
    objectives = np.column_stack([-population[:, :-1].sum(axis=1), -population[:, -1]])

# Save final results
df['pareto_rank'] = pareto_rank
df.to_csv("version 4/data/nsga2_results.csv", index=False)
