import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

# Load the device data from the CSV file
df = pd.read_csv('data/federated_devices_corrected.csv')

# Create the fitness and individual classes for DEAP
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # Minimizing energy and maximizing battery
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Define objective functions
def energy_consumption(individual):
    # Minimizing energy consumption: higher ram, cpu, storage, and bandwidth values increase energy usage
    ram, storage, cpu, bandwidth = individual[:4]
    return (ram + storage + cpu + bandwidth, )  # energy consumption (sum of these attributes)

def battery_level(individual):
    # Maximizing battery level: charging status is the key
    battery_status = individual[4]
    return (battery_status, )  # battery level (charging status)

# Create the toolbox for evolutionary algorithm
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)  # Random float values between 0 and 1
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_float, toolbox.attr_float, toolbox.attr_float, toolbox.attr_float, toolbox.attr_float), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Gaussian mutation
toolbox.register("select", tools.selNSGA2)  # NSGA-II selection

toolbox.register("evaluate", lambda ind: (energy_consumption(ind)[0], battery_level(ind)[0]))

# Create the population
population = toolbox.population(n=100)

# Define the number of generations and the probability of crossover and mutation
generation_count = 50
crossover_prob = 0.7
mutation_prob = 0.2

# Run the algorithm (NSGA-II)
algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=crossover_prob, mutpb=mutation_prob,
                          ngen=generation_count, stats=None, halloffame=None, verbose=True)

# Extract Pareto fronts
pareto_fronts = tools.sortNondominated(population, len(population), first_front_only=False)

# Adding device ids and attributes (Energy Consumption, Battery Level, Pareto Front Rank)
device_ids = df["id"].values  # Getting device ids from the CSV data

# Add attributes to the device class (energy consumption, battery level, pareto front rank)
for front_rank, front in enumerate(pareto_fronts):
    for ind in front:
        device_index = population.index(ind)
        device_id = device_ids[device_index]
        
        # Add attributes to individual (device class)
        ind.device_id = device_id
        ind.energy_consumption = energy_consumption(ind)[0]
        ind.battery_level = battery_level(ind)[0]
        ind.pareto_front_rank = front_rank

# Print all devices and their attributes
for ind in population:
    print(f"Device ID: {ind.device_id}, Energy Consumption: {ind.energy_consumption}, "
          f"Battery Level: {ind.battery_level}, Pareto Front Rank: {ind.pareto_front_rank}")

# Prepare data for CSV
pareto_front_with_attributes = []
for ind in population:
    pareto_front_with_attributes.append([ind.device_id, ind.energy_consumption, ind.battery_level, ind.pareto_front_rank])

# Convert to DataFrame for easy viewing and save to CSV
pareto_df_with_attributes = pd.DataFrame(pareto_front_with_attributes, 
                                         columns=["Device ID", "Energy Consumption", "Battery Level", "Pareto Front Rank"])

# Save to CSV
pareto_csv_filename = "data/pareto_fronts_with_device_attributes.csv"
pareto_df_with_attributes.to_csv(pareto_csv_filename, index=False)
