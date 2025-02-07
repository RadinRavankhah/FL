import random
from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd

# Load the device data (adjust this if using a file path or other method)
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

# Display the Pareto fronts
for front in pareto_fronts:
    print("Pareto Front:")
    for ind in front:
        print(ind.fitness.values)

# Save Pareto front results to CSV
pareto_df = pd.DataFrame([ind.fitness.values for front in pareto_fronts for ind in front], columns=["Energy Consumption", "Battery Level"])
pareto_df.to_csv("data/pareto_fronts.csv", index=False)
