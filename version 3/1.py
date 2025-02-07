import numpy as np

# Simulating devices with random qualities
NUM_DEVICES = 20

devices = [
    {
        "id": i,
        "computation_power": np.random.uniform(1, 10),  # FLOPS (arbitrary scale)
        "bandwidth": np.random.uniform(1, 100),  # Mbps
        "battery": np.random.uniform(10, 100)  # Battery level in percentage
    }
    for i in range(NUM_DEVICES)
]

# Display a few devices
for d in devices[:5]:
    print(d)




import random
from deap import base, creator, tools, algorithms

# Define NSGA-II fitness function (maximize computation, bandwidth, battery, minimize count)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_bool", lambda: random.choice([0, 1]))  # 0: not selected, 1: selected
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NUM_DEVICES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation function for NSGA-II
def evaluate(individual):
    total_computation = sum(dev["computation_power"] for dev, sel in zip(devices, individual) if sel)
    total_bandwidth = sum(dev["bandwidth"] for dev, sel in zip(devices, individual) if sel)
    total_battery = sum(dev["battery"] for dev, sel in zip(devices, individual) if sel)
    selected_count = sum(individual)  # Number of selected devices
    return total_computation, total_bandwidth, total_battery, selected_count

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Bit flip mutation
toolbox.register("select", tools.selNSGA2)

# Run NSGA-II
POP_SIZE = 50
GENS = 100

population = toolbox.population(n=POP_SIZE)
algorithms.eaMuPlusLambda(population, toolbox, mu=POP_SIZE, lambda_=POP_SIZE, cxpb=0.5, mutpb=0.2, ngen=GENS, verbose=True)

# Select the best solution
best_solution = tools.sortNondominated(population, len(population), first_front_only=True)[0]
selected_devices = [devices[i] for i in range(NUM_DEVICES) if best_solution[0][i] == 1]

print(f"Selected {len(selected_devices)} devices:")
for d in selected_devices:
    print(d)
