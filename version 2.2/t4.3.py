import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Define the Device class
class Device:
    # def __init__(self, qualities):
    #     self.qualities = np.array(qualities)  # Ensure qualities are stored as a NumPy array
    def __init__(self, qualities):
        self.qualities = np.array(qualities, dtype=object)  # Store mixed types correctly
        self.qualities[0] = int(self.qualities[0])  # Ensure the first quality is an integer


# Define a custom problem class
class FederatedLearningProblem(Problem):
    def __init__(self, devices, objective_functions):
        self.devices = devices
        self.objective_functions = objective_functions
        super().__init__(
            n_var=len(devices[0].qualities),
            n_obj=len(objective_functions),
            n_constr=0,
            xl=np.array([1] + [0] * (len(devices[0].qualities) - 1)),  # First quality between 1-64
            xu=np.array([64] + [1] * (len(devices[0].qualities) - 1))  # Others between 0-1
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # Evaluate custom objective functions
        objectives = []
        for func in self.objective_functions:
            objectives.append(func(X))
        out["F"] = np.column_stack(objectives)  # Store results in output dictionary

# Modified objective function
def objective1(X):
    """Normalize the first quality to [0,1] and maximize it."""
    normalized_q1 = (X[:, 0] - 1) / (64 - 1)  # Normalize to range [0,1]
    return -normalized_q1  # Maximization is converted to minimization

# Keeping other objective functions unchanged
def objective2(X):
    return X[:, 1] ** 2  # Minimize the square of the second quality

def objective3(X):
    return np.sin(X[:, 2])  # Minimize the sine of the third quality

def objective4(X):
    return X[:, 3]  # Minimize the fourth quality directly

def objective5(X):
    return X[:, 4] + X[:, 5]  # Minimize the sum of the fifth and sixth qualities

def objective6(X):
    return -np.sum(X, axis=1)  # Maximize the sum of all qualities

# Create a list of 100 devices with the new range constraints
np.random.seed(42)  # For reproducibility
devices = [Device([np.random.randint(1, 65)] + list(np.random.rand(5))) for _ in range(100)]

# Create the problem instance
problem = FederatedLearningProblem(devices, [
    objective1, objective2, objective3, objective4, objective5, objective6
])

# Configure the NSGA-II algorithm
algorithm = NSGA2(
    pop_size=100,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# Solve the problem
res = minimize(
    problem,
    algorithm,
    ('n_gen', 50),
    seed=42,
    verbose=False
)

# Extract Pareto-optimal solutions
pareto_indices = NonDominatedSorting().do(res.F, only_non_dominated_front=True)
pareto_front = res.F[pareto_indices]

# Plot each pair of objectives in separate windows
for i in range(problem.n_obj):
    for j in range(i + 1, problem.n_obj):
        plt.figure(figsize=(8, 6))
        plt.scatter(res.F[:, i], res.F[:, j], color="red", alpha=0.5, label="All Solutions")
        plt.scatter(pareto_front[:, i], pareto_front[:, j], color="blue", s=30, label="Pareto Front")
        plt.xlabel(f"Objective {i + 1}")
        plt.ylabel(f"Objective {j + 1}")
        plt.legend(loc="best")
        plt.title(f"Objective {i + 1} vs Objective {j + 1}")
        plt.show()

# # Show Pareto-optimal devices
# print("Pareto-optimal devices (indices):")
# print(pareto_indices)
# for idx in pareto_indices:
#     print(f"Device {idx + 1}: Qualities = {devices[idx].qualities}")

# Show Pareto-optimal devices
print("Pareto-optimal devices (indices):")
print(pareto_indices)

for idx in pareto_indices:
    qualities = devices[idx].qualities
    formatted_qualities = [qualities[0]] + [f"{q:.4f}" for q in qualities[1:]]  # Keep first as int, format floats
    print(f"Device {idx + 1}: Qualities = {formatted_qualities}")







# Get all Pareto fronts (not just the first one)
all_pareto_fronts = NonDominatedSorting().do(res.F)

# Print all Pareto fronts and their devices
print("\n🔹 Pareto Fronts and Devices 🔹\n")

for front_idx, front in enumerate(all_pareto_fronts):
    print(f"⭐ Pareto Front {front_idx + 1} (Rank {front_idx}) ⭐")
    
    for idx in front:
        qualities = devices[idx].qualities
        formatted_qualities = [qualities[0]] + [f"{q:.4f}" for q in qualities[1:]]  # Format floats
        print(f"  - Device {idx + 1}: Qualities = {formatted_qualities}")
    
    print("\n" + "-" * 50 + "\n")  # Separator between fronts
