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
    def __init__(self, qualities):
        self.qualities = qualities

# Define a custom problem class
class FederatedLearningProblem(Problem):
    def __init__(self, devices, objective_functions):
        self.devices = devices
        self.objective_functions = objective_functions
        super().__init__(
            n_var=len(devices[0].qualities),
            n_obj=len(objective_functions),
            n_constr=0,
            xl=0,
            xu=1
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # Evaluate custom objective functions
        objectives = []
        for func in self.objective_functions:
            objectives.append(func(X))
        out["F"] = np.column_stack(objectives)

# Define custom objective functions
def objective1(X):
    return -X[:, 0]  # Maximize the first quality

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

# Create a list of 100 devices
np.random.seed(42)  # For reproducibility
devices = [Device(np.random.rand(6)) for _ in range(100)]

# Create the problem
qualities = np.array([device.qualities for device in devices])
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

# Identify all Pareto fronts
sorting = NonDominatedSorting()
# fronts = sorting.do(res.F, n_stop_if_sorted=len(res.F))
fronts = sorting.do(res.F)


# Plot all solutions and Pareto fronts
for i in range(problem.n_obj):
    for j in range(i + 1, problem.n_obj):
        plt.figure(figsize=(8, 6))

        # Plot all solutions
        plt.scatter(res.F[:, i], res.F[:, j], color="red", alpha=0.5, label="All Solutions")

        # Plot Pareto fronts
        for k, front in enumerate(fronts):
            pareto_front = res.F[front]
            pareto_sorted = pareto_front[np.argsort(pareto_front[:, i])]  # Sort by objective i
            plt.scatter(pareto_front[:, i], pareto_front[:, j], s=30, label=f"Pareto Front {k + 1}")
            plt.plot(pareto_sorted[:, i], pareto_sorted[:, j], linestyle="--", linewidth=1, label=f"Pareto Line {k + 1}")

        plt.xlabel(f"Objective {i + 1}")
        plt.ylabel(f"Objective {j + 1}")
        plt.title(f"Objective {i + 1} vs Objective {j + 1}")
        plt.legend()
        plt.show()

# Show Pareto-optimal devices for the first front
print("Pareto-optimal devices (indices) for the first front:")
print(fronts[0])
for idx in fronts[0]:
    print(f"Device {idx + 1}: Qualities = {devices[idx].qualities}")
