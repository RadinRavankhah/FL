import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter


class Device:
    def __init__(self, qualities):
        self.qualities = qualities

# Example custom objective functions
def objective_1(qualities):
    # Maximize CPU power (e.g., assume it's the first quality)
    return -qualities[0]

def objective_2(qualities):
    # Minimize storage usage (e.g., assume it's the second quality)
    return qualities[1] ** 2  # Example: penalize higher storage usage quadratically

def objective_3(qualities):
    # Complex function: Minimize energy consumption (e.g., third quality)
    return np.sin(qualities[2]) + 0.5 * qualities[3]

# Define a custom problem class
class FederatedLearningProblem(Problem):
    def __init__(self, devices, objective_functions):
        self.devices = devices
        self.objective_functions = objective_functions
        super().__init__(
            n_var=len(devices[0].qualities),
            n_obj=len(objective_functions),  # Number of objectives = number of functions provided
            n_constr=0,
            xl=0,
            xu=1
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # Evaluate custom objective functions for each solution
        objectives = []
        for func in self.objective_functions:
            # Apply each objective function to all solutions
            values = np.array([func(x) for x in X])
            objectives.append(values)

        # Combine all objectives into a matrix
        out["F"] = np.column_stack(objectives)

# Create a list of devices
np.random.seed(42)  # For reproducibility
devices = [Device(np.random.rand(6)) for _ in range(10)]

# Define the custom objective functions
objective_functions = [objective_1, objective_2, objective_3]

# Create the problem
qualities = np.array([device.qualities for device in devices])
problem = FederatedLearningProblem(devices, objective_functions)

# Configure the NSGA-II algorithm
algorithm = NSGA2(
    pop_size=10,
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

# Visualize the Pareto front
plot = Scatter()
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()

# Show selected devices and their qualities
selected_devices = [devices[i] for i in np.argmin(res.F, axis=0)]
for idx, device in enumerate(selected_devices):
    print(f"Selected Device {idx + 1}: Qualities = {device.qualities}")
