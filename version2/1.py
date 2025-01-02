import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.factory import get_problem, get_sampling, get_crossover, get_mutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter

class Device:
    def __init__(self, qualities):
        self.qualities = qualities

# Define a custom problem class
class FederatedLearningProblem(Problem):
    def __init__(self, devices):
        self.devices = devices
        super().__init__(
            n_var=len(devices[0].qualities),
            n_obj=len(devices[0].qualities),  # One objective for each quality
            n_constr=0,
            xl=0,
            xu=1
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # Objective functions for each device
        objectives = []
        for i in range(self.n_obj):
            # Maximize qualities[i] for higher CPU power or minimize for lower storage usage, etc.
            if i % 2 == 0:  # Example: even indices are maximized
                objectives.append(-X[:, i])
            else:  # Example: odd indices are minimized
                objectives.append(X[:, i])

        out["F"] = np.column_stack(objectives)

# Create a list of devices
np.random.seed(42)  # For reproducibility
devices = [Device(np.random.rand(6)) for _ in range(10)]

# Create the problem
qualities = np.array([device.qualities for device in devices])
problem = FederatedLearningProblem(devices)

# Configure the NSGA-II algorithm
algorithm = NSGA2(
    pop_size=20,
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
