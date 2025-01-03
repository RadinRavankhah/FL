import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Define the custom problem
class CustomProblem(Problem):
    def __init__(self):
        super().__init__(n_var=6, n_obj=3, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        # Example objectives for visualization
        f1 = X[:, 0]
        f2 = (1 + X[:, 1]) / X[:, 2]
        f3 = np.sum(X, axis=1)
        out["F"] = np.column_stack([f1, f2, f3])

# Define the problem
problem = CustomProblem()

# Configure NSGA-II
algorithm = NSGA2(pop_size=100)

# Solve the problem
res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

# Perform non-dominated sorting for Pareto front visualization
sorting = NonDominatedSorting()
fronts = sorting.do(res.F)

# Plot the results using pymoo Scatter
for i in range(problem.n_obj):
    for j in range(i + 1, problem.n_obj):
        plot = Scatter(title=f"Objective {i + 1} vs Objective {j + 1}")
        plot.add(res.F[:, [i, j]], facecolor="none", edgecolor="red", label="Solutions")

        # Add Pareto fronts
        for k, front in enumerate(fronts):
            plot.add(res.F[front][:, [i, j]], plot_type="line", alpha=0.7, label=f"Pareto Front {k + 1}")

        plot.show()
