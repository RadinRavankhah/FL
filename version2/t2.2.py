from version2.t2_working_good import *
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter

# Step 1: Extract non-dominated solutions
pareto_indices = NonDominatedSorting().do(res.F, only_non_dominated_front=True)
pareto_front = res.F[pareto_indices]

# Step 2: Plot all solutions
plot = Scatter()
plot.add(res.F, facecolor="none", edgecolor="red", label="All Solutions")

# Step 3: Highlight Pareto front
plot.add(pareto_front, facecolor="blue", edgecolor="blue", s=50, label="Pareto Front")

# Step 4: Show the legend and display the plot
plt.legend()
plot.show()
