import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
import matplotlib.pyplot as plt

# =============================================================================
# 1. FIXED DEVICE VALUES (abstract, no Keras/FL noise)
# =============================================================================
# 8 devices. Each row is [obj1_cost, obj2_cost, obj3_cost].
# Lower is better. Designed so that no single device dominates all objectives.
DEVICE_VALUES = np.array([
    [1, 9, 9],   # D0: amazing at obj1, terrible at obj2/obj3
    [9, 1, 9],   # D1: amazing at obj2
    [9, 9, 1],   # D2: amazing at obj3
    [2, 8, 8],   # D3: good at obj1
    [8, 2, 8],   # D4: good at obj2
    [8, 8, 2],   # D5: good at obj3
    [5, 5, 5],   # D6: balanced mediocre
    [4, 4, 7],   # D7: balanced slightly better at obj1/obj2
], dtype=float)

class SimpleDeviceProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=8,              # 8 devices = 8-bit bitstring
            n_obj=3,              # 3 objectives (like your paper)
            n_constr=1,           # Require at least 2 devices selected
            xl=0, xu=1,
            type_var=np.bool_
        )
        self.values = DEVICE_VALUES
        self.max_val = 9.0      # For normalization

    def _evaluate(self, X, out, *args, **kwargs):
        n_samples = len(X)
        F = np.zeros((n_samples, 3))
        G = np.zeros((n_samples, 1))

        for i, bitstring in enumerate(X):
            selected = bitstring.astype(bool)
            n_selected = selected.sum()

            # Constraint: must select >= 2 devices
            # G <= 0 means satisfied, so 2 - count <= 0  => count >= 2
            G[i, 0] = 2.0 - n_selected

            if n_selected == 0:
                F[i, :] = [1.0, 1.0, 1.0]  # worst possible
            else:
                # Objective = average cost of selected devices, normalized 0-1
                sums = self.values[selected].sum(axis=0)
                F[i, :] = sums / n_selected / self.max_val

        out["F"] = F
        out["G"] = G


# =============================================================================
# 2. NSGA2 SETUP (mirrors your code)
# =============================================================================
problem = SimpleDeviceProblem()

algorithm = NSGA2(
    pop_size=30,
    sampling=BinaryRandomSampling(),
    crossover=TwoPointCrossover(),
    mutation=BitflipMutation(),
    eliminate_duplicates=True
)

res = minimize(
    problem,
    algorithm,
    ('n_gen', 15),   # 15 generations is enough for this tiny problem
    seed=42,
    verbose=True
)

# =============================================================================
# 3. VERIFY THE OUTPUT
# =============================================================================
print("\n" + "="*70)
print("FINAL PARETO FRONT")
print("="*70)
print(f"Total non-dominated solutions found: {len(res.X)}\n")

for i, (x, f) in enumerate(zip(res.X, res.F)):
    bits = "".join(['1' if b else '0' for b in x])
    print(f"  [{i:2d}]  Bitstring: {bits}  |  Obj: [{f[0]:.3f}, {f[1]:.3f}, {f[2]:.3f}]")

# =============================================================================
# 4. PROOF THAT res.X[0] IS ARBITRARY
# =============================================================================
print("\n" + "="*70)
print("PROOF: res.X[0] IS MEANINGLESS WITHOUT A SELECTION RULE")
print("="*70)

bits_0 = "".join(['1' if b else '0' for b in res.X[0]])
print(f"res.X[0]          -> {bits_0}  (Obj: {res.F[0]})")

idx_best_obj1 = np.argmin(res.F[:, 0])
idx_best_obj2 = np.argmin(res.F[:, 1])
idx_best_obj3 = np.argmin(res.F[:, 2])

bits_1 = "".join(['1' if b else '0' for b in res.X[idx_best_obj1]])
bits_2 = "".join(['1' if b else '0' for b in res.X[idx_best_obj2]])
bits_3 = "".join(['1' if b else '0' for b in res.X[idx_best_obj3]])

print(f"Best for Obj1     -> {bits_1}  (Obj: {res.F[idx_best_obj1]})")
print(f"Best for Obj2     -> {bits_2}  (Obj: {res.F[idx_best_obj2]})")
print(f"Best for Obj3     -> {bits_3}  (Obj: {res.F[idx_best_obj3]})")

# =============================================================================
# 5. CORRECT WAY TO PICK A SOLUTION (scalarization)
# =============================================================================
print("\n" + "="*70)
print("CORRECT SELECTION: Weighted scalarization")
print("="*70)

# Example: 20% hardware, 30% fairness, 50% accuracy
weights = np.array([0.2, 0.3, 0.5])
scalar = res.F @ weights
best_idx = np.argmin(scalar)

bits_best = "".join(['1' if b else '0' for b in res.X[best_idx]])
print(f"Chosen weights: {weights}")
print(f"Best index: {best_idx} -> {bits_best} (scalar={scalar[best_idx]:.4f})")

# =============================================================================
# 6. PLOT THE PARETO FRONT
# =============================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot all Pareto-optimal points
ax.scatter(res.F[:, 0], res.F[:, 1], res.F[:, 2], 
           c='red', s=100, depthshade=True, label='Pareto Front')

# Highlight the arbitrary [0] point
ax.scatter([res.F[0, 0]], [res.F[0, 1]], [res.F[0, 2]], 
           c='blue', s=200, marker='X', edgecolors='black', label='res.X[0] (arbitrary!)')

# Highlight the scalarized choice
ax.scatter([res.F[best_idx, 0]], [res.F[best_idx, 1]], [res.F[best_idx, 2]], 
           c='lime', s=200, marker='*', edgecolors='black', label='Scalarized Choice')

ax.set_xlabel('Obj 1 (Hardware-ish)')
ax.set_ylabel('Obj 2 (Fairness-ish)')
ax.set_zlabel('Obj 3 (Accuracy-ish)')
ax.set_title('NSGA2 Pareto Front Test\nRed = all non-dominated solutions')
ax.legend()
plt.tight_layout()
plt.show()