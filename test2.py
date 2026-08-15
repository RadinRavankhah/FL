import numpy as np

# =============================================================================
# STEP 1: Fake a Pareto front (5 solutions, 3 objectives each)
# =============================================================================
# Each row is one solution's objective vector [f1, f2, f3]
# Lower is better for all three (since NSGA-II minimizes)

res_F = np.array([
    [0.10, 0.90, 0.30],   # Solution A: great hardware, terrible fairness, decent accuracy
    [0.80, 0.20, 0.25],   # Solution B: bad hardware, great fairness, good accuracy
    [0.50, 0.50, 0.10],   # Solution C: balanced, amazing accuracy
    [0.20, 0.30, 0.60],   # Solution D: good hardware, good fairness, bad accuracy
    [0.90, 0.10, 0.80],   # Solution E: terrible hardware, amazing fairness, terrible accuracy
])

print("=" * 60)
print("RAW OBJECTIVE VALUES (res.F)")
print("=" * 60)
print("          f1 (Hardware)   f2 (Fairness)   f3 (Accuracy)")
for i, row in enumerate(res_F):
    print(f"Sol {i}:     {row[0]:.2f}            {row[1]:.2f}            {row[2]:.2f}")

# =============================================================================
# STEP 2: Define weights and scalarize
# =============================================================================
# Current weights from your code:
weights = np.array([0.50, 0.30, 0.20])   # hardware, fairness, accuracy

print("\n" + "=" * 60)
print(f"WEIGHTS: {weights}")
print("Meaning: hardware=0.50, fairness=0.30, accuracy=0.20")
print("=" * 60)

# @ is matrix multiplication. Here it computes the weighted sum for each row.
# For each solution i: scalarized[i] = f1[i]*0.50 + f2[i]*0.30 + f3[i]*0.20
scalarized = res_F @ weights

print("\nSCALARIZED SCORES (lower = better):")
for i, score in enumerate(scalarized):
    # Show the manual math so you can verify
    manual = res_F[i,0]*weights[0] + res_F[i,1]*weights[1] + res_F[i,2]*weights[2]
    print(f"Sol {i}: {score:.4f}   (manual check: {manual:.4f})")

# =============================================================================
# STEP 3: Pick the best index
# =============================================================================
best_idx = np.argmin(scalarized)

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)
print(f"Lowest scalarized score: {scalarized[best_idx]:.4f}")
print(f"Best index (np.argmin):  {best_idx}")
print(f"Best solution objectives:  {res_F[best_idx]}")

# =============================================================================
# STEP 4: Compare with INVERTED weights (accuracy-first)
# =============================================================================
weights_acc_first = np.array([0.20, 0.30, 0.50])  # hardware, fairness, accuracy

print("\n" + "=" * 60)
print(f"COMPARISON: Inverted weights {weights_acc_first}")
print("Meaning: hardware=0.20, fairness=0.30, accuracy=0.50")
print("=" * 60)

scalarized_v2 = res_F @ weights_acc_first
best_idx_v2 = np.argmin(scalarized_v2)

print("\nSCALARIZED SCORES (accuracy-first):")
for i, score in enumerate(scalarized_v2):
    print(f"Sol {i}: {score:.4f}")

print(f"\nBest index with accuracy-first: {best_idx_v2}")
print(f"Best solution objectives:       {res_F[best_idx_v2]}")

# =============================================================================
# STEP 5: Visual proof of what @ does
# =============================================================================
print("\n" + "=" * 60)
print("WHAT DOES '@' DO?")
print("=" * 60)
print("res.F shape:", res_F.shape, "  (5 solutions x 3 objectives)")
print("weights shape:", weights.shape, "  (3 objectives)")
print("")
print("res.F @ weights  ==  np.dot(res_F, weights)")
print("It multiplies each objective by its weight and sums across columns.")
print("Result shape:", scalarized.shape, "  (5 scalar scores, one per solution)")