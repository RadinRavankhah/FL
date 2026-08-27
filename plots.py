import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy_fairness(
    methods,
    acc_means,
    acc_stds,
    fairness_means,
    fairness_stds,
    save_path=None,
):
    """
    Plot final global test accuracy and fairness (1/std of client accuracies)
    as side-by-side bar charts with error bars (mean ± std over 3 runs).

    Parameters
    ----------
    methods : list[str]
        Method names for the x-axis, e.g.
        ["NSGA2", "Full Selection", "Random 75%", "Random 50%", "Random 25%"].
    acc_means, acc_stds : list[float]
        Mean and std of final global test accuracy (as fractions, 0-1) per method.
    fairness_means, fairness_stds : list[float]
        Mean and std of the fairness metric (1/std of client accuracies) per method.
    save_path : str, optional
        If given, saves the figure to this path (e.g. "results.png").
    """
    colors = plt.get_cmap("tab10").colors[: len(methods)]
    x = np.arange(len(methods))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Global test accuracy
    ax1.bar(
        x, acc_means, yerr=acc_stds, capsize=5, color=colors,
        edgecolor="black", linewidth=0.8, error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    ax1.set_title("Final Global Test Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=20, ha="right")
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)

    # (b) Fairness metric
    ax2.bar(
        x, fairness_means, yerr=fairness_stds, capsize=5, color=colors,
        edgecolor="black", linewidth=0.8, error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    ax2.set_title("Final Fairness (1 / std of client accuracies)")
    ax2.set_ylabel("Fairness")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=20, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)

    fig.suptitle("Comparison of Client Selection Methods (mean ± std, n=3 runs)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, (ax1, ax2)


# methods = ["NSGA2", "Full Selection", "Random 75%", "Random 50%", "Random 25%"]
# acc_means = [0.91, 0.89, 0.87, 0.85, 0.80]
# acc_stds = [0.01, 0.015, 0.02, 0.02, 0.03]
# fairness_means = [12.5, 10.2, 9.1, 7.8, 5.4]
# fairness_stds = [0.8, 1.0, 1.2, 1.5, 1.8]

# plot_accuracy_fairness(methods, acc_means, acc_stds, fairness_means, fairness_stds)

def plot_accuracy_fairness_worst(
    methods,
    acc_means,
    acc_stds,
    worst_acc_means,
    worst_acc_stds,
    fairness_means,
    fairness_stds,
    save_path=None,
):
    """
    Plot final global test accuracy, worst client accuracy, and fairness
    as side-by-side bar charts with error bars (mean ± std over 3 runs).

    Parameters
    ----------
    methods : list[str]
        Method names for the x-axis, e.g.
        ["NSGA2", "Full Selection", "Random 75%", "Random 50%", "Random 25%"].
    acc_means, acc_stds : list[float]
        Mean and std of final global test accuracy (0-1) per method.
    worst_acc_means, worst_acc_stds : list[float]
        Mean and std of worst client accuracy (0-1) per method.
    fairness_means, fairness_stds : list[float]
        Mean and std of the fairness metric (1 / std of client accuracies) per method.
    save_path : str, optional
        If given, saves the figure to this path (e.g. "results.png" or "results.pdf").
    """
    colors = plt.get_cmap("tab10").colors[: len(methods)]
    x = np.arange(len(methods))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    error_kw = {"elinewidth": 1.2, "capthick": 1.2}

    # (a) Global test accuracy
    ax1.bar(
        x,
        acc_means,
        yerr=acc_stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        error_kw=error_kw,
    )
    ax1.set_title("Final Global Test Accuracy", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=20, ha="right")
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)

    # (b) Worst client accuracy
    ax2.bar(
        x,
        worst_acc_means,
        yerr=worst_acc_stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        error_kw=error_kw,
    )
    ax2.set_title("Worst Client Accuracy", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=20, ha="right")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)

    # (c) Fairness metric
    ax3.bar(
        x,
        fairness_means,
        yerr=fairness_stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        error_kw=error_kw,
    )
    ax3.set_title("Final Fairness ($1 / \\sigma_{\\text{acc}}$)", fontweight="bold", fontsize=11)
    ax3.set_ylabel("Fairness", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=20, ha="right")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.set_axisbelow(True)

    fig.suptitle("Comparison of Client Selection Methods (mean ± std, n=3 runs)", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, (ax1, ax2, ax3)


import numpy as np
import matplotlib.pyplot as plt

def plot_selection_participation(
    methods,
    selected_means, selected_stds,
    participating_means, participating_stds,
    save_path=None
):
    plt.close("all")

    x = np.arange(len(methods))
    width = 0.35
    colors = plt.cm.tab10.colors

    error_kw = dict(elinewidth=1.2, capthick=1.2, capsize=5)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars1 = ax.bar(
        x - width / 2, selected_means, width,
        yerr=selected_stds,
        label="Selected clients",
        color=colors[0], edgecolor="black",
        error_kw=error_kw
    )
    bars2 = ax.bar(
        x + width / 2, participating_means, width,
        yerr=participating_stds,
        label="Participating clients",
        color=colors[1], edgecolor="black",
        error_kw=error_kw
    )

    ax.set_xlabel("Client Selection Strategy")
    ax.set_ylabel("Average Number of Clients per Round")
    ax.set_title("Average Selected vs. Participating Clients per Round")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend()

    for bars in (bars1, bars2):
        for b in bars:
            height = b.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(b.get_x() + b.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8
            )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax


# --- Example usage ---
# methods = ["NSGA-II", "Full Participation", "Random 75%", "Random 50%", "Random 25%"]

# selected_means      = [7.4, 10.0, 7.5, 5.0, 2.5]
# selected_stds       = [0.5, 0.0,  0.3, 0.2, 0.1]

# participating_means = [6.1, 8.7,  6.9, 4.6, 2.3]
# participating_stds  = [0.6, 0.4,  0.4, 0.3, 0.2]

# plot_selection_participation(
#     methods,
#     selected_means, selected_stds,
#     participating_means, participating_stds,
#     save_path="fig2_selection_participation.png"
# )


def compute_client_participation_frequencies(run_dicts, num_clients=30):
    """
    run_dicts: list of per-run round-dicts for one method,
               e.g. [run1_dict, run2_dict, run3_dict]
               each run_dict is {"1": {...}, "2": {...}, ...}
    Returns: array of shape (num_clients,) — participation frequency
             per client, averaged across runs.
    """
    n_runs = len(run_dicts)
    total_rounds = len(run_dicts[0])  # e.g. 10

    freq_per_run = np.zeros((n_runs, num_clients))

    for r_idx, round_dict in enumerate(run_dicts):
        for round_key, round_info in round_dict.items():
            for cid in round_info["participating_clients"]:
                freq_per_run[r_idx, cid] += 1

    freq_per_run /= total_rounds          # per-run frequency in [0, 1]
    return freq_per_run.mean(axis=0)      # average across the 3 runs -> (num_clients,)

def build_participation_frequencies(results_by_method, num_clients=30):
    """
    results_by_method: dict mapping method name -> list of 3 run-dicts
    Returns: methods list, participation_frequencies list (for the box plot)
    """
    methods = list(results_by_method.keys())
    participation_frequencies = [
        compute_client_participation_frequencies(results_by_method[m], num_clients)
        for m in methods
    ]
    return methods, participation_frequencies

def plot_participation_frequency_boxplot(
    methods,
    participation_frequencies,
    save_path=None
):
    """
    methods: list of method name strings
    participation_frequencies: list of np.ndarrays, one per method,
                               each of shape (num_clients,) with values in [0, 1]
    save_path: optional file path to save the figure
    Returns: (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors

    bp = ax.boxplot(
        participation_frequencies,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(
            marker="o",
            markersize=4,
            linestyle="none",
            markeredgewidth=0.8
        ),
        boxprops=dict(linewidth=1.2),
    )

    for patch, color in zip(bp["boxes"], colors[: len(methods)]):
        patch.set_facecolor(color)
        patch.set_alpha(linewidth=1.2),
    

    for patch, color in zip(bp["boxes"], colors[: len(methods)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Match outlier colors to their box)

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Participation Frequency", fontsize=12)
    ax.set_title("Client Participation Frequency Distribution per Method", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax




import json
from pathlib import Path
from typing import Dict, Any


def remove_warm_up_from_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file and remove the 'warm_up' key value pair.
    
    This function reads federated learning results from a JSON file and removes
    the warm_up key which contains initialization round data, returning only
    the regular round data.
    
    Args:
        filepath: Path to the JSON file containing results
        
    Returns:
        Dictionary with 'warm_up' key removed, containing only round data
        
    Example:
        >>> result = remove_warm_up_from_results(
        ...     'version 7 - alpha0.4 fashion mnist\\50p_random_alpha0.4_runseed1_dataseed1_logs.json'
        ... )
        >>> isinstance(result, dict)
        True
        >>> 'warm_up' not in result
        True
        >>> len(result) > 0  # Should have multiple rounds (0, 1, 2, etc.)
        True
    """
    # Load results from JSON file using existing pattern from _visuals.py
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Remove the 'warm_up' key if it exists
    if 'warm_up' in data:
        # Create a new dictionary excluding 'warm_up'
        # This follows the same logic as _extract_rounds_data() in FederatedLearningVisualizer
        data = {k: v for k, v in data.items() if k != 'warm_up'}
    
    return data


def save_cleaned_results(original_filepath: str, output_filepath: str = None) -> None:
    """
    Load results, remove warm_up, and save to a new file.
    
    Args:
        original_filepath: Path to the original JSON file
        output_filepath: Path for the cleaned output (default: original path with _cleaned suffix)
    """
    cleaned_data = remove_warm_up_from_results(original_filepath)
    
    if output_filepath is None:
        # Create output filename by adding _cleaned before .json
        original_path = Path(original_filepath)
        output_filepath = str(original_path.with_name(f"{original_path.stem}_cleaned{original_path.suffix}"))
    
    with open(output_filepath, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f"Cleaned data saved to: {output_filepath}")


def get_round_numbers(cleaned_results: Dict[str, Any]) -> list:
    """
    Extract round numbers from cleaned results for validation.
    
    Args:
        cleaned_results: Dictionary returned from remove_warm_up_from_results
        
    Returns:
        List of round numbers (as strings)
    """
    return [key for key in cleaned_results.keys() if key.isdigit()]


if __name__ == "__main__":
    # Example usage and testing
    test_file = "version 7 - alpha0.4 fashion mnist\\50p_random_alpha0.4_runseed1_dataseed1_logs.json"
    
    print("=== Testing remove_warm_up_from_results function ===")
    try:
        # Test the main function
        result = remove_warm_up_from_results(test_file)
        
        print(f"✓ Successfully loaded and cleaned results")
        print(f"✓ Result type: {type(result)}")
        print(f"✓ 'warm_up' key present: {'warm_up' in result}")
        print(f"✓ Number of rounds (excluding warm_up): {len(result)}")
        
        # Show round numbers
        rounds = get_round_numbers(result)
        print(f"✓ Round numbers: {sorted(rounds)[:5]}{'...' if len(rounds) > 5 else ''} (showing first 5)")
        
        # Show first round data sample
        if rounds:
            first_round = result[rounds[0]]
            print(f"✓ First round ({rounds[0]}) keys: {list(first_round.keys())}")
        
        # Save cleaned results
        save_cleaned_results(test_file)
        
        print("\n=== All tests passed! ===")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
