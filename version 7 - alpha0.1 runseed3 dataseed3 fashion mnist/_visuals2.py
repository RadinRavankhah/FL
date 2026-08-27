import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import logging
from collections import Counter
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FederatedLearningVisualizer:
    """A class for visualizing federated learning simulation results using pure Matplotlib."""
    
    def __init__(self, data: Dict, json_filepath: str, output_dir: str = "."):
        """
        Initialize the visualizer with simulation data.
        
        Args:
            data: Dictionary containing simulation results
            json_filepath: The path to the json file used (to extract its name)
            output_dir: Directory to save visualization images (default: current directory)
        """
        self.data = data
        self.base_filename = Path(json_filepath).stem
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract round data
        self.rounds_data = self._extract_rounds_data()
        self.round_numbers = sorted([int(r) for r in self.rounds_data.keys()])
        
        # Define a standard color palette
        self.colors = {
            'blue': '#2E86AB',
            'purple': '#A23B72',
            'orange': '#F18F01',
            'green': '#38B000',
            'red': '#D90429',
            'cyan': '#00B4D8'
        }
        
    def _extract_rounds_data(self) -> Dict:
        """Extract only regular rounds (exclude warm_up)."""
        return {k: v for k, v in self.data.items() if k != 'warm_up'}
    
    def _get_metric_per_round(self, metric: str, default=None) -> List:
        """Extract a specific metric for all rounds, with an optional default if missing."""
        return [self.rounds_data[str(r)].get(metric, default) for r in self.round_numbers]
    
    def _save_plot(self, fig: plt.Figure, title: str):
        """Save plot to output directory using the requested dynamic naming convention."""
        clean_title = re.sub(r'[^a-zA-Z0-9]', '_', title)
        clean_title = re.sub(r'_+', '_', clean_title).strip('_').lower()
        
        filename = f"{self.base_filename}_{clean_title}.png"
        filepath = self.output_dir / filename
        
        fig.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {filepath}")
        
        plt.show()
        plt.close(fig)
    
    def _add_annotations(self, ax, x_data, y_data, format_str='{:.3f}'):
        """Helper to add text annotations to line plots with robust type handling."""
        for x, y in zip(x_data, y_data):
            if y is not None:
                try:
                    # Cast to float to safely handle strings/nulls/NaNs from JSON parsing
                    val = float(y)
                    if not np.isnan(val):
                        ax.annotate(format_str.format(val), (x, val), textcoords="offset points", 
                                   xytext=(0, 10), ha='center', fontsize=9)
                except (ValueError, TypeError):
                    pass
                           
    def _setup_axes(self, ax, title, xlabel, ylabel):
        """Helper to standardize grid, labels, and plot padding."""
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.margins(y=0.15) # Ensures annotations don't hit the top border

    def plot_global_test_accuracy(self):
        title = 'Global Test Accuracy per Round'
        accuracies = self._get_metric_per_round('global_test_accuracy')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.round_numbers, accuracies, 'o-', linewidth=2, markersize=8, 
                color=self.colors['blue'], label='Global Test Accuracy')
        
        self._add_annotations(ax, self.round_numbers, accuracies)
        self._setup_axes(ax, title, 'Round', 'Accuracy')
        ax.fill_between(self.round_numbers, accuracies, alpha=0.2, color=self.colors['blue'])
        
        self._save_plot(fig, title)
        
    def plot_mean_client_accuracy(self):
        title = 'Mean Client Accuracy per Round'
        accuracies = self._get_metric_per_round('mean_client_accuracy')
        
        if accuracies[0] is None:
            logger.warning("mean_client_accuracy missing. Skipping plot.")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.round_numbers, accuracies, 'o-', linewidth=2, markersize=8, 
                color=self.colors['green'], label='Mean Client Accuracy')
        
        self._add_annotations(ax, self.round_numbers, accuracies)
        self._setup_axes(ax, title, 'Round', 'Mean Accuracy')
        ax.fill_between(self.round_numbers, accuracies, alpha=0.2, color=self.colors['green'])
        
        self._save_plot(fig, title)
    
    def plot_fairness(self):
        title = 'Fairness per Round (Higher is Better)'
        fairness_values = self._get_metric_per_round('fairness')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.round_numbers, fairness_values, 's-', linewidth=2, markersize=8, 
                color=self.colors['purple'], label='Fairness (1/Std)')
        
        self._add_annotations(ax, self.round_numbers, fairness_values, '{:.2f}')
        self._setup_axes(ax, title, 'Round', 'Fairness (1/Std Dev)')
        ax.fill_between(self.round_numbers, fairness_values, alpha=0.2, color=self.colors['purple'])
        
        self._save_plot(fig, title)
    
    def plot_worst_client_accuracy(self):
        title = 'Worst Client Accuracy per Round'
        worst_accuracies = self._get_metric_per_round('worst_client_accuracy')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.round_numbers, worst_accuracies, 'D-', linewidth=2, markersize=8, 
                color=self.colors['orange'], label='Worst Client Accuracy')
        
        self._add_annotations(ax, self.round_numbers, worst_accuracies)
        self._setup_axes(ax, title, 'Round', 'Accuracy')
        ax.fill_between(self.round_numbers, worst_accuracies, alpha=0.2, color=self.colors['orange'])
        
        self._save_plot(fig, title)
        
    def plot_average_utility(self):
        title = 'Average Hardware Utility per Round'
        
        # Load exactly what the notebook dictates from the JSON structure
        util_selected = self._get_metric_per_round('average_utility_selected_clients_before_fit')
        if util_selected[0] is None:
            # Fallback for older JSONs
            util_selected = self._get_metric_per_round('average_utility_selected_clients')
            
        util_participating = self._get_metric_per_round('average_utility_participating_clients_before_fit')
        if util_participating[0] is None:
            # Fallback for older JSONs
            util_participating = self._get_metric_per_round('average_utility_participating_clients')
                            
        if util_selected[0] is None and util_participating[0] is None:
            logger.warning("Utility metrics missing. Skipping utility plot.")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if util_selected[0] is not None:
            ax.plot(self.round_numbers, util_selected, 'o-', linewidth=2, markersize=8, 
                    color=self.colors['cyan'], label='Selected Clients (St)')
            self._add_annotations(ax, self.round_numbers, util_selected, '{:.2f}')
            
        if util_participating[0] is not None:
            ax.plot(self.round_numbers, util_participating, 's-', linewidth=2, markersize=8, 
                    color=self.colors['red'], label='Participating Clients (At)')
            self._add_annotations(ax, self.round_numbers, util_participating, '{:.2f}')
            
        self._setup_axes(ax, title, 'Round', 'Average Utility [0, 1]')
        ax.legend(loc='lower right')
        
        self._save_plot(fig, title)
    
    def plot_participation(self):
        title = 'Client Selection vs Participation per Round'
        selected = self._get_metric_per_round('num_selected_clients')
        participating = self._get_metric_per_round('num_participating_clients')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        x_pos = np.arange(len(self.round_numbers))
        
        bars1 = ax.bar(x_pos - width/2, selected, width, label='Selected (St)', 
                       color=self.colors['blue'], alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, participating, width, label='Participating (At)', 
                       color=self.colors['purple'], alpha=0.8)
        
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        self._setup_axes(ax, title, 'Round', 'Number of Clients')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.round_numbers)
        ax.legend(loc='upper right')
        
        self._save_plot(fig, title)
        
    def plot_participation_frequency(self):
        title = 'Device Selection and Participation Frequency'
        selected_counts = Counter()
        participating_counts = Counter()
        
        for r in self.round_numbers:
            round_data = self.rounds_data[str(r)]
            sel_clients = round_data.get('selected_clients', [])
            part_clients = round_data.get('participating_clients', [])
            
            selected_counts.update(sel_clients)
            participating_counts.update(part_clients)
            
        if not selected_counts:
            logger.warning("List of 'selected_clients' missing. Skipping device frequency plot.")
            return
            
        all_devices = sorted(list(set(selected_counts.keys()) | set(participating_counts.keys())))
        
        sel_freq = [selected_counts[d] for d in all_devices]
        part_freq = [participating_counts[d] for d in all_devices]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        width = 0.4
        x_pos = np.arange(len(all_devices))
        
        ax.bar(x_pos - width/2, sel_freq, width, label='Times Selected (Fi)', 
               color=self.colors['orange'], alpha=0.8)
        ax.bar(x_pos + width/2, part_freq, width, label='Times Participated (Pi)', 
               color=self.colors['green'], alpha=0.8)
               
        self._setup_axes(ax, title, 'Device ID', 'Frequency (Rounds)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_devices)
        ax.legend(loc='upper right')
        
        self._save_plot(fig, title)
    
    def plot_all_metrics(self):
        """Generate all plots."""
        logger.info("Generating all visualization plots...")
        
        self.plot_global_test_accuracy()
        self.plot_mean_client_accuracy()
        self.plot_fairness()
        self.plot_worst_client_accuracy()
        self.plot_average_utility()
        self.plot_participation()
        self.plot_participation_frequency()
        
        logger.info(f"All plots saved to {self.output_dir.absolute()}")
    
    def create_results_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'Round': self.round_numbers,
            'Global_Accuracy': self._get_metric_per_round('global_test_accuracy', 0),
            'Mean_Client_Accuracy': self._get_metric_per_round('mean_client_accuracy', 0),
            'Fairness': self._get_metric_per_round('fairness', 0),
            'Worst_Client_Accuracy': self._get_metric_per_round('worst_client_accuracy', 0),
            'Num_Selected': self._get_metric_per_round('num_selected_clients', 0),
            'Num_Participating': self._get_metric_per_round('num_participating_clients', 0),
        })
        
        df['Participation_Rate'] = np.where(df['Num_Selected'] > 0, 
                                            (df['Num_Participating'] / df['Num_Selected'] * 100).round(2), 
                                            0)
        return df
    
    def print_summary_statistics(self):
        df = self.create_results_dataframe()
        
        print("\n" + "="*80)
        print("FEDERATED LEARNING RESULTS SUMMARY")
        print("="*80)
        print("\nResults per Round:")
        print("-"*80)
        print(df.to_string(index=False))
        
        print("\n\nKey Statistics:")
        print("-"*80)
        stats = {
            'Final Global Accuracy': df['Global_Accuracy'].iloc[-1],
            'Best Global Accuracy': df['Global_Accuracy'].max(),
            'Avg Global Accuracy': df['Global_Accuracy'].mean(),
            'Final Fairness': df['Fairness'].iloc[-1],
            'Best Fairness': df['Fairness'].max(),
            'Final Worst-Client Acc': df['Worst_Client_Accuracy'].iloc[-1],
            'Best Worst-Client Acc': df['Worst_Client_Accuracy'].max(),
            'Avg Selected Clients': df['Num_Selected'].mean(),
            'Avg Participating Clients': df['Num_Participating'].mean(),
        }
        
        for metric, value in stats.items():
            print(f"  {metric:30s}: {value:.4f}")

def load_results(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def main(json_filepath: str, output_dir: str = "."):
    """
    Main function to generate all visualizations.
    
    Args:
        json_filepath: Path to JSON file containing results
        output_dir: Directory to save visualization images (default: current directory)
    """
    logger.info(f"Loading results from {json_filepath}")
    
    data = load_results(json_filepath)
    visualizer = FederatedLearningVisualizer(data, json_filepath, output_dir)
    
    visualizer.plot_all_metrics()
    visualizer.print_summary_statistics()

if __name__ == "__main__":
    # Example usage:
    main("version 7 - alpha0.1 runseed3 dataseed3 fashion mnist/75p_random_alpha0.1_runseed3_dataseed3_logs.json")
    pass
