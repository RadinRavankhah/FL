import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FederatedLearningVisualizer:
    """A class for visualizing federated learning simulation results."""
    
    def __init__(self, data: Dict, output_dir: str = "visualizations"):
        """
        Initialize the visualizer with simulation data.
        
        Args:
            data: Dictionary containing simulation results
            output_dir: Directory to save visualization images
        """
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Extract round data
        self.rounds_data = self._extract_rounds_data()
        self.round_numbers = sorted([int(r) for r in self.rounds_data.keys()])
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def _extract_rounds_data(self) -> Dict:
        """Extract only regular rounds (exclude warm_up)."""
        return {k: v for k, v in self.data.items() if k != 'warm_up'}
    
    def _get_metric_per_round(self, metric: str) -> List[float]:
        """Extract a specific metric for all rounds."""
        return [self.rounds_data[str(r)][metric] for r in self.round_numbers]
    
    def _save_plot(self, fig: plt.Figure, filename: str):
        """Save plot to output directory and display it."""
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {filepath}")
        plt.show()
        plt.close(fig)
    
    def plot_global_test_accuracy(self):
        """Plot global test accuracy per round."""
        accuracies = self._get_metric_per_round('global_test_accuracy')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.round_numbers, accuracies, 'o-', linewidth=2, markersize=8, 
                color='#2E86AB', label='Global Test Accuracy')
        
        # Add value annotations
        for x, y in zip(self.round_numbers, accuracies):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=9)
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Global Test Accuracy per Round', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.fill_between(self.round_numbers, accuracies, alpha=0.3, color='#2E86AB')
        
        self._save_plot(fig, 'global_test_accuracy.png')
    
    def plot_fairness(self):
        """Plot fairness per round (1/std, higher is better)."""
        fairness_values = self._get_metric_per_round('fairness')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.round_numbers, fairness_values, 's-', linewidth=2, markersize=8, 
                color='#A23B72', label='Fairness')
        
        # Add value annotations
        for x, y in zip(self.round_numbers, fairness_values):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=9)
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Fairness (1/Std Dev)', fontsize=12)
        ax.set_title('Fairness per Round (Higher is Better)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.fill_between(self.round_numbers, fairness_values, alpha=0.3, color='#A23B72')
        
        self._save_plot(fig, 'fairness_per_round.png')
    
    def plot_worst_client_accuracy(self):
        """Plot worst-client accuracy per round."""
        worst_accuracies = self._get_metric_per_round('worst_client_accuracy')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.round_numbers, worst_accuracies, 'D-', linewidth=2, markersize=8, 
                color='#F18F01', label='Worst Client Accuracy')
        
        # Add value annotations
        for x, y in zip(self.round_numbers, worst_accuracies):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=9)
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Worst-Client Accuracy per Round', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.fill_between(self.round_numbers, worst_accuracies, alpha=0.3, color='#F18F01')
        
        self._save_plot(fig, 'worst_client_accuracy.png')
    
    def plot_participation(self):
        """Plot number of selected vs participating clients per round."""
        selected = self._get_metric_per_round('num_selected_clients')
        participating = self._get_metric_per_round('num_participating_clients')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        x_pos = np.arange(len(self.round_numbers))
        
        bars1 = ax.bar(x_pos - width/2, selected, width, label='Selected', 
                       color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, participating, width, label='Participating', 
                       color='#A23B72', alpha=0.8)
        
        # Add value annotations
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Number of Clients', fontsize=12)
        ax.set_title('Client Selection and Participation per Round', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.round_numbers)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        self._save_plot(fig, 'participation_per_round.png')
    
    def plot_all_metrics(self):
        """Generate all plots."""
        logger.info("Generating all visualization plots...")
        
        self.plot_global_test_accuracy()
        self.plot_fairness()
        self.plot_worst_client_accuracy()
        self.plot_participation()
        
        logger.info(f"All plots saved to {self.output_dir}")
    
    def create_results_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with all metrics per round."""
        df = pd.DataFrame({
            'Round': self.round_numbers,
            'Global_Accuracy': self._get_metric_per_round('global_test_accuracy'),
            'Mean_Client_Accuracy': self._get_metric_per_round('mean_client_accuracy'),
            'Fairness': self._get_metric_per_round('fairness'),
            'Worst_Client_Accuracy': self._get_metric_per_round('worst_client_accuracy'),
            'Num_Selected': self._get_metric_per_round('num_selected_clients'),
            'Num_Participating': self._get_metric_per_round('num_participating_clients'),
        })
        
        # Add participation rate
        df['Participation_Rate'] = (df['Num_Participating'] / df['Num_Selected'] * 100).round(2)
        
        return df
    
    def print_summary_statistics(self):
        """Print summary statistics of the results."""
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
        
        # Calculate improvements
        print("\n\nImprovements (Round 1 to Final Round):")
        print("-"*80)
        improvements = {
            'Global Accuracy': df['Global_Accuracy'].iloc[-1] - df['Global_Accuracy'].iloc[0],
            'Fairness': df['Fairness'].iloc[-1] - df['Fairness'].iloc[0],
            'Worst-Client Accuracy': df['Worst_Client_Accuracy'].iloc[-1] - df['Worst_Client_Accuracy'].iloc[0],
        }
        
        for metric, value in improvements.items():
            print(f"  {metric:30s}: {value:+.4f}")


def load_results(filepath: str) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def main(json_filepath: str, output_dir: str = "visualizations"):
    """
    Main function to generate all visualizations.
    
    Args:
        json_filepath: Path to JSON file containing results
        output_dir: Directory to save visualization images
    """
    # Load data
    logger.info(f"Loading results from {json_filepath}")
    data = load_results(json_filepath)
    
    # Create visualizer
    visualizer = FederatedLearningVisualizer(data, output_dir)
    
    # Generate all plots
    visualizer.plot_all_metrics()
    
    # Print summary statistics
    visualizer.print_summary_statistics()


if __name__ == "__main__":
    # Usage example
    # main("version 7 - alpha0.1 runseed3 dataseed3 fashion mnist\\50p_random_alpha0.1_runseed3_dataseed3_logs.json", "visualizations")
    main("version 7 - alpha0.1 runseed3 dataseed3 fashion mnist\\nsga2_weights70_20_10_alpha0.1_runseed3_dataseed3_logs.json", "visualizations")