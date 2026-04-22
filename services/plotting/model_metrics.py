import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

class ModelMetricsPlotter:
    def plot_model_metrics_comparison(self, model_results: Dict[str, Dict], output_dir: str) -> None:
        """Plot and save model comparison visualization with multiple metrics.
        
        Args:
            model_results: Dictionary mapping model names to their WFV metrics
            output_dir: Output directory for the plot
        """
        try:
            # Extract metrics
            models = list(model_results.keys())
            mean_ics = [model_results[m]['mean_ic'] for m in models]
            std_ics = [model_results[m]['std_ic'] for m in models]
            scores = [model_results[m]['wfv_score'] for m in models]
            accuracies = [model_results[m].get('accuracy', 0) for m in models]
            f1_scores = [model_results[m].get('f1_score', 0) for m in models]
            aucs = [model_results[m].get('auc', 0.5) for m in models]
            
            # Create 3x2 subplots
            fig, axes = plt.subplots(3, 2, figsize=(14, 14))
            
            # Plot 1: Mean IC
            ax = axes[0, 0]
            colors = ['green' if x > 0 else 'red' for x in mean_ics]
            ax.bar(models, mean_ics, color=colors, alpha=0.7)
            ax.set_title("Mean Information Coefficient (IC)", fontweight='bold', fontsize=12)
            ax.set_ylabel("Mean IC")
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(mean_ics):
                ax.text(i, v, f'{v:.4f}', ha='center', va='bottom' if v > 0 else 'top', fontsize=9, fontweight='bold')
            
            # Plot 2: Std IC (lower is more stable)
            ax = axes[0, 1]
            colors = ['green' if x < 0.035 else 'orange' for x in std_ics]
            ax.bar(models, std_ics, color=colors, alpha=0.7)
            ax.set_title("Std Dev of IC (Lower = More Stable)", fontweight='bold', fontsize=12)
            ax.set_ylabel("Std IC")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(std_ics):
                ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Plot 3: WFV Score (Robust Score)
            ax = axes[1, 0]
            colors_score = ['green' if x > 0 else 'red' for x in scores]
            bars = ax.bar(models, scores, color=colors_score, alpha=0.7, edgecolor='black', linewidth=2)
            best_idx = np.argmax(scores)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
            ax.set_title("Walk-Forward Validation Score\n(Mean IC - 0.5×Std IC) [SELECTION METRIC]", fontweight='bold', fontsize=12)
            ax.set_ylabel("WFV Score")
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(scores):
                ax.text(i, v, f'{v:.4f}', ha='center', va='bottom' if v > 0 else 'top', fontsize=9, fontweight='bold')
            
            # Plot 4: Accuracy
            ax = axes[1, 1]
            bars = ax.bar(models, accuracies, color='steelblue', alpha=0.7)
            ax.set_title("Classification Accuracy", fontweight='bold', fontsize=12)
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(accuracies):
                ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Plot 5: F1 Score
            ax = axes[2, 0]
            bars = ax.bar(models, f1_scores, color='coral', alpha=0.7)
            ax.set_title("F1 Score", fontweight='bold', fontsize=12)
            ax.set_ylabel("F1 Score")
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(f1_scores):
                ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Plot 6: AUC (ROC)
            ax = axes[2, 1]
            bars = ax.bar(models, aucs, color='mediumseagreen', alpha=0.7)
            ax.set_title("ROC-AUC Score", fontweight='bold', fontsize=12)
            ax.set_ylabel("AUC")
            ax.set_ylim(0.4, 1.0)
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Random')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
            for i, v in enumerate(aucs):
                ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            filepath = f"{output_dir}/model_selection_comparison.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            # Log error silently (don't use logger here to avoid circular imports)
            print(f"Warning: Failed to plot model comparison: {e}")
            raise
    