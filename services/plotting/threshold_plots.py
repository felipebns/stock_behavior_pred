import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Dict

class ThresholdPlotter:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital

    def plot_threshold_effect(self, ml_results: Dict, output_dir: str) -> None:
        """Plot how threshold affects returns for each ML strategy."""
        if not ml_results:
            return
        
        # Group by base strategy
        strategies = {}
        for name, data in ml_results.items():
            parts = name.split()
            base_strategy = parts[1]
            threshold = float(parts[2])
            
            if base_strategy not in strategies:
                strategies[base_strategy] = {"thresholds": [], "returns": [], "sharpe": []}
            
            strategies[base_strategy]["thresholds"].append(threshold)
            strategies[base_strategy]["returns"].append(data.get("total_return", 0) * 100)
            strategies[base_strategy]["sharpe"].append(data.get("sharpe_ratio", 0))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Extended color palette
        colors = ["darkblue", "cornflowerblue", "green", "orange", "red"]
        
        for color, (strategy_name, data) in zip(colors, strategies.items()):
            thresholds = sorted(zip(data["thresholds"], data["returns"], data["sharpe"]))
            thresholds_sorted = [t[0] for t in thresholds]
            returns_sorted = [t[1] for t in thresholds]
            sharpe_sorted = [t[2] for t in thresholds]
            
            ax1.plot(thresholds_sorted, returns_sorted, marker='o', label=strategy_name.replace("_", " ").title(),
                    color=color, linewidth=2, markersize=6)
            ax2.plot(thresholds_sorted, sharpe_sorted, marker='s', label=strategy_name.replace("_", " ").title(),
                    color=color, linewidth=2, markersize=6)
        
        ax1.set_xlabel("Probability Threshold")
        ax1.set_ylabel("Total Return (%)")
        ax1.set_title("Effect of Threshold on Returns", fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel("Probability Threshold")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_title("Effect of Threshold on Sharpe Ratio", fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/threshold_effect.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_threshold_only_analysis(self, results: Dict, dates: np.ndarray, output_dir: str) -> None:
        """Plot comprehensive analysis of Threshold Only baseline across all thresholds using line charts."""
        # Extract Threshold Only data
        threshold_data = {"thresholds": [], "returns": [], "sharpe": [], "max_dd": [], "equity_curves": []}
        
        for name, data in results.items():
            if name.startswith("Threshold Only "):
                threshold = float(name.split()[-1])
                threshold_data["thresholds"].append(threshold)
                threshold_data["returns"].append(data.get("total_return", 0) * 100)
                threshold_data["sharpe"].append(data.get("sharpe_ratio", 0))
                threshold_data["max_dd"].append(data.get("max_drawdown", 0) * 100)
                threshold_data["equity_curves"].append(data.get("equity_curve", np.ones(len(dates)) * self.initial_capital))
        
        if not threshold_data["thresholds"]:
            return
        
        # Sort by threshold
        sorted_data = sorted(zip(threshold_data["thresholds"], 
                                 threshold_data["returns"],
                                 threshold_data["sharpe"],
                                 threshold_data["equity_curves"]))
        thresholds, returns, sharpe, equity_curves = zip(*sorted_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Total Return
        ax1.plot(thresholds, returns, marker='o', linewidth=2, markersize=8, color='green', label='Total Return')
        ax1.set_xlabel("Probability Threshold", fontweight='bold')
        ax1.set_ylabel("Total Return (%)", fontweight='bold')
        ax1.set_title("Threshold Only: Return Sensitivity", fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 2: Equity Curves
        import matplotlib.cm as cm
        colors = cm.viridis(np.linspace(0, 1, len(thresholds)))
        
        for i, (thresh, eq_curve) in enumerate(zip(thresholds, equity_curves)):
            normalized = eq_curve / self.initial_capital
            ax2.plot(dates, normalized, label=f"Threshold {thresh:.2f}", linewidth=1.5, color=colors[i], alpha=0.8)
            
        ax2.set_xlabel("Date", fontweight='bold')
        ax2.set_ylabel("Normalized Equity", fontweight='bold')
        ax2.set_title("Threshold Only: Equity Curve per Threshold", fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=0.5)
        ax2.legend(loc='best', fontsize=8, ncol=2)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/threshold_only_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
    