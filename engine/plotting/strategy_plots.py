import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

class StrategyPlotter:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital

    def plot_ml_strategies_by_threshold(self, ml_results: Dict, dates: np.ndarray, output_dir: str) -> None:
        """Plot ML strategies grouped by threshold with different colors."""
        if not ml_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Group by strategy name (ensemble_smart, momentum, mean_reversion, volatility_weighted)
        strategies = {}
        for name, data in ml_results.items():
            parts = name.split()  # "ML strategy_name threshold"
            strategy_name = f"{parts[1]} {parts[2]}"  # e.g., "ensemble_smart 0.50"
            base_strategy = parts[1]  # e.g., "ensemble_smart"
            if base_strategy not in strategies:
                strategies[base_strategy] = {}
            strategies[base_strategy][name] = data
        
        # Extended color palette for more thresholds
        colors_by_threshold = {
            "0.50": "darkblue",
            "0.51": "blue",
            "0.52": "cornflowerblue",
            "0.53": "green",
            "0.54": "lightgreen",
            "0.55": "orange",
            "0.56": "darkorange",
            "0.57": "red",
            "0.58": "darkred",
            "0.59": "purple",
            "0.60": "magenta"
        }
        
        # Plot each base strategy with different thresholds
        for idx, (base_strategy, strategy_results) in enumerate(strategies.items()):
            ax = axes[idx]
            
            for full_name, data in sorted(strategy_results.items()):
                equity_curve = data["equity_curve"]
                normalized = equity_curve / self.initial_capital
                threshold = full_name.split()[-1]  # Extract threshold
                color = colors_by_threshold.get(threshold, "gray")
                
                ax.plot(dates, normalized, label=f"Threshold {threshold}", linewidth=2.5, 
                       color=color, alpha=0.8)
            
            ax.set_title(f"ML Strategy: {base_strategy.replace('_', ' ').title()}", 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel("Date")
            ax.set_ylabel("Normalized Equity")
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ml_strategies_by_threshold.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_all_strategies_comparison(self, results: Dict, dates: np.ndarray, output_dir: str) -> None:
        """Plot all strategies together for overall comparison."""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Select BEST threshold for each ML strategy, and include all benchmarks
        selected = {}
        ml_by_strategy = {}
        threshold_only_dict = {}
        
        # Group ML strategies by name (different thresholds)
        for name, data in results.items():
            if name.startswith("ML "):
                parts = name.split()
                strategy_name = " ".join(parts[1:-1])  # "ensemble_smart", "momentum", etc.
                threshold = float(parts[-1])
                
                if strategy_name not in ml_by_strategy:
                    ml_by_strategy[strategy_name] = {}
                ml_by_strategy[strategy_name][threshold] = (name, data)
            elif name.startswith("Threshold Only "):
                # Collect all Threshold Only variants
                threshold = float(name.split()[-1])
                threshold_only_dict[threshold] = (name, data)
            else:
                # Include benchmarks only (not baselines)
                if name not in ["Threshold Only (baseline)"]:
                    selected[name] = data
        
        # For each strategy, select the threshold with highest return
        for strategy_name, threshold_dict in ml_by_strategy.items():
            best_threshold = max(threshold_dict.items(), 
                                key=lambda x: x[1][1].get("total_return", 0))
            best_name, best_data = best_threshold[1]
            short_name = strategy_name.replace("_", " ").title()
            selected[short_name] = best_data
        
        # Select best Threshold Only
        if threshold_only_dict:
            best_threshold_only = max(threshold_only_dict.items(),
                                     key=lambda x: x[1][1].get("total_return", 0))
            best_name, best_data = best_threshold_only[1]
            selected["Threshold Only (baseline)"] = best_data
        
        # Plot with different colors
        color_map = {
            "Ensemble Smart": "darkblue",
            "Momentum": "green",
            "Mean Reversion": "orange",
            "Volatility Weighted": "red",
            "Threshold Only (baseline)": "gray",
            "Buy & Hold (benchmark)": "purple",
            "Fixed Income 5.00% (benchmark)": "brown",
            "Simple Reversal SMA20 (benchmark)": "pink"
        }
        
        for strategy_name, data in selected.items():
            equity_curve = data["equity_curve"]
            normalized = equity_curve / self.initial_capital
            color = color_map.get(strategy_name, "black")
            
            ax.plot(dates, normalized, label=strategy_name, linewidth=2.5, 
                   color=color, alpha=0.8)
        
        ax.set_title("All Strategies Comparison (Best Threshold Configuration)", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Equity")
        ax.legend(loc='best', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/all_strategies_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_strategy_metrics_comparison(self, results: Dict, output_dir: str) -> None:
        """Plot metrics comparison for best threshold of each strategy only."""
        # Select best threshold for each strategy
        selected = {}
        ml_by_strategy = {}
        threshold_only_dict = {}
        
        # Group ML strategies by name
        for name, data in results.items():
            if name.startswith("ML "):
                parts = name.split()
                strategy_name = " ".join(parts[1:-1])
                threshold = float(parts[-1])
                
                if strategy_name not in ml_by_strategy:
                    ml_by_strategy[strategy_name] = {}
                ml_by_strategy[strategy_name][threshold] = (name, data)
            elif name.startswith("Threshold Only "):
                # Collect Threshold Only variants
                threshold = float(name.split()[-1])
                threshold_only_dict[threshold] = (name, data)
            else:
                # Include benchmarks only
                selected[name] = data
        
        # For each strategy, select the threshold with highest return
        for strategy_name, threshold_dict in ml_by_strategy.items():
            best_threshold = max(threshold_dict.items(), 
                                key=lambda x: x[1][1].get("total_return", 0))
            best_name, best_data = best_threshold[1]
            short_name = strategy_name.replace("_", " ").title()
            selected[short_name] = best_data
        
        # Select best Threshold Only
        if threshold_only_dict:
            best_threshold_only = max(threshold_only_dict.items(),
                                     key=lambda x: x[1][1].get("total_return", 0))
            best_name, best_data = best_threshold_only[1]
            selected["Threshold Only (baseline)"] = best_data
        
        metrics = ["annualized_return", "sharpe_ratio", "max_drawdown", "active_hit_rate"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [selected[s].get(metric, 0) for s in selected.keys()]
            colors = ["green" if v > 0 else "red" for v in values]
            
            bars = ax.bar(range(len(selected)), values, color=colors, alpha=0.7)
            ax.set_xticks(range(len(selected)))
            ax.set_xticklabels(selected.keys(), rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').title()} Comparison", fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_drawdown_analysis(self, results: Dict, dates: np.ndarray, output_dir: str) -> None:
        """Plot maximum drawdown for each strategy."""
        # Select best threshold for each strategy
        selected = {}
        ml_by_strategy = {}
        threshold_only_dict = {}
        
        # Group ML strategies by name
        for name, data in results.items():
            if name.startswith("ML "):
                parts = name.split()
                strategy_name = " ".join(parts[1:-1])
                threshold = float(parts[-1])
                
                if strategy_name not in ml_by_strategy:
                    ml_by_strategy[strategy_name] = {}
                ml_by_strategy[strategy_name][threshold] = (name, data)
            elif name.startswith("Threshold Only "):
                threshold = float(name.split()[-1])
                threshold_only_dict[threshold] = (name, data)
            else:
                selected[name] = data
        
        # For each ML strategy, select the threshold with highest return
        for strategy_name, threshold_dict in ml_by_strategy.items():
            best_threshold = max(threshold_dict.items(), 
                                key=lambda x: x[1][1].get("total_return", 0))
            best_name, best_data = best_threshold[1]
            short_name = strategy_name.replace("_", " ").title()
            selected[short_name] = best_data
        
        # Select best Threshold Only
        if threshold_only_dict:
            best_threshold_only = max(threshold_only_dict.items(),
                                     key=lambda x: x[1][1].get("total_return", 0))
            best_name, best_data = best_threshold_only[1]
            selected["Threshold Only (baseline)"] = best_data
        
        # Prepare data for plotting
        max_drawdowns = []
        strategy_names = []
        for name, data in selected.items():
            strategy_names.append(name)
            max_drawdowns.append(data.get("max_drawdown", 0) * 100)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ["red" if x < 0 else "green" for x in max_drawdowns]
        bars = ax.barh(strategy_names, max_drawdowns, color=colors, alpha=0.7)
        
        ax.set_xlabel("Maximum Drawdown (%)")
        ax.set_title("Maximum Drawdown by Strategy", fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2., f'{width:.1f}%',
                   ha='left' if width > 0 else 'right', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/max_drawdown.png", dpi=150, bbox_inches='tight')
        plt.close()
    