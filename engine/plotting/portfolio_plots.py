import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict

class PortfolioPlotter:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital

    def plot_stock_selection_summary(self, backtest_results: Dict, test_df: pd.DataFrame, output_dir: str) -> None:
        """Plot summary of stock selection frequency and data availability.
        
        Args:
            backtest_results: Backtest results containing position data
            test_df: Test dataframe with ticker column (used for data availability stats)
            output_dir: Output directory for the plot
        """
        try:
            # Find the best ML strategy
            best_strategy_key = None
            best_return = -np.inf
            
            for strategy_name, data in backtest_results.items():
                if strategy_name.startswith("ML "):
                    ret = data.get("total_return", 0)
                    if ret > best_return:
                        best_return = ret
                        best_strategy_key = strategy_name
            
            if best_strategy_key is None or "positions_by_ticker" not in backtest_results[best_strategy_key]:
                return
            
            positions_by_ticker = backtest_results[best_strategy_key]["positions_by_ticker"]
            
            # Use selection_rate directly from positions_by_ticker (works even without test_df)
            all_tickers = sorted(positions_by_ticker.keys())
            
            # Get stats
            selection_rates = {}
            for ticker in all_tickers:
                data = positions_by_ticker[ticker]
                selection_rates[ticker] = data.get("selection_rate", 0) * 100
                
            # Get data availability from test_df if available
            try:
                ticker_counts = test_df['ticker'].value_counts()
                data_avail = [ticker_counts.get(t, 0) for t in all_tickers]
            except:
                # Fallback: estimate from days_selected and selection_rate
                data_avail = [int(positions_by_ticker[t].get("total_days", 1)) for t in all_tickers]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Data availability vs Selection frequency
            select_freq = [selection_rates.get(t, 0) for t in all_tickers]
            
            x = np.arange(len(all_tickers))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, data_avail, width, label='Data Points', color='steelblue', alpha=0.8)
            ax1_twin = ax1.twinx()
            bars2 = ax1_twin.bar(x + width/2, select_freq, width, label='Selected %', color='darkgreen', alpha=0.8)
            
            ax1.set_xlabel("Ticker", fontsize=11, fontweight='bold')
            ax1.set_ylabel("Number of Daily Observations", fontsize=10, color='steelblue', fontweight='bold')
            ax1_twin.set_ylabel("Selection Frequency (%)", fontsize=10, color='darkgreen', fontweight='bold')
            ax1.set_title(f"Data Availability vs Selection Frequency\nStrategy: {best_strategy_key}", 
                         fontweight='bold', fontsize=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(all_tickers)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
            
            # Plot 2: Summary statistics from positions_by_ticker
            n_stocks = len(all_tickers)
            
            # Calculate test period info from positions_by_ticker
            total_days = sum(positions_by_ticker[t].get("total_days", 0) for t in all_tickers) // n_stocks
            
            summary_text = f"""
                Test Set Summary
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                Total Trading Days:        {total_days}
                Total Unique Stocks:       {n_stocks}
                Strategy:                  {best_strategy_key}

                Selection Frequency by Stock:
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join([f"  {ticker:6s}: {selection_rates.get(ticker, 0):5.1f}% selected ({int(positions_by_ticker[ticker].get('days_selected', 0)):4d} days)" 
              for ticker in all_tickers])}

                Key Metrics:
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                → Selection Rate: How often the model
                  chose to hold each stock during
                  the test period
                  
                → Shows model's preference for
                  different stocks based on signals
            """
            
            ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, 
                    fontfamily='monospace', fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/stock_selection_summary.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            # Log error for debugging
            print(f"Warning: Could not generate stock_selection_summary: {e}")
            pass
    
    def plot_portfolio_allocation_summary(self, backtest_results: Dict, test_df: pd.DataFrame, output_dir: str) -> None:
        """Plot portfolio allocation: FINAL WEIGHTS chosen by the best ML strategy.
        
        Shows the FINAL position weights (last day of backtest) to use in production.
        These are the weights you should use tomorrow to invest.
        
        Args:
            backtest_results: Backtest results with position data
            test_df: Test dataframe with ticker column
            output_dir: Output directory for the plot
        """
        try:
            # Find the best ML strategy result (highest return)
            best_strategy_key = None
            best_return = -np.inf
            
            for strategy_name, data in backtest_results.items():
                if strategy_name.startswith("ML "):
                    ret = data.get("total_return", 0)
                    if ret > best_return:
                        best_return = ret
                        best_strategy_key = strategy_name
            
            if best_strategy_key is None:
                return
            
            if "positions_by_ticker" not in backtest_results[best_strategy_key]:
                return
            
            positions_by_ticker = backtest_results[best_strategy_key]["positions_by_ticker"]
            
            # Prepare data - USE FINAL WEIGHTS
            tickers = sorted(positions_by_ticker.keys())
            final_weights = [positions_by_ticker[t]["final_weight"] * 100 for t in tickers]  # Convert to %
            avg_weights = [positions_by_ticker[t]["avg_position_weight"] * 100 for t in tickers]
            selection_rates = [positions_by_ticker[t]["selection_rate"] * 100 for t in tickers]
            
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # ===== TOP: FINAL WEIGHTS (What to use tomorrow) =====
            ax_final = fig.add_subplot(gs[0, :])
            colors_final = ['darkgreen' if w > 0 else 'lightgray' for w in final_weights]
            bars = ax_final.barh(tickers, final_weights, color=colors_final, alpha=0.85, edgecolor='black', linewidth=2)
            
            ax_final.set_xlabel("Position Weight (%)", fontsize=12, fontweight='bold')
            ax_final.set_title(f"PRODUCTION WEIGHTS - Use These Tomorrow!\nStrategy: {best_strategy_key}\nLast Day Weight Allocation", 
                              fontweight='bold', fontsize=14, color='darkgreen')
            ax_final.set_xlim(0, max(final_weights) * 1.15 if final_weights else 10)
            ax_final.grid(True, alpha=0.3, axis='x')
            
            # Add weight labels with % and explanation
            for i, (ticker, weight) in enumerate(zip(tickers, final_weights)):
                if weight > 0:
                    ax_final.text(weight + 0.2, i, f'{weight:.1f}%  ← Invest this amount', 
                                 va='center', fontsize=10, fontweight='bold')
            
            # ===== BOTTOM LEFT: Historical Average Weights =====
            ax_avg = fig.add_subplot(gs[1, 0])
            colors_avg = ['steelblue' if w > 5 else 'orange' if w > 1 else 'lightgray' for w in avg_weights]
            bars = ax_avg.barh(tickers, avg_weights, color=colors_avg, alpha=0.8, edgecolor='black', linewidth=1)
            
            ax_avg.set_xlabel("Average Weight (%)", fontsize=11, fontweight='bold')
            ax_avg.set_title("Historical Average Weights\n(What the model preferred historically)", 
                            fontweight='bold', fontsize=12)
            ax_avg.set_xlim(0, max(avg_weights) * 1.1 if avg_weights else 10)
            ax_avg.grid(True, alpha=0.3, axis='x')
            
            # Add average weight labels
            for i, (ticker, weight) in enumerate(zip(tickers, avg_weights)):
                ax_avg.text(weight + 0.1, i, f'{weight:.2f}%', va='center', fontsize=9, fontweight='bold')
            
            # ===== BOTTOM RIGHT: Selection Frequency =====
            ax_freq = fig.add_subplot(gs[1, 1])
            colors_freq = ['darkgreen' if r > 50 else 'orange' if r > 10 else 'red' for r in selection_rates]
            bars = ax_freq.barh(tickers, selection_rates, color=colors_freq, alpha=0.8, edgecolor='black', linewidth=1)
            
            ax_freq.set_xlabel("Selected Days (%)", fontsize=11, fontweight='bold')
            ax_freq.set_title("Selection Frequency\n(How often was this stock chosen historically)", 
                            fontweight='bold', fontsize=12)
            ax_freq.set_xlim(0, 105)
            ax_freq.grid(True, alpha=0.3, axis='x')
            
            # Add frequency labels
            for i, (ticker, freq) in enumerate(zip(tickers, selection_rates)):
                days = int(positions_by_ticker[ticker]["days_selected"])
                ax_freq.text(freq + 1, i, f'{freq:.0f}% ({days}d)', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/portfolio_allocation_summary.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        except Exception as e:
            # Silently skip if error
            print(f"Warning: Could not generate portfolio allocation summary: {e}")
