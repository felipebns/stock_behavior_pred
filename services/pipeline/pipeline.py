import pandas as pd
import time
from pathlib import Path
from typing import Tuple, Dict, Any
from services.pipeline.model_selector import ModelSelector
from services.pipeline.metrics_evaluator import MetricsEvaluator
from services.log.reporters import PipelineReporter
from services.backtesting import Backtest
from services.plotting.plot_generator import PlotGenerator
from services.stock.stock import Stock
from services.stock.transform import FeatureEngineer
from services.algorithms.base import Algorithm
from services.log.logger_config import get_logger


class Pipeline:
    """Main ML pipeline orchestrator.
    
    Composes: ModelSelector, MetricsEvaluator, PipelineOrchestrator, Backtest
    Workflow: Data → Features → Model Selection → Backtesting
    """
    
    def __init__(
        self,
        stock: Stock,
        algorithms: list[Algorithm],
        output_dir: str = "output",
        test_size: float = 0.20,
        wfv_train_window: int = 750,
        wfv_test_window: int = 250,
        initial_capital: float = 10000,
        transaction_cost: float = 0.0005,
        slippage: float = 0.0005,
        annual_rf_rate: float = 0.05,
        probability_thresholds: list[float] = None,
        position_sizing: str = "equal_weight",
        position_selection: str = "top_5",
        allocation_mode: str = "full_deployment",
        purchase_threshold: float = 0.50,
        parallelization: Dict[str, int] = None,
    ):
        """Initialize pipeline with data source and algorithms."""
        self.stock = stock
        self.algorithms = algorithms
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        self.features = FeatureEngineer()
        
        # Walk-Forward Validation parameters
        self.wfv_train_window = wfv_train_window
        self.wfv_test_window = wfv_test_window
        
        # Backtesting parameters
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.annual_rf_rate = annual_rf_rate
        self.probability_thresholds = probability_thresholds or [0.50, 0.55, 0.60, 0.65, 0.70]
        self.position_sizing = position_sizing
        self.position_selection = position_selection
        self.allocation_mode = allocation_mode
        self.purchase_threshold = purchase_threshold
        
        # Parallelization settings (with defaults)
        self.parallelization = parallelization or {
            "algorithm_selection": 2,
            "fold_evaluation": 4,
            "threshold_testing": 3,
        }
        
        # Initialize components
        self.model_selector = ModelSelector(
            wfv_train_window, wfv_test_window,
            max_workers=self.parallelization.get("algorithm_selection", 2),
            fold_workers=self.parallelization.get("fold_evaluation", 4)
        )
        self.metrics_evaluator = MetricsEvaluator()
        self.reporter = PipelineReporter(self.output_dir)
        self.logger = get_logger()
        
        # Timing tracking
        self.phase_times = {}
    
    def run(self, save: bool = True) -> Dict[str, Any]:
        """Execute complete pipeline: data → features → model selection → backtest.
        
        Returns:
            Dictionary with metrics and backtest results
        """
        pipeline_start = time.time()
        
        self.reporter.log_phase_start(
            "0: Data Preparation",
            "Loading historical data and engineering features"
        )
        phase_start = time.time()
        
        try:
            # Phase 0: Data preparation
            self.logger.info("Loading historical stock data...")
            raw_df = self.stock.fetch()
            self.logger.info(f"✓ Fetched {len(raw_df)} records")
            
            self.logger.info("Building features...")
            dataset, feature_cols, target_col = self.features.build(
                raw_df, self.algorithms[0].feature_profile()
            )
            self.logger.info(f"✓ Built {len(feature_cols)} features, dataset shape: {dataset.shape}")
            
            self.logger.info("Performing temporal train-test split (80-20)...")
            train_df, test_df = self._split(dataset)
            self.phase_times["0_data_preparation"] = time.time() - phase_start
            self.reporter.log_phase_end("0: Data Preparation")
            
            # Phase 1: Model selection via walk-forward validation
            self.reporter.log_phase_start(
                "1: Model Selection",
                "Using walk-forward validation with Information Coefficient"
            )
            phase_start = time.time()
            self.logger.info(f"Testing {len(self.algorithms)} algorithms...")
            best_model, best_model_name, best_score = self.model_selector.select(
                self.algorithms, train_df, feature_cols, target_col
            )
            self.phase_times["1_model_selection"] = time.time() - phase_start
            self.reporter.log_model_selection(best_model_name, best_score, {})
            self.reporter.log_phase_end("1: Model Selection")
            
            # Save model selection results for visualization
            model_results = {}
            for model_name, model_data in self.model_selector.results.items():
                model_results[model_name] = {
                    "mean_ic": model_data['mean_ic'],
                    "std_ic": model_data['std_ic'],
                    "wfv_score": model_data['score'],
                    "accuracy": model_data.get('accuracy', 0),
                    "f1_score": model_data.get('f1_score', 0),
                    "auc": model_data.get('auc', 0.5),
                    "elapsed_time": model_data['elapsed']
                }
            
            # Phase 2: Fit best model on full train set (required for predictions)
            self.reporter.log_phase_start(
                "2: Model Training",
                f"Training {best_model_name} on full training dataset"
            )
            phase_start = time.time()
            self.logger.info(f"Training {best_model_name} on full train set ({len(train_df)} samples)...")
            best_model.fit(train_df, pd.DataFrame(), feature_cols, target_col)
            train_time = time.time() - phase_start
            self.logger.info(f"✓ Training completed in {train_time:.2f}s")
            self.phase_times["2_model_training"] = train_time
            self.reporter.log_phase_end("2: Model Training")
            
            # Phase 3: Backtesting (use trained model for predictions)
            if save:
                self.reporter.log_phase_start(
                    "3: Backtesting",
                    f"Running backtest on test set using {best_model_name}"
                )
                phase_start = time.time()
                self.logger.info(f"Running backtest with {best_model_name} predictions on test set...")
                backtest_results = self._run_backtest(best_model, test_df, feature_cols)
                self.phase_times["3_backtesting"] = time.time() - phase_start
                self.reporter.log_phase_end("3: Backtesting")
            
            # Phase 4: Visualization (save results and generate plots)
            if save:
                self.reporter.log_phase_start(
                    "4: Visualization",
                    "Saving results and generating all plots"
                )
                phase_start = time.time()
                
                # Save model selection results to JSON
                import json
                models_output = self.output_dir / "models_comparison.json"
                with open(models_output, 'w') as f:
                    json.dump(model_results, f, indent=2)
                self.logger.info(f"✓ Model comparison saved to {models_output}")
                
                # Generate model metrics comparison plot via PlotGenerator
                self.logger.info(f"Generating model selection comparison plot...")
                plot_gen_for_models = PlotGenerator(self.initial_capital)
                plot_gen_for_models.plot_model_metrics_comparison(model_results, str(self.output_dir))
                
                self.phase_times["4_visualization"] = time.time() - phase_start
                self.reporter.log_phase_end("4: Visualization")
            
            # Log timing summary
            total_time = time.time() - pipeline_start
            self.reporter.log_timing_summary(self.phase_times, total_time)
            
            self.logger.info("="*80)
            self.logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            self.logger.info("="*80)
            
            return backtest_results if save else {}
        
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            self.reporter.log_phase_end("Pipeline", "FAILED")
            raise
    
    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/test by date (temporal split)."""
        dates = df["date"].sort_values().unique()
        n = len(dates)
        train_end_idx = int(n * (1 - self.test_size))
        split_date = dates[train_end_idx]
        
        train_df = df[df["date"] < split_date].copy()
        test_df = df[df["date"] >= split_date].copy()
        
        self.logger.info(
            f"Train: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})"
        )
        self.logger.info(
            f"Test:  {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})"
        )
        
        return train_df, test_df
    
    def _run_backtest(self, best_algo: Algorithm, test_df: pd.DataFrame, 
                     features: list[str]) -> Dict[str, Any]:
        """Run backtesting with multiple probability thresholds and smart trading strategies.
        
        Returns:
            Dictionary with backtest results from all strategies
        """
        self.logger.info("Initializing backtest engine...")
        
        # Get predictions
        y_prob = best_algo.predict_proba(test_df, features)
        
        # Run backtest
        backtest = Backtest(
            test_df,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            slippage=self.slippage,
            annual_rf_rate=self.annual_rf_rate,
            position_sizing=self.position_sizing,
            position_selection=self.position_selection,
            allocation_mode=self.allocation_mode,
            purchase_threshold=self.purchase_threshold,
            threshold_workers=self.parallelization.get("threshold_testing", 3),
            output_dir=str(self.output_dir)
        )
        
        self.logger.info(f"Running backtest with all 8 strategies and {len(self.probability_thresholds)} thresholds...")
        self.logger.info(f"Probability thresholds: {self.probability_thresholds}")
        results = backtest.run_threshold_strategies(y_prob, self.probability_thresholds)
        
        # Save and plot
        self.logger.info("Saving backtest summary...")
        backtest.save_summary(results, str(self.output_dir))
        
        self.logger.info("Generating backtest visualizations...")
        backtest.plot_results(results, str(self.output_dir))
        
        self.reporter.log_backtest_results(results)
        
        return results

