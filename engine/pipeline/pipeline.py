import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Tuple, Dict, Any
from engine.validation.walk_forward_validator import WalkForwardValidator
from engine.log.reporters import PipelineReporter
from engine.backtesting import Backtest
from engine.stock.stock import Stock
from engine.algorithms.base import Algorithm
from engine.log.logger_config import get_logger


class Pipeline:
    """Main ML pipeline orchestrator.
    
    Composes: WalkForwardValidator, Backtest
    Workflow: Data → Features → Walk-Forward Validation → Backtesting
    """
    
    def __init__(
        self,
        stock: Stock,
        algorithm: Algorithm,
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
        """Initialize pipeline with data source and algorithm."""
        self.stock = stock
        self.algorithm = algorithm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        
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
            "fold_evaluation": 4,
            "threshold_testing": 3,
        }
        
        # Initialize components
        self.reporter = PipelineReporter(self.output_dir)
        self.logger = get_logger()
        
        # Timing tracking
        self.phase_times = {}
    
    def run(self, save: bool = True) -> Dict[str, Any]:
        """Execute complete pipeline: data → features → walk-forward validation → backtest.
        
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
            dataset, feature_cols, target_col = self.stock.build_features(raw_df)
            self.logger.info(f"✓ Built {len(feature_cols)} features, dataset shape: {dataset.shape}")
            
            self.logger.info("Performing temporal train-test split (80-20)...")
            train_df, test_df = self._split(dataset)
            self.phase_times["0_data_preparation"] = time.time() - phase_start
            self.reporter.log_phase_end("0: Data Preparation")
            
            # Phase 1: Walk-forward validation
            self.reporter.log_phase_start(
                "1: Walk-Forward Validation",
                "Using walk-forward validation with Information Coefficient"
            )
            phase_start = time.time()
            self.logger.info(f"Validating model: {self.algorithm.name()}")
            validator = WalkForwardValidator(
                self.wfv_train_window,
                self.wfv_test_window,
                max_workers=self.parallelization.get("fold_evaluation", 4),
            )
            wfv_df, predictions, probs, fold_ics, mean_ic, std_ic = validator.validate(
                train_df, self.algorithm, feature_cols, target_col
            )
            self.phase_times["1_walk_forward_validation"] = time.time() - phase_start
            metrics = self._calculate_wfv_metrics(wfv_df, predictions, probs, target_col)
            self.reporter.log_wfv_results(self.algorithm.name(), mean_ic, std_ic, fold_ics, metrics)
            self.reporter.log_phase_end("1: Walk-Forward Validation")
            
            # Phase 2: Fit best model on full train set (required for predictions)
            self.reporter.log_phase_start(
                "2: Model Training",
                f"Training {self.algorithm.name()} on full training dataset"
            )
            phase_start = time.time()
            self.logger.info(f"Training {self.algorithm.name()} on full train set ({len(train_df)} samples)...")
            self.algorithm.fit(train_df, pd.DataFrame(), feature_cols, target_col)
            train_time = time.time() - phase_start
            self.logger.info(f"✓ Training completed in {train_time:.2f}s")
            self.phase_times["2_model_training"] = train_time
            self.reporter.log_phase_end("2: Model Training")
            
            # Phase 3: Backtesting (use trained model for predictions)
            if save:
                self.reporter.log_phase_start(
                    "3: Backtesting",
                    f"Running backtest on test set using {self.algorithm.name()}"
                )
                phase_start = time.time()
                self.logger.info(f"Running backtest with {self.algorithm.name()} predictions on test set...")
                backtest_results = self._run_backtest(self.algorithm, test_df, feature_cols)
                self.phase_times["3_backtesting"] = time.time() - phase_start
                self.reporter.log_phase_end("3: Backtesting")
            
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

    def _calculate_wfv_metrics(
        self,
        wfv_df: pd.DataFrame,
        predictions: np.ndarray,
        probs: np.ndarray,
        target_col: str,
    ) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        y_true = wfv_df[target_col].to_numpy()
        y_pred = np.array(predictions)
        y_prob = np.array(probs)

        try:
            acc = float(accuracy_score(y_true, y_pred))
        except Exception:
            acc = 0.0

        try:
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
        except Exception:
            f1 = 0.0

        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5

        return {
            "accuracy": acc,
            "f1_score": f1,
            "auc": auc,
        }
    
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
        backtest.save_summary(results)
        
        self.logger.info("Generating backtest visualizations...")
        backtest.plot_results(results, str(self.output_dir))
        
        self.reporter.log_backtest_results(results)
        
        return results

