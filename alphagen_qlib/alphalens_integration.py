"""
Alphalens integration for community-proven alpha factor analysis.
Replaces custom alpha evaluation with industry-standard Alphalens framework.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from alphagen.data.expression import Expression

try:
    import alphalens as al
    ALPHALENS_AVAILABLE = True
except ImportError:
    ALPHALENS_AVAILABLE = False
    logging.warning("alphalens-reloaded not available. Run: uv add alphalens-reloaded")

logger = logging.getLogger(__name__)

class AlphalensAnalyzer:
    """
    Community-proven alpha factor analysis using Alphalens framework.

    Provides industry-standard alpha factor evaluation, tear sheets,
    and performance metrics with established best practices.
    """

    def __init__(
        self,
        pricing_data: pd.DataFrame,
        quantiles: int = 5,
        periods: List[int] = [1, 5, 10],
        max_loss: float = 0.35,
        zero_aware: bool = True
    ):
        """
        Initialize Alphalens analyzer with market data.

        Args:
            pricing_data: Multi-index DataFrame with (date, symbol) index and price columns
            quantiles: Number of quantiles for factor analysis
            periods: Forward return periods to analyze
            max_loss: Maximum data loss tolerance for cleaning
            zero_aware: Whether to handle zero returns specially
        """
        if not ALPHALENS_AVAILABLE:
            raise ImportError("alphalens-reloaded required. Run: uv add alphalens-reloaded")

        self.pricing_data = pricing_data
        self.quantiles = quantiles
        self.periods = periods
        self.max_loss = max_loss
        self.zero_aware = zero_aware

        # Prepare pricing data for Alphalens
        self._prepare_pricing_data()

    def _prepare_pricing_data(self):
        """Prepare pricing data in Alphalens-compatible format."""
        if 'close' not in self.pricing_data.columns:
            raise ValueError("Pricing data must contain 'close' column")

        # Ensure proper MultiIndex structure
        if not isinstance(self.pricing_data.index, pd.MultiIndex):
            raise ValueError("Pricing data must have MultiIndex (date, symbol)")

        # Extract close prices and pivot for Alphalens
        close_prices = self.pricing_data['close'].unstack(level=0)

        # Ensure datetime index
        close_prices.index = pd.to_datetime(close_prices.index)

        self.prices = close_prices
        logger.info(f"Prepared pricing data: {self.prices.shape} for {len(self.prices.columns)} symbols")

    def evaluate_factor(
        self,
        factor_data: pd.Series,
        factor_name: str = "alpha_factor"
    ) -> Dict[str, pd.DataFrame]:
        """
        Evaluate alpha factor using Alphalens framework.

        Args:
            factor_data: Factor values with MultiIndex (date, symbol)
            factor_name: Name for the factor being analyzed

        Returns:
            Dict with Alphalens analysis results
        """
        try:
            # Clean and prepare factor data
            clean_factor_data = al.utils.get_clean_factor_and_forward_returns(
                factor=factor_data,
                prices=self.prices,
                quantiles=self.quantiles,
                periods=self.periods,
                max_loss=self.max_loss,
                zero_aware=self.zero_aware
            )

            logger.info(f"Factor data cleaned: {len(clean_factor_data)} observations")

            # Comprehensive factor analysis
            results = {}

            # Performance analysis
            results['returns'] = al.performance.mean_return_by_quantile(
                clean_factor_data,
                by_date=False,
                by_group=False
            )

            results['information_coefficient'] = al.performance.factor_information_coefficient(
                clean_factor_data
            )

            results['quantile_returns'] = al.performance.mean_return_by_quantile(
                clean_factor_data,
                by_date=True
            )[0]  # Get the returns DataFrame

            # Turnover analysis
            results['turnover'] = al.performance.quantile_turnover(
                clean_factor_data,
                quantile=1,  # Top quantile
                period=self.periods[0]
            )

            # Summary statistics
            ic_summary = results['information_coefficient'].describe()
            results['ic_summary'] = ic_summary

            logger.info(f"Factor analysis complete. IC Mean: {ic_summary['mean']:.4f}, "
                       f"IC Std: {ic_summary['std']:.4f}")

            return results

        except Exception as e:
            logger.error(f"Factor evaluation failed: {e}")
            raise

    def create_tear_sheet(
        self,
        factor_data: pd.Series,
        factor_name: str = "alpha_factor",
        output_path: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive Alphalens tear sheet.

        Args:
            factor_data: Factor values with MultiIndex (date, symbol)
            factor_name: Name for the factor
            output_path: Optional path to save tear sheet

        Returns:
            Analysis results dictionary
        """
        try:
            # Clean factor data
            clean_factor_data = al.utils.get_clean_factor_and_forward_returns(
                factor=factor_data,
                prices=self.prices,
                quantiles=self.quantiles,
                periods=self.periods,
                max_loss=self.max_loss,
                zero_aware=self.zero_aware
            )

            # Generate full tear sheet
            if output_path:
                import matplotlib.pyplot as plt
                fig = al.tears.create_full_tear_sheet(
                    clean_factor_data,
                    long_short=False,
                    group_neutral=False,
                    by_group=False
                )
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                plt.close()
                logger.info(f"Tear sheet saved to: {output_path}")
            else:
                al.tears.create_full_tear_sheet(
                    clean_factor_data,
                    long_short=False,
                    group_neutral=False,
                    by_group=False
                )

            # Return detailed analysis
            return self.evaluate_factor(factor_data, factor_name)

        except Exception as e:
            logger.error(f"Tear sheet generation failed: {e}")
            raise

    def compare_factors(
        self,
        factors_dict: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Compare multiple alpha factors using Alphalens.

        Args:
            factors_dict: Dictionary of {factor_name: factor_series}

        Returns:
            Comparison DataFrame with IC statistics
        """
        comparison_results = []

        for name, factor_data in factors_dict.items():
            try:
                results = self.evaluate_factor(factor_data, name)
                ic_stats = results['ic_summary']

                comparison_results.append({
                    'factor_name': name,
                    'ic_mean': ic_stats['mean'],
                    'ic_std': ic_stats['std'],
                    'ic_ir': ic_stats['mean'] / ic_stats['std'] if ic_stats['std'] > 0 else 0,
                    'ic_skew': ic_stats.get('skew', np.nan),
                    'ic_kurtosis': ic_stats.get('kurtosis', np.nan)
                })

            except Exception as e:
                logger.warning(f"Factor {name} analysis failed: {e}")
                comparison_results.append({
                    'factor_name': name,
                    'ic_mean': np.nan,
                    'ic_std': np.nan,
                    'ic_ir': np.nan,
                    'ic_skew': np.nan,
                    'ic_kurtosis': np.nan
                })

        return pd.DataFrame(comparison_results).set_index('factor_name')

class AlphaForgeAlphalensAdapter:
    """
    Adapter to integrate AlphaForge expressions with Alphalens analysis.
    """

    def __init__(self, stock_data, alphalens_analyzer: AlphalensAnalyzer):
        """
        Initialize adapter.

        Args:
            stock_data: AlphaForge StockData instance
            alphalens_analyzer: Configured AlphalensAnalyzer
        """
        self.stock_data = stock_data
        self.analyzer = alphalens_analyzer

    def evaluate_expression(
        self,
        expression: Expression,
        expression_name: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Evaluate AlphaForge expression using Alphalens.

        Args:
            expression: AlphaForge Expression object
            expression_name: Name for the expression

        Returns:
            Alphalens analysis results
        """
        if expression_name is None:
            expression_name = str(expression)

        try:
            # Evaluate expression on stock data
            # Note: This requires proper period slicing for evaluation
            period = slice(self.stock_data.max_backtrack_days,
                          self.stock_data.n_days + self.stock_data.max_backtrack_days)

            factor_values = expression.evaluate(self.stock_data, period)

            # Convert to pandas Series with proper MultiIndex
            factor_series = self._tensor_to_factor_series(factor_values)

            # Analyze using Alphalens
            results = self.analyzer.evaluate_factor(factor_series, expression_name)

            logger.info(f"Expression '{expression_name}' evaluated successfully")
            return results

        except Exception as e:
            logger.error(f"Expression evaluation failed: {e}")
            raise

    def _tensor_to_factor_series(self, tensor) -> pd.Series:
        """Convert AlphaForge tensor to Alphalens-compatible Series."""
        # Convert tensor to numpy
        if hasattr(tensor, 'cpu'):
            values = tensor.cpu().numpy()
        else:
            values = np.array(tensor)

        # Create MultiIndex for factor data
        dates = self.stock_data._dates[self.stock_data.max_backtrack_days:
                                     self.stock_data.max_backtrack_days + values.shape[0]]
        symbols = self.stock_data._stock_ids

        # Flatten tensor and create Series
        factor_data = []
        index_data = []

        for i, date in enumerate(dates):
            for j, symbol in enumerate(symbols):
                if i < values.shape[0] and j < values.shape[1]:
                    factor_data.append(values[i, j])
                    index_data.append((date, symbol))

        index = pd.MultiIndex.from_tuples(index_data, names=['date', 'symbol'])
        return pd.Series(factor_data, index=index)

    def batch_evaluate_expressions(
        self,
        expressions: List[Expression],
        expression_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate multiple expressions and compare results.

        Args:
            expressions: List of AlphaForge expressions
            expression_names: Optional names for expressions

        Returns:
            Comparison DataFrame
        """
        if expression_names is None:
            expression_names = [str(expr) for expr in expressions]

        factors_dict = {}
        for expr, name in zip(expressions, expression_names):
            try:
                period = slice(self.stock_data.max_backtrack_days,
                              self.stock_data.n_days + self.stock_data.max_backtrack_days)
                factor_values = expr.evaluate(self.stock_data, period)
                factors_dict[name] = self._tensor_to_factor_series(factor_values)
            except Exception as e:
                logger.warning(f"Expression '{name}' evaluation failed: {e}")

        return self.analyzer.compare_factors(factors_dict)

def create_alphalens_analyzer(crypto_data_provider) -> Optional[AlphalensAnalyzer]:
    """
    Factory function to create Alphalens analyzer from crypto data.

    Args:
        crypto_data_provider: CryptoDataProvider instance

    Returns:
        Configured AlphalensAnalyzer or None if setup fails
    """
    if not ALPHALENS_AVAILABLE:
        logger.warning("Alphalens not available")
        return None

    try:
        # Load sample data to create analyzer
        sample_data = crypto_data_provider.load_data(
            symbols=['BTCUSDT', 'ETHUSDT'],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        analyzer = AlphalensAnalyzer(
            pricing_data=sample_data,
            quantiles=5,
            periods=[1, 5, 10],
            max_loss=0.35
        )

        logger.info("Alphalens analyzer created successfully")
        return analyzer

    except Exception as e:
        logger.error(f"Failed to create Alphalens analyzer: {e}")
        return None