"""
Crypto market data integration using gapless-crypto-data.
Replaces synthetic data with authentic Binance OHLCV microstructure data.
"""
import pandas as pd
import numpy as np
import torch
from typing import List, Union, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CryptoDataProvider:
    """
    Cryptocurrency data provider using gapless-crypto-data for authentic market data.
    Implements same interface as synthetic fallback for seamless integration.
    """

    def __init__(
        self,
        data_path: str = "crypto_data",
        symbols: List[str] = None,
        timeframe: str = "1h",
        device: torch.device = None
    ):
        self.data_path = Path(data_path)
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        self.timeframe = timeframe
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._data_cache = {}

    def _ensure_data_exists(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Ensure cryptocurrency data exists for specified period."""
        try:
            from gapless_crypto_data import BinancePublicDataCollector

            output_path = self.data_path / symbol
            output_path.mkdir(parents=True, exist_ok=True)

            # Collect data if not exists
            csv_file = output_path / f"{symbol}_{self.timeframe}_{start_date}_{end_date}.csv"
            if not csv_file.exists():
                logger.info(f"Collecting {symbol} data from {start_date} to {end_date}")

                # Use correct gapless-crypto-data API
                collector = BinancePublicDataCollector(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=str(output_path)
                )

                # Collect timeframe data
                collector.collect_timeframe_data(trading_timeframe=self.timeframe)

                # Find the generated CSV file
                generated_files = list(output_path.glob(f"*{symbol}*{self.timeframe}*.csv"))
                if generated_files:
                    # Use the first generated file
                    csv_file = generated_files[0]
                    logger.info(f"Found generated file: {csv_file}")
                else:
                    raise FileNotFoundError(f"No CSV file generated for {symbol}")

            return csv_file

        except ImportError:
            raise ImportError("gapless-crypto-data not installed. Run: uv add gapless-crypto-data")
        except Exception as e:
            logger.error(f"Data collection failed for {symbol}: {e}")
            raise

    def load_data(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        features: List[str] = None
    ) -> pd.DataFrame:
        """
        Load cryptocurrency OHLCV data with microstructure features.

        Args:
            symbols: Symbol or list of symbols (e.g., 'BTCUSDT' or ['BTCUSDT', 'ETHUSDT'])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            features: Feature columns to extract (default: OHLCV + volume metrics)

        Returns:
            pd.DataFrame: Multi-index DataFrame with (date, symbol) index
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        if features is None:
            # Full 11-column microstructure dataset
            features = [
                'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
            ]

        all_data = []

        for symbol in symbols:
            try:
                # Ensure data exists
                csv_file = self._ensure_data_exists(symbol, start_date, end_date)

                # Load CSV data (skip metadata header lines starting with #)
                df = pd.read_csv(csv_file, comment='#')

                # gapless-crypto-data already provides correctly named columns

                # Convert date column (already in datetime format)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

                # Add symbol level
                df['symbol'] = symbol
                df = df.set_index('symbol', append=True)
                df = df.swaplevel().sort_index()

                # Filter to requested features
                available_features = [f for f in features if f in df.columns]
                if len(available_features) < len(features):
                    missing = set(features) - set(available_features)
                    logger.warning(f"Missing features for {symbol}: {missing}")

                df = df[available_features]
                all_data.append(df)

                logger.info(f"Loaded {symbol}: {len(df)} records from {df.index.get_level_values(1).min()} to {df.index.get_level_values(1).max()}")

            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No data loaded for any symbols")

        # Combine all symbol data
        combined_df = pd.concat(all_data)
        combined_df = combined_df.sort_index()

        # Calculate derived features
        combined_df = self._add_derived_features(combined_df)

        logger.info(f"Combined dataset: {len(combined_df)} total records across {len(symbols)} symbols")
        return combined_df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive microstructure features for quantitative analysis."""

        # Basic VWAP calculation
        if 'volume' in df.columns and 'close' in df.columns:
            df['vwap'] = (df['close'] * df['volume']).groupby(level=0).cumsum() / df['volume'].groupby(level=0).cumsum()

        # Price-based features
        if 'close' in df.columns:
            df['returns'] = df.groupby(level=0)['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df.groupby(level=0)['close'].shift(1))

        # Volatility (rolling 24-period for hourly data)
        if 'returns' in df.columns:
            df['volatility'] = df.groupby(level=0)['returns'].rolling(24, min_periods=1).std().reset_index(level=0, drop=True)

        # Microstructure features from order flow data
        if all(col in df.columns for col in ['taker_buy_base_asset_volume', 'volume']):
            # Order flow imbalance (taker buy vs total volume)
            df['order_flow_imbalance'] = (df['taker_buy_base_asset_volume'] / df['volume']) - 0.5

        if all(col in df.columns for col in ['taker_buy_quote_asset_volume', 'quote_asset_volume']):
            # Quote asset order flow imbalance
            df['quote_flow_imbalance'] = (df['taker_buy_quote_asset_volume'] / df['quote_asset_volume']) - 0.5

        # Trade intensity and market activity
        if 'number_of_trades' in df.columns:
            # Average trade size
            if 'volume' in df.columns:
                df['avg_trade_size'] = df['volume'] / df['number_of_trades']

            # Trade intensity (normalized by rolling window)
            df['trade_intensity'] = df.groupby(level=0)['number_of_trades'].rolling(24, min_periods=1).rank(pct=True).reset_index(level=0, drop=True)

        # Volume ratios and spreads
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # Price spread relative to close
            df['price_spread_ratio'] = (df['high'] - df['low']) / df['close']

        if all(col in df.columns for col in ['volume', 'quote_asset_volume']):
            # Volume concentration ratio
            df['volume_concentration'] = df['volume'] / df['quote_asset_volume']

        # Market impact proxies
        if all(col in df.columns for col in ['taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']):
            # Taker pressure (aggressive vs passive flow)
            total_taker = df['taker_buy_base_asset_volume'] + (df['volume'] - df['taker_buy_base_asset_volume'])
            df['taker_pressure'] = df['taker_buy_base_asset_volume'] / total_taker

        # Rolling statistics for microstructure features
        window = 24  # 24-period rolling window
        microstructure_cols = ['order_flow_imbalance', 'avg_trade_size', 'price_spread_ratio']

        for col in microstructure_cols:
            if col in df.columns:
                # Rolling mean and std for normalization
                df[f'{col}_rolling_mean'] = df.groupby(level=0)[col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
                df[f'{col}_rolling_std'] = df.groupby(level=0)[col].rolling(window, min_periods=1).std().reset_index(level=0, drop=True)

                # Z-score normalization
                df[f'{col}_zscore'] = ((df[col] - df[f'{col}_rolling_mean']) /
                                     df[f'{col}_rolling_std']).fillna(0)

        # Factor column for compatibility
        df['factor'] = 1.0

        logger.info(f"Added {len([c for c in df.columns if c not in ['factor']]) - len(['open', 'high', 'low', 'close', 'volume'])} microstructure features")

        return df

    def get_expressions_data(
        self,
        expressions: List[str],
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load data and create expression columns compatible with AlphaForge format.

        Args:
            expressions: List of expression strings (e.g., ['$open*$factor', '$close*$factor'])
            symbols: Symbol or list of symbols
            start_date: Start date
            end_date: End date

        Returns:
            pd.DataFrame: Data with expression columns
        """
        # Load base data
        df = self.load_data(symbols, start_date, end_date)

        # Create expression columns
        result_data = {}

        for expr in expressions:
            try:
                # Map AlphaForge expressions to data columns
                if expr == '$open*$factor' or expr == '$open':
                    result_data[expr] = df['open'] * df['factor']
                elif expr == '$close*$factor' or expr == '$close':
                    result_data[expr] = df['close'] * df['factor']
                elif expr == '$high*$factor' or expr == '$high':
                    result_data[expr] = df['high'] * df['factor']
                elif expr == '$low*$factor' or expr == '$low':
                    result_data[expr] = df['low'] * df['factor']
                elif expr == '$volume/$factor/1000000' or expr == '$volume':
                    result_data[expr] = df['volume'] / df['factor'] / 1000000
                elif expr == '$vwap*$factor' or expr == '$vwap':
                    result_data[expr] = df['vwap'] * df['factor']
                else:
                    logger.warning(f"Unknown expression: {expr}, using close price")
                    result_data[expr] = df['close'] * df['factor']

            except Exception as e:
                logger.error(f"Error evaluating expression {expr}: {e}")
                # Fallback to close price
                result_data[expr] = df['close'] * df.get('factor', 1.0)

        result_df = pd.DataFrame(result_data, index=df.index)

        logger.info(f"Generated expression data: {result_df.shape}")
        return result_df

def get_crypto_data_provider(**kwargs) -> CryptoDataProvider:
    """Factory function for crypto data provider."""
    return CryptoDataProvider(**kwargs)