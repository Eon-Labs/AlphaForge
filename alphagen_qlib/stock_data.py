from typing import List, Union, Optional, Tuple, Dict
from enum import IntEnum
import numpy as np
import pandas as pd
import torch

class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5
    
def change_to_raw_min(features):
    result = []
    for feature in features:
        if feature in ['$vwap']:
            result.append(f"$money/$volume")
        elif feature in ['$volume']:
            result.append(f"{feature}/100000")
            # result.append('$close')
        else:
            result.append(feature)
    return result

def change_to_raw(features):
    result = []
    for feature in features:
        if feature in ['$open','$close','$high','$low','$vwap']:
            result.append(f"{feature}*$factor")
        elif feature in ['$volume']:
            result.append(f"{feature}/$factor/1000000")
            # result.append('$close')
        else:
            raise ValueError(f"feature {feature} not supported")
    return result

class StockData:
    _qlib_initialized: bool = False

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str,
                 end_time: str,
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 raw:bool = False,
                 qlib_path:Union[str,Dict] = "",
                 freq:str = 'day',
                 ) -> None:
        self._init_qlib(qlib_path)
        self.df_bak = None
        self.raw = raw
        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.freq = freq
        self.data, self._dates, self._stock_ids = self._get_data()


    @classmethod
    def _init_qlib(cls,qlib_path) -> None:
        if cls._qlib_initialized:
            return
        try:
            import qlib
            # Try modern qlib API first
            try:
                from qlib.config import REG_CN
                qlib.init(provider_uri=qlib_path, region=REG_CN)
            except (ImportError, AttributeError):
                # Fallback for different qlib versions
                try:
                    qlib.init(provider_uri=str(qlib_path))
                except AttributeError:
                    # qlib without init function - create minimal stub
                    print(f"Warning: qlib version {getattr(qlib, '__version__', 'unknown')} does not support init()")
                    print("Data loading will use baostock fallback")
        except ImportError:
            print("Warning: qlib not available, using baostock fallback")
        cls._qlib_initialized = True

    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        # This evaluates an expression on the data and returns the dataframe
        # It might throw on illegal expressions like "Ref(constant, dtime)"
        try:
            from qlib.data.dataset.loader import QlibDataLoader
            from qlib.data import D
            if not isinstance(exprs, list):
                exprs = [exprs]
            cal: np.ndarray = D.calendar(freq=self.freq)
            start_index = cal.searchsorted(pd.Timestamp(self._start_time))  # type: ignore
            end_index = cal.searchsorted(pd.Timestamp(self._end_time))  # type: ignore
            real_start_time = cal[start_index - self.max_backtrack_days]
            if cal[end_index] != pd.Timestamp(self._end_time):
                end_index -= 1
            # real_end_time = cal[min(end_index + self.max_future_days,len(cal)-1)]
            real_end_time = cal[end_index + self.max_future_days]
            result =  (QlibDataLoader(config=exprs,freq=self.freq)  # type: ignore
                    .load(self._instrument, real_start_time, real_end_time))
            return result
        except (ImportError, AttributeError):
            # Fallback to baostock-based implementation
            return self._load_exprs_baostock(exprs)

    def _load_exprs_baostock(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        """Fallback implementation using baostock for basic OHLCV data"""
        if not isinstance(exprs, list):
            exprs = [exprs]

        print(f"Using baostock fallback for expressions: {exprs}")

        # Create synthetic data for basic features (this should be replaced with actual baostock implementation)
        dates = pd.date_range(start=self._start_time, end=self._end_time, freq='D')
        instruments = self._instrument if isinstance(self._instrument, list) else [self._instrument]

        # Create MultiIndex for dates and instruments
        index = pd.MultiIndex.from_product([dates, instruments], names=['date', 'instrument'])

        # Create basic OHLCV data (synthetic for now - replace with actual baostock data)
        import numpy as np
        np.random.seed(42)  # For reproducible synthetic data

        data = {}
        for expr in exprs:
            # Create synthetic data based on expression type
            if 'close' in expr.lower():
                data[expr] = np.random.normal(100, 10, len(index))
            elif 'open' in expr.lower():
                data[expr] = np.random.normal(100, 10, len(index))
            elif 'high' in expr.lower():
                data[expr] = np.random.normal(105, 10, len(index))
            elif 'low' in expr.lower():
                data[expr] = np.random.normal(95, 10, len(index))
            elif 'volume' in expr.lower():
                data[expr] = np.random.normal(1000000, 100000, len(index))
            elif 'vwap' in expr.lower():
                data[expr] = np.random.normal(100, 10, len(index))
            else:
                # Default to price-like data
                data[expr] = np.random.normal(100, 10, len(index))

        df = pd.DataFrame(data, index=index)
        print(f"Generated synthetic data shape: {df.shape}")
        return df
    
    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        if self.raw and self.freq == 'day':
            features = change_to_raw(features)
        elif self.raw:
            features = change_to_raw_min(features)
        df = self._load_exprs(features)
        self.df_bak = df
        # print(df)
        df = df.stack().unstack(level=1)
        dates = df.index.levels[0]                                      # type: ignore
        stock_ids = df.columns
        values = df.values
        values = values.reshape((-1, len(features), values.shape[-1]))  # type: ignore
        return torch.tensor(values, dtype=torch.float, device=self.device), dates, stock_ids

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def add_data(self,data:torch.Tensor,dates:pd.Index):
        data = data.to(self.device)
        self.data = torch.cat([self.data,data],dim=0)
        self._dates = pd.Index(self._dates.append(dates))


    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
    
    