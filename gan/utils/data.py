"""
Cryptocurrency market data loading pipeline using gapless-crypto-data.
Replaces qlib/StockData synthetic fallback with authentic Binance market data.
"""

from alphagen_generic.features import open_
from gan.utils import Builders
from alphagen_generic.features import *
from alphagen.data.expression import *
import torch
import os

def get_data_by_year(
    train_start = 2010,train_end=2019,valid_year=2020,test_year =2021,
    instruments=None, target=None,freq=None,
                    ):
    """
    Load cryptocurrency market data with train/valid/test temporal splits.

    Uses CryptoDataProvider for authentic Binance OHLCV + microstructure data
    instead of synthetic fallback mechanisms.
    """

    from gan.utils import load_pickle,save_pickle
    from alphagen_qlib.crypto_data import CryptoDataProvider

    # Map traditional instruments to crypto symbols
    crypto_symbols = _map_instruments_to_crypto(instruments)

    # Convert years to date strings for crypto data
    train_dates=(f"{train_start}-01-01", f"{train_end}-12-31")
    val_dates=(f"{valid_year}-01-01", f"{valid_year}-12-31")
    test_dates=(f"{test_year}-01-01", f"{test_year}-12-31")

    train_start,train_end = train_dates
    valid_start,valid_end = val_dates
    valid_head_start = f"{valid_year-2}-01-01"
    test_start,test_end = test_dates
    test_head_start = f"{test_year-2}-01-01"

    # Create cache identifier including crypto symbols
    symbol_str = '_'.join(crypto_symbols)
    name = f"{symbol_str}_crypto_pkl_{str(target).replace('/','_').replace(' ','')}_{freq}"
    name = f"{name}_{train_start}_{train_end}_{valid_start}_{valid_end}_{test_start}_{test_end}"

    # Attempt to load from cache
    try:
        data = load_pickle(f'pkl/{name}/data.pkl')
        data_valid = load_pickle(f'pkl/{name}/data_valid.pkl')
        data_valid_withhead = load_pickle(f'pkl/{name}/data_valid_withhead.pkl')
        data_test = load_pickle(f'pkl/{name}/data_test.pkl')
        data_test_withhead = load_pickle(f'pkl/{name}/data_test_withhead.pkl')
        print(f"Loaded cached crypto data: {name}")

    except:
        print('Crypto data cache miss, loading from gapless-crypto-data')

        # Initialize crypto data provider
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        timeframe = "1h" if freq == 'hour' else "1d"

        # Load data for each time period using authentic crypto market data
        data = _load_crypto_stockdata(crypto_symbols, train_start, train_end, timeframe, device)
        data_valid = _load_crypto_stockdata(crypto_symbols, valid_start, valid_end, timeframe, device)
        data_valid_withhead = _load_crypto_stockdata(crypto_symbols, valid_head_start, valid_end, timeframe, device)
        data_test = _load_crypto_stockdata(crypto_symbols, test_start, test_end, timeframe, device)
        data_test_withhead = _load_crypto_stockdata(crypto_symbols, test_head_start, test_end, timeframe, device)

        # Cache the loaded data
        os.makedirs(f"pkl/{name}",exist_ok=True)
        save_pickle(data,f'pkl/{name}/data.pkl')
        save_pickle(data_valid,f'pkl/{name}/data_valid.pkl')
        save_pickle(data_valid_withhead,f'pkl/{name}/data_valid_withhead.pkl')
        save_pickle(data_test,f'pkl/{name}/data_test.pkl')
        save_pickle(data_test_withhead,f'pkl/{name}/data_test_withhead.pkl')

    # Load/create data_all (full period)
    try:
        data_all = load_pickle(f'pkl/{name}/data_all.pkl')
    except:
        data_all = _load_crypto_stockdata(crypto_symbols, train_start, test_end, timeframe, device)
        save_pickle(data_all,f'pkl/{name}/data_all.pkl')

    return data_all,data,data_valid,data_valid_withhead,data_test,data_test_withhead,name


def _map_instruments_to_crypto(instruments):
    """
    Map traditional stock instrument identifiers to cryptocurrency symbols.

    Args:
        instruments: Stock market instrument identifier

    Returns:
        List of cryptocurrency trading pairs for gapless-crypto-data
    """
    if instruments is None:
        return ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']

    # Map common stock indices to representative crypto portfolios
    instrument_mapping = {
        'csi300': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],  # Top 3 crypto by market cap
        'csi500': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT'],  # Top 5 crypto
        'csi1000': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT',
                   'SOLUSDT', 'DOTUSDT', 'LINKUSDT'],  # Expanded crypto portfolio
        'spy': ['BTCUSDT', 'ETHUSDT'],  # US market proxy
        'qqq': ['ETHUSDT', 'ADAUSDT'],  # Tech proxy
    }

    # Direct crypto symbol mapping
    if isinstance(instruments, str):
        if instruments.lower() in instrument_mapping:
            return instrument_mapping[instruments.lower()]
        elif instruments.upper().endswith('USDT'):
            return [instruments.upper()]
        else:
            # Default mapping for unknown instruments
            return ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']

    return ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']  # Default portfolio


def _load_crypto_stockdata(symbols, start_date, end_date, timeframe, device):
    """
    Load cryptocurrency data and wrap in StockData interface for compatibility.

    Args:
        symbols: List of crypto symbols
        start_date: Start date string
        end_date: End date string
        timeframe: Data frequency ('1h' or '1d')
        device: PyTorch device

    Returns:
        StockData object with authentic crypto market data
    """
    from alphagen_qlib.stock_data import StockData

    # Create StockData with crypto data backend
    # The StockData constructor will use our CryptoDataProvider via _load_exprs_baostock
    stock_data = StockData(
        instrument=symbols,
        start_time=start_date,
        end_time=end_date,
        device=device,
        freq='day' if timeframe == '1d' else 'hour'
    )

    print(f"Loaded crypto StockData: {len(symbols)} symbols, "
          f"{start_date} to {end_date}, {stock_data.n_days} days")

    return stock_data
