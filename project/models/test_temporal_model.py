import os
import pandas as pd
import numpy as np
from temporal_model import build_sequences


def write_test_csvs(tmpdir):
    # market: 10 days
    market = pd.DataFrame({
        'day': list(range(10)),
        'mkt_ret': np.linspace(0, 0.01, 10),
        'mkt_vol': np.linspace(0.1, 0.2, 10)
    })
    market_path = os.path.join(tmpdir, 'market.csv')
    market.to_csv(market_path, index=False)

    # transactions: one user with missing day (day 3 missing)
    tx = []
    for d in [0,1,2,4,5,6,7,8,9]:
        # alternate actions
        act = 1 if d % 3 == 0 else 0
        tx.append([0, d, market.loc[market['day']==d,'mkt_ret'].values[0], market.loc[market['day']==d,'mkt_vol'].values[0], act])
    tx = pd.DataFrame(tx, columns=['user_id','day','mkt_ret','mkt_vol','action'])
    tx_path = os.path.join(tmpdir, 'transactions.csv')
    tx.to_csv(tx_path, index=False)
    return tx_path, market_path


def test_build_sequences_basic(tmp_path):
    tx_path, market_path = write_test_csvs(str(tmp_path))
    X, y, feat_cols = build_sequences(tx_path, market_path, seq_len=3, max_samples=100, add_rolling=True, rolling_windows=(3,), binary=False)
    # Should produce at least one sample
    assert X.shape[0] > 0
    # check feature dimension: action, mkt_ret, mkt_vol, mkt_ret_roll_3, mkt_vol_roll_3 => 5
    assert X.shape[2] == 5
    # labels length matches samples
    assert X.shape[0] == y.shape[0]


def test_build_sequences_binary(tmp_path):
    tx_path, market_path = write_test_csvs(str(tmp_path))
    X, y, feat_cols = build_sequences(tx_path, market_path, seq_len=3, max_samples=100, add_rolling=False, binary=True)
    assert X.shape[0] > 0
    # binary labels should be 0 or 1
    assert set(np.unique(y)).issubset({0,1})
