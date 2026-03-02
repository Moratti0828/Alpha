import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd
import tushare as ts


def _today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def _days_ago_yyyymmdd(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")


class TushareClient:
    def __init__(self, token: Optional[str] = None):
        token = token or os.getenv("TUSHARE_TOKEN")

        if not token:
            # Fallback: try reading from tushare_token.txt in the same directory
            try:
                token_path = os.path.join(os.path.dirname(__file__), "tushare_token.txt")
                if os.path.exists(token_path):
                    with open(token_path, "r") as f:
                        token = f.read().strip()
            except Exception:
                pass

        if not token:
            raise RuntimeError("Missing TUSHARE_TOKEN env var or tushare_token.txt")
        ts.set_token(token)
        self.pro = ts.pro_api()

    def index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        # 返回字段：ts_code, trade_date, close, open, high, low, pre_close, change, pct_chg, vol, amount ...
        return self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    def shibor(self, start_date: str, end_date: str) -> pd.DataFrame:
        # 字段通常包含 date, on, 1w, 2w, 1m, 3m, 6m, 9m, 1y等（以返回为准）
        return self.pro.shibor(start_date=start_date, end_date=end_date)

    # 下面两个接口是否可用取决于 TuShare 权限/版本；你后续打开 enabled 再调
    def yield_curve(self, start_date: str, end_date: str) -> pd.DataFrame:
        return self.pro.yield_curve(start_date=start_date, end_date=end_date)

    def cn_cpi(self, start_m: str, end_m: str) -> pd.DataFrame:
        # CPI通常按月：YYYYMM
        return self.pro.cn_cpi(start_m=start_m, end_m=end_m)


def normalize_index_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d")
    out = out.sort_values("trade_date")
    return out


def calc_volatility(close: pd.Series, window: int = 20) -> float:
    # 日收益率标准差 * sqrt(252)
    r = close.pct_change().dropna()
    if len(r) < window:
        return float("nan")
    return float(r.tail(window).std() * (252 ** 0.5))


def calc_max_drawdown(close: pd.Series, window: int = 120) -> float:
    s = close.dropna()
    if len(s) < window:
        return float("nan")
    s = s.tail(window)
    cummax = s.cummax()
    dd = (s / cummax) - 1.0
    return float(dd.min())


if __name__ == "__main__":
    try:
        print("Initializing TushareClient...")
        client = TushareClient()
        # Print masked token to verify
        token = ts.get_token()
        if token:
            print(f"Current Token: {token[:6]}******{token[-6:]}")

        print("TushareClient initialized successfully.")

        start_date = _days_ago_yyyymmdd(10)
        end_date = _today_yyyymmdd()

        # Test 1: Stock Basic (Lowest barrier usually)
        print("\n--- Test 1: stock_basic (List of stocks) ---")
        try:
            df_basic = client.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date', limit=5)
            if df_basic is not None and not df_basic.empty:
                print("Success! stock_basic data:")
                print(df_basic.head(2))
            else:
                print("stock_basic returned empty.")
        except Exception as e:
            print(f"stock_basic failed: {e}")

        # Test 2: Trade Calendar
        print("\n--- Test 2: trade_cal (Trading Calendar) ---")
        try:
            df_cal = client.pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
            if df_cal is not None and not df_cal.empty:
                print("Success! trade_cal data:")
                print(df_cal.head(2))
            else:
                print("trade_cal returned empty.")
        except Exception as e:
            print(f"trade_cal failed: {e}")

        # Test 3: Index Daily (Target)
        print(f"\n--- Test 3: index_daily (000001.SH) from {start_date} to {end_date} ---")
        try:
            df = client.index_daily(ts_code="000001.SH", start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                print("Success! index_daily data:")
                print(df.head(2))
            else:
                print("No data fetched or empty DataFrame.")
        except Exception as e:
            print(f"index_daily failed: {e}")

    except Exception as e:
        print(f"Error during debug: {e}")
