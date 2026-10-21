"""Smoke demo: generate one signal for a real cointegrated-ish pair."""

from datetime import datetime, timedelta

from pairs_trading.data.fetcher import fetch_pair_data
from pairs_trading.data.schemas import Asset, Pair
from pairs_trading.signals import generate_signal


def main():
    pair = Pair(asset_a=Asset(symbol="KO"), asset_b=Asset(symbol="PEP"))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    data_a, data_b = fetch_pair_data(pair.asset_a, pair.asset_b, start_date, end_date)
    signal = generate_signal(pair, data_a, data_b)

    print(f"Pair:         {pair.pair_id}")
    print(f"As of:        {signal.timestamp.date()}")
    print(f"Z-score:      {signal.z_score:.3f}")
    print(f"Should enter: {signal.should_enter}")
    print(f"Side:         {signal.side.value}")


if __name__ == "__main__":
    main()
