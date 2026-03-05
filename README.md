# Regime-Aware Pairs Trading

**Status:** Active Development  
**Goal:** Out-of-sample Sharpe Ratio > 2.0

## 📖 Overview

Traditional pairs trading relies on the assumption of stable mean reversion. However, this assumption frequently breaks down during regime shifts caused by macro shocks, earnings announcements, or volatility bursts.

This project implements a **Regime-Aware Pairs Trading System**. It utilizes a "Deep Gating" mechanism—a compact deep-learning model—to determine when classical mean-reversion signals are reliable. By filtering out false positives during unstable regimes, the system aims to significantly improve drawdown stability and risk-adjusted returns.

## 🧠 Core Architecture

The system operates on a hybrid pipeline combining classical econometrics with modern deep learning:

1.  **Signal Generation (ARIMA):** Forecasts the short-term direction of the spread's reversion.
2.  **Risk Management (GARCH):** Forecasts volatility to dynamically adjust position sizing.
3.  **Regime Gating (Transformer/LSTM):** A "Tiny Transformer" analyzes technical features, event data, and (optionally) news sentiment to output a **Regime Score**.
4.  **Execution Logic:** Trades are executed only when:
    * ARIMA predicts reversion.
    * The Transformer confirms a favorable regime.
    * Sizing is inversely weighted by GARCH volatility.

## 🚀 Key Features

* **Hybrid Modeling:** Bridges the gap between statistical arbitrage (Cointegration) and Deep Learning.
* **False Positive Filtering:** Specifically targets the reduction of "whipsaw" losses during market stress.
* **Reproducible Pipeline:** complete workflow from data ingestion to backtesting.
* **Scalability:** Includes optional clustering algorithms to handle large universes of equity pairs.

## 🗺️ Project Roadmap

| Phase | Component | Status | Description |
| :--- | :--- | :--- | :--- |
| **1** | **Data & Preprocessing** | ⬜ | Identify cointegration pairs, compute z-scores, generate ARIMA & GARCH forecasts. |
| **2** | **Label Generation** | ⬜ | Simulate reversion trades to create "Profit/Loss" labels for the deep learning model. |
| **3** | **Regime Gate Training** | ⬜ | Train the LSTM/Transformer using Walk-Forward Cross-Validation. |
| **4** | **Backtesting** | ⬜ | Integrate signals and gating logic to simulate performance. |
| **5** | **News Augmentation** | ⬜ | *(Optional)* Incorporate sentiment analysis to detect regime changes earlier. |
| **6** | **Pair Filtering** | ⬜ | *(Optional)* Use clustering to optimize candidate selection for large equity datasets. |
| **7** | **Reporting** | ⬜ | Final evaluation, visualization of gate activation, and research paper generation. |

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/regime-aware-pairs-trading.git](https://github.com/yourusername/regime-aware-pairs-trading.git)

# Install dependencies
pip install -r requirements.txt
