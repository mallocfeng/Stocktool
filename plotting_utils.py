from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import numpy as np
import pandas as pd

from backtesting import BacktestResult


def plot_strategy_stats(results: List[Tuple[str, BacktestResult]]):
    if not results:
        return
    names = [name for name, _ in results]
    total_returns = [res.total_return * 100 for _, res in results]
    drawdowns = [res.max_drawdown * 100 for _, res in results]
    win_rates = [res.win_rate * 100 for _, res in results]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    x = np.arange(len(names))

    axes[0].bar(x, total_returns, color="#2ca02c")
    axes[0].set_ylabel("总收益率(%)")
    axes[0].set_title("策略表现对比")

    axes[1].bar(x, drawdowns, color="#d62728")
    axes[1].set_ylabel("最大回撤(%)")

    axes[2].bar(x, win_rates, color="#1f77b4")
    axes[2].set_ylabel("胜率(%)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    plt.show()


def plot_trade_chart(
    df: pd.DataFrame,
    result: BacktestResult,
    strategy_name: str,
    signal_buy: pd.Series | None = None,
    signal_sell: pd.Series | None = None,
):
    fig, (ax_price, ax_equity) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax_price.plot(df.index, df["close"], label="收盘价", color="black")

    if signal_buy is not None:
        mask = signal_buy.reindex(df.index).fillna(False)
        ax_price.scatter(df.index[mask], df.loc[mask, "close"], marker="^", color="#2ca02c", s=30, label="信号买入")

    if signal_sell is not None:
        mask = signal_sell.reindex(df.index).fillna(False)
        ax_price.scatter(df.index[mask], df.loc[mask, "close"], marker="v", color="#d62728", s=30, label="信号卖出")

    if result.trades:
        entry_dates = [t.entry_date for t in result.trades]
        entry_prices = [t.entry_price for t in result.trades]
        exit_dates = [t.exit_date for t in result.trades]
        exit_prices = [t.exit_price for t in result.trades]
        ax_price.scatter(entry_dates, entry_prices, marker="^", color="green", s=80, label="实际买入")
        ax_price.scatter(exit_dates, exit_prices, marker="v", color="red", s=80, label="实际卖出")

    ax_price.set_title(f"{strategy_name} 交易示意")
    ax_price.set_ylabel("价格")
    ax_price.grid(alpha=0.3)
    ax_price.legend(loc="upper left")

    ax_equity.plot(result.equity_curve.index, result.equity_curve.values, color="#1f77b4", label="权益曲线")
    ax_equity.set_ylabel("资金")
    ax_equity.set_xlabel("日期")
    ax_equity.legend(loc="upper left")
    ax_equity.grid(alpha=0.3)
    ax_equity.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


def plot_multi_timeframe(signals: Dict[str, Dict[str, pd.Series]], df_dict: Dict[str, pd.DataFrame]):
    rows = len(signals)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 3 * rows), sharex=False)
    if rows == 1:
        axes = [axes]
    for ax, (freq, sig) in zip(axes, signals.items()):
        df_freq = df_dict[freq]
        ax.plot(df_freq.index, df_freq["close"], label=f"{freq} 收盘")
        buy_mask = sig["buy"].reindex(df_freq.index).fillna(False)
        sell_mask = sig["sell"].reindex(df_freq.index).fillna(False)
        ax.scatter(df_freq.index[buy_mask], df_freq.loc[buy_mask, "close"], marker="^", color="green", s=40, label="买入")
        ax.scatter(df_freq.index[sell_mask], df_freq.loc[sell_mask, "close"], marker="v", color="red", s=40, label="卖出")
        ax.set_title(f"{freq} 周期信号")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


def plot_heatmap(heatmap_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    data = heatmap_df.T
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=0.2)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_xticks(range(0, len(heatmap_df.index), max(len(heatmap_df.index) // 10, 1)))
    ax.set_xticklabels(
        [heatmap_df.index[i].strftime("%Y-%m-%d") for i in range(0, len(heatmap_df.index), max(len(heatmap_df.index) // 10, 1))],
        rotation=45,
        ha="right",
    )
    ax.set_title("不同买入日期+持有天数收益热力图")
    plt.colorbar(im, ax=ax, label="收益率")
    fig.tight_layout()
    plt.show()
