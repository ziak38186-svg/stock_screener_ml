"""
backtest_engine.py
==================
Backtest Engineer — Nifty 50 Strategy Comparison

Files needed in the same folder (Google Colab or local):
    fused_dataset.csv              — from Role 4
    Nifty50_Master_Cleaned_Full.csv — from Role 1

How to run:
    python backtest_engine.py

Or in Google Colab:
    %run backtest_engine.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')           # works in Colab and headless environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
FUSED_PATH        = "fused_dataset.csv"
PRICE_PATH        = "Nifty50_Master_Cleaned_Full.csv"

INITIAL_CAPITAL   = 100_000   # Starting capital ₹
TRANSACTION_COST  = 0.001     # 0.1% per leg (entry + exit both charged)
HOLD_DAYS         = 5         # Hold each position for 5 trading days
POSITION_SIZE_PCT = 0.02      # Allocate 2% of capital per signal


# ─────────────────────────────────────────────────────────────
#  STEP 1 — LOAD & MERGE DATA
# ─────────────────────────────────────────────────────────────
def load_data():
    """
    Loads fused_dataset.csv (Role 4) and merges open/close prices
    from Nifty50_Master_Cleaned_Full.csv (Role 1).

    Role 4's file has no open/close — we need them for trade execution.
    Also fixes lowercase column names that Role 4 delivered.

    Returns: merged pd.DataFrame
    """
    print("=" * 55)
    print("  STEP 1 — LOADING DATA")
    print("=" * 55)

    # ── Fused dataset (Role 4) ────────────────────────────────
    fused = pd.read_csv(FUSED_PATH, parse_dates=["date"])

    # Fix lowercase column names from Role 4
    fused.rename(columns={
        'roc'            : 'ROC',
        'rsi'            : 'RSI',
        'ma_ratio'       : 'MA_ratio',
        'volatility'     : 'Volatility',
        'momentum_score' : 'Momentum_score',
        'sentiment_score': 'Sentiment_score',
        'buy'            : 'BUY'
    }, inplace=True)

    print(f"  Fused dataset  : {fused.shape[0]:,} rows | "
          f"{fused['stock'].nunique()} stocks | "
          f"{fused['date'].min().date()} to {fused['date'].max().date()}")

    # ── Price master (Role 1) ─────────────────────────────────
    price = pd.read_csv(PRICE_PATH, parse_dates=["date"])

    # Clean price columns in case commas are present
    for col in ["open", "high", "low", "close"]:
        if col in price.columns:
            price[col] = (
                price[col].astype(str)
                .str.replace(",", "", regex=False)
                .astype(float)
            )

    price = price[["date", "stock", "open", "close"]].copy()
    price = price.sort_values(["stock", "date"]).reset_index(drop=True)

    print(f"  Price master   : {price.shape[0]:,} rows | "
          f"{price['date'].min().date()} to {price['date'].max().date()}")

    # ── Merge: fused gets open/close from price master ────────
    df = fused.merge(price, on=["date", "stock"], how="left")
    df = df.sort_values(["date", "stock"]).reset_index(drop=True)

    missing = df["open"].isnull().sum()
    if missing > 0:
        print(f"  WARNING: {missing} rows missing price data — will be skipped")

    print(f"  Merged dataset : {df.shape[0]:,} rows | "
          f"Columns: {list(df.columns)}")
    print(f"  BUY=1 signals  : {df['BUY'].sum():,} ({df['BUY'].mean()*100:.1f}% of rows)")
    print()

    return df


# ─────────────────────────────────────────────────────────────
#  STEP 2 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
def split_data(df, train_pct=0.70):
    """
    Time-based split — NEVER random split on time-series data.
    First 70% of dates = train, last 30% = test.

    Returns: train_df, test_df, split_date
    """
    print("=" * 55)
    print("  STEP 2 — TRAIN / TEST SPLIT (70 / 30 time-based)")
    print("=" * 55)

    dates      = sorted(df["date"].unique())
    split_idx  = int(len(dates) * train_pct)
    split_date = dates[split_idx]

    train = df[df["date"] <  split_date].copy()
    test  = df[df["date"] >= split_date].copy()

    print(f"  Train : {train['date'].min().date()} to "
          f"{train['date'].max().date()} "
          f"({train['date'].nunique()} days | {len(train):,} rows | "
          f"BUY=1: {train['BUY'].mean()*100:.1f}%)")
    print(f"  Test  : {test['date'].min().date()} to "
          f"{test['date'].max().date()} "
          f"({test['date'].nunique()} days | {len(test):,} rows | "
          f"BUY=1: {test['BUY'].mean()*100:.1f}%)")
    print()

    return train, test, split_date


# ─────────────────────────────────────────────────────────────
#  STEP 3 — SIGNAL GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_signals(df, signal_col, threshold=0.0, is_binary=False):
    """
    Adds a 'signal' column (1 = BUY, 0 = hold).

    is_binary = True  → signal_col is already 0/1  (Role 4 BUY column)
    is_binary = False → buy when signal_col > threshold
    """
    df = df.copy()

    if is_binary:
        df["signal"] = df[signal_col].fillna(0).astype(int)
    else:
        df["signal"] = (df[signal_col].fillna(0) > threshold).astype(int)

    n = df["signal"].sum()
    print(f"  Signal col='{signal_col}' | threshold={threshold} | "
          f"BUY signals={n:,} ({n/len(df)*100:.1f}% of rows)")

    return df


# ─────────────────────────────────────────────────────────────
#  STEP 4 — CORE BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────
def run_backtest(
    df,
    signal_col,
    threshold         = 0.0,
    is_binary         = False,
    initial_capital   = INITIAL_CAPITAL,
    transaction_cost  = TRANSACTION_COST,
    hold_days         = HOLD_DAYS,
    position_size_pct = POSITION_SIZE_PCT,
    strategy_name     = "Strategy"
):
    """
    Core backtesting loop.

    Entry  : Signal fires on Day N → BUY at Day N+1 OPEN
    Exit   : Sell at CLOSE after hold_days trading days
    Cost   : transaction_cost charged on both entry and exit legs

    Returns dict with: equity_curve, trades, metrics, strategy_name
    """
    print(f"  Running : {strategy_name}")

    # Generate signals
    df = generate_signals(df, signal_col, threshold, is_binary)
    df = df.sort_values(["date", "stock"]).reset_index(drop=True)

    # Fast price lookup: (date, stock) → {open, close}
    df_valid     = df.dropna(subset=["open", "close"])
    price_lookup = (
        df_valid.set_index(["date", "stock"])[["open", "close"]]
        .to_dict("index")
    )

    trading_dates = sorted(df["date"].unique())
    date_index    = {d: i for i, d in enumerate(trading_dates)}

    # State
    cash          = float(initial_capital)
    positions     = {}       # {(entry_date, stock, entry_price, shares): exit_date}
    closed_trades = []
    equity_curve  = {}

    # ── Main loop ─────────────────────────────────────────────
    for i, today in enumerate(trading_dates):

        # EXIT: close positions that have matured
        to_remove = []
        for pos_key, exit_date in positions.items():
            entry_date, stock, entry_price, shares = pos_key
            if today >= exit_date:
                prices     = price_lookup.get((today, stock), {})
                exit_price = prices.get("close", entry_price)

                proceeds   = shares * exit_price * (1 - transaction_cost)
                cost_basis = shares * entry_price * (1 + transaction_cost)
                trade_pnl  = proceeds - cost_basis
                trade_ret  = trade_pnl / cost_basis

                cash += proceeds
                closed_trades.append({
                    "entry_date"  : entry_date.date(),
                    "exit_date"   : today.date(),
                    "stock"       : stock,
                    "entry_price" : round(float(entry_price), 2),
                    "exit_price"  : round(float(exit_price), 2),
                    "shares"      : round(shares, 4),
                    "pnl"         : round(trade_pnl, 2),
                    "return_pct"  : round(trade_ret * 100, 3),
                    "win"         : int(trade_pnl > 0)
                })
                to_remove.append(pos_key)

        for k in to_remove:
            del positions[k]

        # ENTRY: buy tomorrow's open if signal fires today
        if i + 1 < len(trading_dates):
            tomorrow      = trading_dates[i + 1]
            today_signals = df[
                (df["date"] == today) & (df["signal"] == 1)
            ]["stock"].tolist()

            for stock in today_signals:
                # Skip if already holding this stock
                if any(pk[1] == stock for pk in positions):
                    continue

                prices      = price_lookup.get((tomorrow, stock), {})
                entry_price = prices.get("open", None)

                if entry_price is None or entry_price <= 0:
                    continue

                alloc          = cash * position_size_pct
                if alloc < 100:
                    continue

                cost_per_share = entry_price * (1 + transaction_cost)
                shares         = alloc / cost_per_share
                total_cost     = shares * cost_per_share

                if total_cost > cash:
                    continue

                cash -= total_cost

                entry_idx = date_index.get(tomorrow, i + 1)
                exit_idx  = min(entry_idx + hold_days, len(trading_dates) - 1)
                exit_date = trading_dates[exit_idx]

                positions[(tomorrow, stock, entry_price, shares)] = exit_date

        # MARK-TO-MARKET: daily portfolio value
        open_value = sum(
            price_lookup.get((today, pk[1]), {}).get("close", pk[2]) * pk[3]
            for pk in positions
        )
        equity_curve[today] = cash + open_value

    # Force-close any remaining positions at end
    last_date = trading_dates[-1]
    for pos_key in list(positions.keys()):
        entry_date, stock, entry_price, shares = pos_key
        prices     = price_lookup.get((last_date, stock), {})
        exit_price = prices.get("close", entry_price)
        proceeds   = shares * exit_price * (1 - transaction_cost)
        cost_basis = shares * entry_price * (1 + transaction_cost)
        trade_pnl  = proceeds - cost_basis
        trade_ret  = trade_pnl / cost_basis
        cash      += proceeds
        closed_trades.append({
            "entry_date"  : entry_date.date(),
            "exit_date"   : last_date.date(),
            "stock"       : stock,
            "entry_price" : round(float(entry_price), 2),
            "exit_price"  : round(float(exit_price), 2),
            "shares"      : round(shares, 4),
            "pnl"         : round(trade_pnl, 2),
            "return_pct"  : round(trade_ret * 100, 3),
            "win"         : int(trade_pnl > 0)
        })

    equity_series = pd.Series(equity_curve).sort_index()
    trades_df     = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()
    metrics       = compute_metrics(equity_series, trades_df, initial_capital)

    print(f"  Trades executed : {len(trades_df)}")
    print(f"  Final capital   : ₹{equity_series.iloc[-1]:,.2f}")
    print()

    return {
        "equity_curve"  : equity_series,
        "trades"        : trades_df,
        "metrics"       : metrics,
        "strategy_name" : strategy_name
    }


# ─────────────────────────────────────────────────────────────
#  STEP 5 — METRICS CALCULATOR
# ─────────────────────────────────────────────────────────────
def compute_metrics(equity_series, trades_df, initial_capital=INITIAL_CAPITAL):
    """
    Computes CAGR, Sharpe, Max Drawdown, Win % from equity curve.
    """
    if len(equity_series) < 2:
        return {}

    daily_returns = equity_series.pct_change().dropna()
    end_val       = equity_series.iloc[-1]
    n_days        = (equity_series.index[-1] - equity_series.index[0]).days
    n_years       = max(n_days / 365.25, 0.01)

    cagr         = (end_val / initial_capital) ** (1 / n_years) - 1
    sharpe       = ((daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
                    if daily_returns.std() > 0 else 0.0)
    rolling_max  = equity_series.cummax()
    max_drawdown = ((equity_series - rolling_max) / rolling_max).min()

    total_trades = len(trades_df)
    win_pct      = trades_df["win"].mean() * 100 if total_trades > 0 else 0.0
    total_pnl    = trades_df["pnl"].sum()  if total_trades > 0 else 0.0

    return {
        "CAGR"           : round(cagr * 100, 2),
        "Sharpe"         : round(sharpe, 3),
        "Max_Drawdown"   : round(max_drawdown * 100, 2),
        "Win_Pct"        : round(win_pct, 1),
        "Total_Trades"   : total_trades,
        "Total_PnL"      : round(total_pnl, 2),
        "Final_Capital"  : round(float(end_val), 2),
        "Initial_Capital": round(initial_capital, 2)
    }


# ─────────────────────────────────────────────────────────────
#  STEP 6 — METRICS PRINTER
# ─────────────────────────────────────────────────────────────
def print_metrics(metrics, strategy_name="Strategy"):
    print("=" * 50)
    print(f"  {strategy_name}")
    print("=" * 50)
    print(f"  CAGR             : {metrics.get('CAGR', 0):>8.2f} %")
    print(f"  Sharpe Ratio     : {metrics.get('Sharpe', 0):>8.3f}")
    print(f"  Max Drawdown     : {metrics.get('Max_Drawdown', 0):>8.2f} %")
    print(f"  Win Rate         : {metrics.get('Win_Pct', 0):>8.1f} %")
    print(f"  Total Trades     : {metrics.get('Total_Trades', 0):>8d}")
    print(f"  Total PnL        : ₹{metrics.get('Total_PnL', 0):>12,.2f}")
    print(f"  Final Capital    : ₹{metrics.get('Final_Capital', 0):>12,.2f}")
    print(f"  Initial Capital  : ₹{metrics.get('Initial_Capital', 0):>12,.2f}")
    print("=" * 50)
    print()


# ─────────────────────────────────────────────────────────────
#  STEP 7 — EQUITY CURVE PLOTTER
# ─────────────────────────────────────────────────────────────
def plot_equity_curve(equity_series, title, color, save_path,
                      initial_capital=INITIAL_CAPITAL):
    """
    Plots equity curve (top panel) + drawdown (bottom panel).
    Saves as PNG automatically.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # Equity curve
    ax1.plot(equity_series.index, equity_series.values,
             color=color, linewidth=1.8, label=title)
    ax1.axhline(initial_capital, color="black", linestyle="--",
                linewidth=0.8, alpha=0.4,
                label=f"Initial ₹{initial_capital:,.0f}")
    ax1.fill_between(equity_series.index, equity_series.values,
                     initial_capital,
                     where=equity_series.values >= initial_capital,
                     alpha=0.12, color="green")
    ax1.fill_between(equity_series.index, equity_series.values,
                     initial_capital,
                     where=equity_series.values < initial_capital,
                     alpha=0.12, color="red")
    ax1.set_ylabel("Portfolio Value (₹)", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax1.grid(True, alpha=0.25)

    # Drawdown
    rolling_max = equity_series.cummax()
    drawdown    = (equity_series - rolling_max) / rolling_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color="red", alpha=0.35, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")
    print()


# ─────────────────────────────────────────────────────────────
#  STEP 8 — TRADE LOG SAVER
# ─────────────────────────────────────────────────────────────
def save_trade_log(trades_df, filename):
    if trades_df.empty:
        print(f"  No trades — {filename} not saved.\n")
        return
    trades_df.to_csv(filename, index=False)
    print(f"  Saved: {filename} ({len(trades_df)} trades)")
    print(trades_df.head(10).to_string(index=False))
    print()


# ─────────────────────────────────────────────────────────────
#  STEP 9 — STRATEGY COMPARISON
# ─────────────────────────────────────────────────────────────
def compare_strategies(results_dict, save_path="equity_comparison.png"):
    """
    Prints comparison table and saves combined equity curve plot.
    """
    print("=" * 70)
    print("  STRATEGY COMPARISON TABLE")
    print("=" * 70)

    rows = []
    for name, res in results_dict.items():
        m = res["metrics"]
        rows.append({
            "Strategy"      : name,
            "CAGR (%)"      : m.get("CAGR", 0),
            "Sharpe"        : m.get("Sharpe", 0),
            "Max DD (%)"    : m.get("Max_Drawdown", 0),
            "Win %"         : m.get("Win_Pct", 0),
            "Trades"        : m.get("Total_Trades", 0),
            "Final Cap (₹)" : f"₹{m.get('Final_Capital', 0):,.0f}"
        })

    compare_df = pd.DataFrame(rows).set_index("Strategy")
    print(compare_df.to_string())
    print("=" * 70)
    print()

    # Combined plot
    colors  = ["#1D9E75", "#378ADD", "#EF9F27"]
    fig, ax = plt.subplots(figsize=(14, 6))

    for (name, res), color in zip(results_dict.items(), colors):
        eq = res["equity_curve"]
        ax.plot(eq.index, eq.values, label=name, color=color, linewidth=1.8)

    ax.axhline(INITIAL_CAPITAL, color="black", linestyle="--",
               linewidth=0.8, alpha=0.4,
               label=f"Initial ₹{INITIAL_CAPITAL:,.0f}")
    ax.set_title("Strategy Comparison — All 3 Equity Curves",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value (₹)", fontsize=11)
    ax.set_xlabel("Date", fontsize=10)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")

    return compare_df


# ─────────────────────────────────────────────────────────────
#  MAIN — runs everything end to end
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("  BACKTEST ENGINE — NIFTY 50 STRATEGY COMPARISON")
    print("=" * 55 + "\n")

    # ── 1. Load & merge ───────────────────────────────────────
    df = load_data()

    # ── 2. Train / test split ─────────────────────────────────
    train_df, test_df, split_date = split_data(df, train_pct=0.70)

    # ── 3. Momentum Only ──────────────────────────────────────
    # threshold=0.5 → top ~30% of momentum signals
    # (your data: Momentum_score mean=0.038, std=0.806)
    print("=" * 55)
    print("  STRATEGY 1 — MOMENTUM ONLY")
    print("=" * 55)
    res_mom = run_backtest(
        test_df,
        signal_col    = "Momentum_score",
        threshold     = 0.5,
        is_binary     = False,
        strategy_name = "Momentum Only"
    )
    print_metrics(res_mom["metrics"], "Momentum Only")
    plot_equity_curve(
        res_mom["equity_curve"],
        title     = "Momentum Only Strategy",
        color     = "#1D9E75",
        save_path = "equity_curve_momentum.png"
    )
    save_trade_log(res_mom["trades"], "trade_log_momentum.csv")

    # ── 4. Sentiment Only ─────────────────────────────────────
    # threshold=0.1 because 96.7% of rows have sentiment=0
    # (no-news days filled with 0 — only actual news days matter)
    print("=" * 55)
    print("  STRATEGY 2 — SENTIMENT ONLY")
    print("=" * 55)
    res_sent = run_backtest(
        test_df,
        signal_col    = "Sentiment_score",
        threshold     = 0.1,
        is_binary     = False,
        strategy_name = "Sentiment Only"
    )
    print_metrics(res_sent["metrics"], "Sentiment Only")
    plot_equity_curve(
        res_sent["equity_curve"],
        title     = "Sentiment Only Strategy",
        color     = "#378ADD",
        save_path = "equity_curve_sentiment.png"
    )
    save_trade_log(res_sent["trades"], "trade_log_sentiment.csv")

    # ── 5. Hybrid ML ──────────────────────────────────────────
    # Uses Role 4's BUY column (Logistic Regression output)
    # is_binary=True because BUY is already 0 or 1
    print("=" * 55)
    print("  STRATEGY 3 — HYBRID ML")
    print("=" * 55)
    res_hybrid = run_backtest(
        test_df,
        signal_col    = "BUY",
        threshold     = 0.0,
        is_binary     = True,
        strategy_name = "Hybrid ML"
    )
    print_metrics(res_hybrid["metrics"], "Hybrid ML")
    plot_equity_curve(
        res_hybrid["equity_curve"],
        title     = "Hybrid ML Strategy",
        color     = "#EF9F27",
        save_path = "equity_curve_hybrid.png"
    )
    save_trade_log(res_hybrid["trades"], "trade_log_hybrid.csv")

    # ── 6. Compare all 3 ─────────────────────────────────────
    print("=" * 55)
    print("  FINAL — COMPARING ALL 3 STRATEGIES")
    print("=" * 55)
    compare_strategies(
        {
            "Momentum Only" : res_mom,
            "Sentiment Only": res_sent,
            "Hybrid ML"     : res_hybrid
        },
        save_path = "equity_comparison.png"
    )

    print("\n" + "=" * 55)
    print("  ALL DONE — FILES SAVED:")
    print("=" * 55)
    print("  equity_curve_momentum.png")
    print("  equity_curve_sentiment.png")
    print("  equity_curve_hybrid.png")
    print("  equity_comparison.png")
    print("  trade_log_momentum.csv")
    print("  trade_log_sentiment.csv")
    print("  trade_log_hybrid.csv")
    print("=" * 55)
