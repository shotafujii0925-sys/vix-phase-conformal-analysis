import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def fetch_and_prepare_data(period="3mo"):
    """
    指定期間の各種金融データを取得し、分析用のデータフレームを作成する
    """
    # 取得するティッカーシンボル
    # TQQQ: 3倍ナスダック, SPY: S&P500, ^TNX: 米10年債利回り, TLT: 米長期債ETF, ^VIX: 恐怖指数
    # ^NDX: ナスダック100, ^GSPC: S&P500指数
    tickers = ["TQQQ", "SPY", "^TNX", "TLT", "^VIX", "^NDX", "^GSPC"]
    
    print(f"データ取得中... (期間: {period})")
    data = yf.download(tickers, period=period)['Close']
    
    # 欠損値の補完（前日終値で埋める）
    data = data.ffill()
    
    # 分析用データフレームの構築
    df = pd.DataFrame()
    df['TQQQ'] = data['TQQQ']
    df['SPY'] = data['SPY']
    df['VIX'] = data['^VIX']
    df['US10Y_Yield'] = data['^TNX'] # 金利
    df['TLT_Bond'] = data['TLT']     # 債券価格
    
    # NDX/SPX 比率の計算（ナスダックがS&P500に対してどれだけ強いか）
    df['NDX_SPX_Ratio'] = data['^NDX'] / data['^GSPC']
    
    # TQQQがSPYに勝っているか（アウトパフォーム）の差分を計算
    # 基準日（初日）を100とした際の変化率で比較
    df['TQQQ_Return'] = df['TQQQ'] / df['TQQQ'].iloc[0] * 100
    df['SPY_Return'] = df['SPY'] / df['SPY'].iloc[0] * 100
    df['TQQQ_Excess_Return'] = df['TQQQ_Return'] - df['SPY_Return']
    
    return df

def plot_correlation_and_trends(df):
    """
    取得したデータの相関関係とトレンドを視覚化する
    """
    # 1. 変化率（日次リターン）の計算
    daily_returns = df[['TQQQ', 'SPY', 'VIX', 'US10Y_Yield', 'TLT_Bond', 'NDX_SPX_Ratio']].pct_change().dropna()
    
    # 描画設定
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(15, 10))
    
    # グラフ1: TQQQ vs SPY のパフォーマンス比較（初期=100）
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(df.index, df['TQQQ_Return'], label='TQQQ (3x)', color='blue')
    ax1.plot(df.index, df['SPY_Return'], label='SPY (1x)', color='orange')
    ax1.set_title('TQQQ vs SPY Performance (Base 100)')
    ax1.legend()
    
    # グラフ2: NDX/SPX比率とVIXの推移
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(df.index, df['NDX_SPX_Ratio'], label='NDX/SPX Ratio', color='purple')
    ax2.set_ylabel('Ratio')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df.index, df['VIX'], label='VIX', color='red', alpha=0.5)
    ax2_twin.set_ylabel('VIX Index')
    ax2.set_title('Market Trend (NDX/SPX) & Volatility (VIX)')
    
    # グラフ3: 日次リターンの相関ヒートマップ
    ax3 = fig.add_subplot(2, 2, (3, 4))
    correlation_matrix = daily_returns.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
    ax3.set_title('Daily Returns Correlation Matrix')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 直近3ヶ月のデータを取得して実行
    # ※特定の期間（例：2023年の特定の3ヶ月）を指定する場合は、
    # yf.download(tickers, start="2023-01-01", end="2023-03-31") のように変更します。
    df_analyzed = fetch_and_prepare_data(period="3mo")
    
    print("\n--- データの先頭5行 ---")
    print(df_analyzed.head())
    
    plot_correlation_and_trends(df_analyzed)
    import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats

def analyze_statistical_significance(period="3mo"):
    # データ取得
    tickers = ["TQQQ", "SPY"]
    data = yf.download(tickers, period=period)['Close'].ffill()
    
    # 日次リターンの計算
    returns = data.pct_change().dropna()
    
    # 1. 基本統計量の計算
    stats_dict = {}
    for col in tickers:
        daily_mean = returns[col].mean()
        daily_std = returns[col].std()
        # 年率換算 (252営業日)
        ann_return = daily_mean * 252
        ann_vol = daily_std * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        
        # 最大下落率 (Max Drawdown)
        cumulative = (1 + returns[col]).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        
        stats_dict[col] = {
            "年率リターン": ann_return,
            "年率ボラティリティ": ann_vol,
            "シャープレシオ": sharpe,
            "最大下落率": max_dd
        }
    
    # 2. 統計的検定 (対応のあるt検定)
    # 帰無仮説 H0: TQQQのリターン平均 = SPYのリターン平均
    t_stat, p_value = stats.ttest_rel(returns['TQQQ'], returns['SPY'])
    
    # 結果の表示
    res_df = pd.DataFrame(stats_dict).T
    print("--- 投資指標比較 ---")
    print(res_df)
    print(f"\n--- 統計検定 (t検定) ---")
    print(f"t値: {t_stat:.4f}")
    print(f"p値: {p_value:.4f}")
    
    if p_value > 0.05:
        print("\n【結論】p値が0.05を超えているため、TQQQのリターンがSPYを上回っていることに統計的有意差は認められません。")
        print("つまり、リターンの差は偶然（ボラティリティの範疇）である可能性が高いと言えます。")
    else:
        print("\n【結論】統計的有意差が認められました（ただし、リスク調整後のシャープレシオを確認する必要があります）。")

if __name__ == "__main__":
    analyze_statistical_significance(period="3mo")