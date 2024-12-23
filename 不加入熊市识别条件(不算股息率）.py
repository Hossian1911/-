import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from arch import arch_model  # 用于 GARCH 模型

# 设置 Matplotlib 中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

def read_excel_file(file_path):
    """
    读取 Excel 文件并显示内容
    """
    try:
        df = pd.read_excel(file_path)
        print("Excel 文件内容（前五行）：")
        print(df.head())  # 打印前五行以检查数据格式
        print("列名：", df.columns)  # 打印列名
        return df
    except Exception as e:
        print(f"读取 Excel 文件失败: {e}")
        return None

def calculate_dynamic_trend(data, column, window_high=5, window_low=20):
    """
    根据波动率状态动态计算趋势：高波动计算短期趋势（5日），低波动计算长期趋势（20日）
    """
    trends = []
    for i in range(len(data)):
        if data.loc[i, 'volatility_label'] == '高波动' and i >= window_high:
            y = data[column].iloc[i - window_high:i].values
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            trends.append(slope)
        elif data.loc[i, 'volatility_label'] == '低波动' and i >= window_low:
            y = data[column].iloc[i - window_low:i].values
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            trends.append(slope)
        else:
            trends.append(np.nan)
    return trends

def calculate_garch_volatility(data, column):
    """
    使用 GARCH 模型计算条件波动率并检测高波动与低波动时期
    """
    returns = data[column].pct_change().dropna() * 100
    am = arch_model(returns, vol='Garch', p=1, q=1)
    res = am.fit(disp="off")

    data.loc[1:, 'conditional_volatility'] = res.conditional_volatility

    high_vol_threshold = np.percentile(data['conditional_volatility'].dropna(), 90)
    data['volatility_label'] = data['conditional_volatility'].apply(
        lambda x: '高波动' if x > high_vol_threshold else '低波动' if pd.notna(x) else np.nan
    )

    return data

def backtest_strategy(data, index_column, bond_etf_column, dynamic_trend, initial_cash=10000, transaction_fee=0.0003):
    """
    回测逻辑：基于动态趋势进行买卖操作，加入国债 ETF 的交易逻辑，并加入交易手续费。
    """
    cash = initial_cash
    holdings_equity = 0
    holdings_bond = 0
    daily_portfolio = []
    position = None
    last_trend = None

    for i in range(len(data) - 1):
        price_equity_today = data[index_column].iloc[i]
        price_equity_next_day = data[index_column].iloc[i + 1]
        price_bond_today = data[bond_etf_column].iloc[i]
        price_bond_next_day = data[bond_etf_column].iloc[i + 1]

        current_trend = dynamic_trend[i]

        # 买入中证红利指数
        if position is None and current_trend < 0 and last_trend != "down":
            if not np.isnan(price_equity_next_day):
                holdings_equity = (cash * (1 - transaction_fee)) / price_equity_next_day
                cash = 0
                position = "equity"
                last_trend = "down"
                print(f"买入中证红利指数: 日期={data['日期'].iloc[i + 1]}, 价格={price_equity_next_day:.2f}, 持仓={holdings_equity:.2f}")

        # 卖出中证红利指数，买入国债 ETF
        elif position == "equity" and current_trend > 0 and last_trend != "up":
            if not np.isnan(price_equity_next_day) and not np.isnan(price_bond_next_day):
                cash = holdings_equity * price_equity_next_day * (1 - transaction_fee)
                holdings_equity = 0
                holdings_bond = (cash * (1 - transaction_fee)) / price_bond_next_day
                cash = 0
                position = "bond"
                last_trend = "up"
                print(f"卖出中证红利指数，买入国债 ETF: 日期={data['日期'].iloc[i + 1]}, 中证红利价格={price_equity_next_day:.2f}, 国债 ETF 价格={price_bond_next_day:.2f}, 持仓={holdings_bond:.2f}")

        # 卖出国债 ETF，买入中证红利指数
        elif position == "bond" and current_trend < 0 and last_trend != "down":
            if not np.isnan(price_bond_next_day) and not np.isnan(price_equity_next_day):
                cash = holdings_bond * price_bond_next_day * (1 - transaction_fee)
                holdings_bond = 0
                holdings_equity = (cash * (1 - transaction_fee)) / price_equity_next_day
                cash = 0
                position = "equity"
                last_trend = "down"
                print(f"卖出国债 ETF，买入中证红利指数: 日期={data['日期'].iloc[i + 1]}, 国债 ETF 价格={price_bond_next_day:.2f}, 中证红利价格={price_equity_next_day:.2f}, 持仓={holdings_equity:.2f}")

        # 记录每日组合价值
        portfolio_value = cash
        if holdings_equity > 0:
            portfolio_value += holdings_equity * price_equity_today
        if holdings_bond > 0 and not np.isnan(price_bond_today):
            portfolio_value += holdings_bond * price_bond_today

        daily_portfolio.append(portfolio_value)

    # 计算最终组合价值
    portfolio_value = cash
    if holdings_equity > 0:
        portfolio_value += holdings_equity * data[index_column].iloc[-1]
    if holdings_bond > 0 and not np.isnan(data[bond_etf_column].iloc[-1]):
        portfolio_value += holdings_bond * data[bond_etf_column].iloc[-1]
    daily_portfolio.append(portfolio_value)

    total_return = (portfolio_value - initial_cash) / initial_cash

    daily_returns = np.diff(daily_portfolio) / daily_portfolio[:-1]
    mean_daily_return = np.mean(daily_returns)
    std_daily_return = np.std(daily_returns)

    risk_free_rate = 0.0
    sharpe_ratio = (mean_daily_return * 252) / (std_daily_return * np.sqrt(252)) if std_daily_return > 0 else np.nan

    return portfolio_value, total_return, daily_portfolio, daily_returns, sharpe_ratio

# 读取 Excel 文件并执行回测
file_path = "C:/Users/Think/Desktop/股息率与国债收益率1.xlsx"
data = read_excel_file(file_path)

if data is not None:
    x_column = '日期'
    y1_column = '股息率市值加权'
    y2_column = '国债收益率'
    index_column = '点位'
    bond_etf_column = '国债etf价格'
    hs300_column = '沪深300'  # 新增沪深300列

    data = data.sort_values(by=x_column, ascending=True).reset_index(drop=True)

    data['股息率减国债收益率'] = data[y1_column] - data[y2_column]

    data = calculate_garch_volatility(data, index_column)

    dynamic_trend = calculate_dynamic_trend(data, '股息率减国债收益率', window_high=5, window_low=10)

    final_value, total_return, daily_portfolio, daily_returns, sharpe_ratio = backtest_strategy(
        data, index_column, bond_etf_column, dynamic_trend
    )

    print(f"最终资金: {final_value:.2f} 元")
    print(f"总收益率: {total_return:.2%}")
    print(f"夏普率: {sharpe_ratio:.2f}")

    # 绘制资金曲线并添加沪深300数据
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左轴：资金曲线
    ax1.plot(data[x_column], daily_portfolio, label='资金曲线', color='blue')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('资金价值', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # 右轴：沪深300
    ax2 = ax1.twinx()
    ax2.plot(data[x_column], data[hs300_column], label='沪深300', color='green')
    ax2.set_ylabel('沪深300指数', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # 图例
    fig.tight_layout()
    plt.title('资金曲线与沪深300')
    plt.show()