import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from arch import arch_model  # 用于 GARCH 模型

# 设置 Matplotlib 中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 参数设置（便于调节）
GARCH_THRESHOLD_PERCENTILE = 90  # GARCH 高波动阈值百分位
SHORT_TERM_TREND_WINDOW = 5  # 短期趋势对应的天数
LONG_TERM_TREND_WINDOW = 10  # 长期趋势对应的天数
HS300_TREND_WINDOW = 65  # 沪深300趋势计算用的天数
dividend_rate =0.03 #每年股息率

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

def calculate_trend(data, column, window):
    """
    使用线性回归方法计算趋势（指定窗口斜率）
    """
    trends = []
    for i in range(len(data)):
        if i >= window:
            y = data[column].iloc[i - window:i].values
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)  # 线性回归计算斜率
            trends.append(slope)
        else:
            trends.append(np.nan)  # 前期不足window天的数据填充为NaN
    return trends

def calculate_garch_volatility(data, column, threshold_percentile):
    """
    使用 GARCH 模型计算条件波动率并检测高波动与低波动时期
    """
    returns = data[column].pct_change().dropna() * 100
    am = arch_model(returns, vol='Garch', p=1, q=1)
    res = am.fit(disp="off")

    data.loc[1:, 'conditional_volatility'] = res.conditional_volatility

    high_vol_threshold = np.percentile(data['conditional_volatility'].dropna(), threshold_percentile)
    data['volatility_label'] = data['conditional_volatility'].apply(
        lambda x: '高波动' if x > high_vol_threshold else '低波动' if pd.notna(x) else np.nan
    )

    return data

def calculate_max_drawdown(portfolio):
    """
    计算最大回撤
    """
    cumulative_max = np.maximum.accumulate(portfolio)
    drawdowns = (portfolio - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()
    return max_drawdown

def backtest_strategy(data, index_column, bond_etf_column, hs300_column, dynamic_trend, hs300_trend, initial_cash=10000, transaction_fee=0.0003):
    """
    回测逻辑：基于动态趋势进行买卖操作，加入国债 ETF 的交易逻辑，并加入每年最后一个交易日自动分红的逻辑。
    """
    cash = initial_cash
    holdings_equity = 0
    holdings_bond = 0
    daily_portfolio = []
    position = None
    last_hs300_trend = None  # 上次沪深300的趋势
    last_trend = None  # 上次红利指数的趋势

    # 确定每年最后一个交易日
    data['year'] = pd.to_datetime(data['日期']).dt.year
    last_days = data.groupby('year').tail(1).index  # 每年最后一个交易日的索引

    for i in range(len(data)):
        price_equity_today = data[index_column].iloc[i]
        price_bond_today = data[bond_etf_column].iloc[i]
        hs300_current_trend = hs300_trend[i]  # 获取沪深300的趋势
        current_volatility = data['volatility_label'].iloc[i]  # 当前的波动率状态

        # 判断沪深300趋势变化
        if last_hs300_trend is not None:
            # 沪深300从平稳或上升转为下降
            if last_hs300_trend >= 0 and hs300_current_trend < 0:
                # 卖出红利指数，买入国债（全仓国债）
                if position != "bond":
                    cash = holdings_equity * price_equity_today * (1 - transaction_fee) if position == "equity" else cash
                    holdings_equity = 0
                    holdings_bond = (cash * (1 - transaction_fee)) / price_bond_today
                    cash = 0
                    position = "bond"

        # 沪深300趋势为上升或平稳，正常轮动逻辑
        if hs300_current_trend >= 0:
            # 根据波动率判断趋势窗口
            if current_volatility == '高波动':
                window = SHORT_TERM_TREND_WINDOW
            elif current_volatility == '低波动':
                window = LONG_TERM_TREND_WINDOW
            else:
                window = None  # 如果波动率无法判定，不进行操作

            # 计算趋势
            if window is not None and i >= window:
                y = data['股息率减国债收益率'].iloc[i - window:i].values
                x = np.arange(len(y))
                current_trend, _ = np.polyfit(x, y, 1)  # 当前趋势

                # 买入中证红利指数
                if position != "equity" and current_trend < 0 and last_trend != "down":
                    cash = holdings_bond * price_bond_today * (1 - transaction_fee) if position == "bond" else cash
                    holdings_bond = 0
                    holdings_equity = (cash * (1 - transaction_fee)) / price_equity_today
                    cash = 0
                    position = "equity"
                    last_trend = "down"

                # 卖出中证红利指数，买入国债 ETF
                elif position == "equity" and current_trend > 0 and last_trend != "up":
                    cash = holdings_equity * price_equity_today * (1 - transaction_fee)
                    holdings_equity = 0
                    holdings_bond = (cash * (1 - transaction_fee)) / price_bond_today
                    cash = 0
                    position = "bond"
                    last_trend = "up"

        # 每年最后一个交易日增加5%收益到现金
        if i in last_days:
            portfolio_value = cash + holdings_equity * price_equity_today + holdings_bond * price_bond_today
            cash += portfolio_value * dividend_rate  # 增加5%的收益

        # 更新沪深300的趋势记录
        last_hs300_trend = hs300_current_trend

        # 记录每日组合价值
        portfolio_value = cash
        if holdings_equity > 0:
            portfolio_value += holdings_equity * price_equity_today
        if holdings_bond > 0:
            portfolio_value += holdings_bond * price_bond_today
        daily_portfolio.append(portfolio_value)

    # 计算夏普率
    daily_returns = np.diff(daily_portfolio) / daily_portfolio[:-1]
    mean_daily_return = np.mean(daily_returns)
    std_daily_return = np.std(daily_returns)
    sharpe_ratio = (mean_daily_return * 252) / (std_daily_return * np.sqrt(252)) if std_daily_return > 0 else np.nan

    # 计算最大回撤
    max_drawdown = calculate_max_drawdown(np.array(daily_portfolio))

    total_return = (portfolio_value - initial_cash) / initial_cash
    return portfolio_value, total_return, daily_portfolio, sharpe_ratio, max_drawdown

# 读取 Excel 文件并执行回测
file_path = "C:/Users/Think/Desktop/股息率与国债收益率1.xlsx"
data = read_excel_file(file_path)

if data is not None:
    x_column = '日期'
    y1_column = '股息率市值加权'
    y2_column = '国债收益率'
    index_column = '点位'
    bond_etf_column = '国债etf价格'
    hs300_column = '沪深300'

    data = data.sort_values(by=x_column, ascending=True).reset_index(drop=True)

    data['股息率减国债收益率'] = data[y1_column] - data[y2_column]

    data = calculate_garch_volatility(data, index_column, GARCH_THRESHOLD_PERCENTILE)

    dynamic_trend = calculate_trend(data, '股息率减国债收益率', window=SHORT_TERM_TREND_WINDOW)
    hs300_trend = calculate_trend(data, hs300_column, window=HS300_TREND_WINDOW)

    final_value, total_return, daily_portfolio, sharpe_ratio, max_drawdown = backtest_strategy(
        data, index_column, bond_etf_column, hs300_column, dynamic_trend, hs300_trend
    )

    print(f"最终资金: {final_value:.2f} 元")
    print(f"总收益率: {total_return:.2%}")
    print(f"夏普率: {sharpe_ratio:.2f}")
    print(f"最大回撤: {max_drawdown:.2%}")

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




