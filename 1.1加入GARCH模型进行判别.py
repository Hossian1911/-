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
    :param file_path: Excel 文件路径
    """
    try:
        # 读取 Excel 文件
        df = pd.read_excel(file_path)
        print("Excel 文件内容（前五行）：")
        print(df.head())  # 打印前五行以检查数据格式
        print("列名：", df.columns)  # 打印列名
        return df
    except Exception as e:
        print(f"读取 Excel 文件失败: {e}")
        return None

def calculate_trend_for_each_point(data, column, window=5):
    """
    为每一个数据点计算其前 window 日的趋势
    :param data: DataFrame 数据
    :param column: 待分析的列名
    :param window: 窗口大小，默认为 5 日
    :return: 包含趋势斜率的列表
    """
    slopes = []

    for i in range(len(data)):
        if i < window:
            # 如果数据不足 window 大小，无法计算，标记为 NaN
            slopes.append(np.nan)
        else:
            # 取前 window 天的数据
            y = data[column].iloc[i - window:i].values
            x = np.arange(len(y))
            # 使用 numpy 线性回归拟合
            slope, _ = np.polyfit(x, y, 1)
            slopes.append(slope)

    return slopes

def calculate_garch_volatility(data, column):
    """
    使用 GARCH 模型计算条件波动率并检测高波动与低波动时期
    :param data: DataFrame 数据
    :param column: 待分析的列名（如中证红利指数点位）
    :return: 更新后的 DataFrame，包括条件波动率和波动性分类
    """
    # 拟合 GARCH 模型
    returns = data[column].pct_change().dropna() * 100  # 计算日收益率（百分比形式）
    am = arch_model(returns, vol='Garch', p=1, q=1)
    res = am.fit(disp="off")

    # 获取条件波动率
    data.loc[1:, 'conditional_volatility'] = res.conditional_volatility

    # 计算高波动与低波动阈值
    high_vol_threshold = np.percentile(data['conditional_volatility'].dropna(), 90)  # 90%分位数
    data['volatility_label'] = data['conditional_volatility'].apply(
        lambda x: '高波动' if x > high_vol_threshold else '低波动' if pd.notna(x) else np.nan
    )

    return data

def plot_data_with_dynamic_trends(df, x_column, diff_column, index_column, trend_5, trend_20):
    """
    绘制股息率减国债收益率与重新计算的点位，同时动态标注前 5 日或前 20 日的趋势
    :param df: DataFrame 数据
    :param x_column: 日期列
    :param diff_column: 股息率减国债收益率列名
    :param index_column: 中证红利指数点位列名
    :param trend_5: 前 5 日趋势斜率列表
    :param trend_20: 前 20 日趋势斜率列表
    """
    try:
        # 确保日期列为日期格式
        if not pd.api.types.is_datetime64_any_dtype(df[x_column]):
            df[x_column] = pd.to_datetime(df[x_column])

        # 绘图
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 绘制股息率减国债收益率（左纵坐标）
        ax1.plot(df[x_column], df[diff_column], label='股息率减国债收益率', marker='o', color='r')
        ax1.set_xlabel("日期", fontsize=12)
        ax1.set_ylabel("股息率减国债收益率（%）", fontsize=12, color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True)

        # 动态标注趋势（根据波动率状态选择前 5 日或前 20 日趋势）
        for i in range(len(df)):
            if df.loc[i, 'volatility_label'] == '高波动' and not np.isnan(trend_5[i]):
                trend_direction = "增" if trend_5[i] > 0 else "减" if trend_5[i] < 0 else "平稳"
                ax1.text(df[x_column].iloc[i], df[diff_column].iloc[i],
                         f"{trend_direction}(5日)",
                         fontsize=8, color='purple', verticalalignment='bottom', horizontalalignment='right')
            elif df.loc[i, 'volatility_label'] == '低波动' and not np.isnan(trend_20[i]):
                trend_direction = "增" if trend_20[i] > 0 else "减" if trend_20[i] < 0 else "平稳"
                ax1.text(df[x_column].iloc[i], df[diff_column].iloc[i],
                         f"{trend_direction}(20日)",
                         fontsize=8, color='blue', verticalalignment='bottom', horizontalalignment='right')

        # 创建右纵坐标并绘制点位数据
        ax2 = ax1.twinx()
        # 绘制中证红利指数点位
        df['adjusted_index'] = df[index_column]  # 重新计算点位数据
        ax2.plot(df[x_column], df['adjusted_index'], label='点位', marker='^', color='g', linestyle='--')
        ax2.set_ylabel("中证红利指数点位（调整后）", fontsize=12, color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        # 绘制波动率曲线（橙色）
        ax2.plot(df[x_column], df['conditional_volatility'], label='波动率', color='orange', linestyle='-')

        # 在点位旁边标注高波动和低波动状态（橙色字体）
        for i in range(len(df)):
            if pd.notna(df.loc[i, 'volatility_label']):
                ax2.text(df[x_column].iloc[i], df['adjusted_index'].iloc[i],
                         f"{df.loc[i, 'volatility_label']}",
                         fontsize=8, color='orange', verticalalignment='bottom', horizontalalignment='right')

        ax2.legend(loc='upper right', fontsize=12)

        # 添加标题和布局
        plt.title(f"股息率减国债收益率与中证红利指数点位对比（动态趋势标注）", fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 显示图表
        plt.show()
    except Exception as e:
        print(f"绘图失败: {e}")

# 示例：读取 Excel 文件并绘图
file_path = "C:/Users/Think/Desktop/股息率与国债收益率.xls"  # 包含指数点位的文件路径
data = read_excel_file(file_path)

if data is not None:
    # 假设 Excel 文件中有以下列：'日期'、'股息率市值加权'、'国债收益率'、'中证红利指数点位'
    x_column = '日期'  # 日期列
    y1_column = '股息率市值加权'  # 股息率列
    y2_column = '国债收益率'  # 国债收益率列
    index_column = '点位'  # 中证红利指数点位列

    # 按时间升序排序
    data = data.sort_values(by=x_column, ascending=True).reset_index(drop=True)

    # 创建新列：股息率减国债收益率
    data['股息率减国债收益率'] = data[y1_column] - data[y2_column]

    # 计算每个点的短期趋势（5 日和 20 日）
    trend_5 = calculate_trend_for_each_point(data, '股息率减国债收益率', window=5)
    trend_20 = calculate_trend_for_each_point(data, '股息率减国债收益率', window=20)

    # 使用 GARCH 模型计算条件波动率并标注高低波动性
    data = calculate_garch_volatility(data, index_column)

    # 调用绘图函数并动态标注趋势
    plot_data_with_dynamic_trends(data, x_column, '股息率减国债收益率', index_column, trend_5, trend_20)
