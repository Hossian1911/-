import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

def calculate_historical_quantile_position(data_column):
    """
    将数据转换为其对应的历史分位数
    :param data_column: Pandas Series，股息率减国债收益率的列
    :return: 转换为历史分位数的 Pandas Series
    """
    sorted_data = data_column.sort_values()
    quantile_positions = data_column.apply(lambda x: (sorted_data < x).sum() / len(sorted_data))
    return quantile_positions

def plot_data_with_index(df, x_column, quantile_column, index_column):
    """
    绘制股息率减国债收益率的历史分位数与中证红利指数点位
    :param df: DataFrame 数据
    :param x_column: 日期列
    :param quantile_column: 历史分位数列
    :param index_column: 中证红利指数点位列
    """
    try:
        # 确保日期列为日期格式
        if not pd.api.types.is_datetime64_any_dtype(df[x_column]):
            df[x_column] = pd.to_datetime(df[x_column])

        # 绘图
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 绘制股息率减国债收益率的历史分位数（左纵坐标）
        ax1.plot(df[x_column], df[quantile_column], label='股息率减国债收益率（历史分位数）', marker='o', color='r')
        ax1.set_xlabel("日期", fontsize=12)
        ax1.set_ylabel("历史分位数", fontsize=12, color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True)

        # 创建右纵坐标并绘制中证红利指数点位
        ax2 = ax1.twinx()
        # 假设点位数据需要重新计算（例如缩小100倍）
        df['adjusted_index'] = df[index_column]# 重新计算点位数据
        ax2.plot(df[x_column], df['adjusted_index'], label='中证红利指数点位（调整后）', marker='^', color='g', linestyle='--')
        ax2.set_ylabel("中证红利指数点位（调整后）", fontsize=12, color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.legend(loc='upper right', fontsize=12)

        # 添加标题和布局
        plt.title("股息率减国债收益率历史分位数与中证红利指数点位对比", fontsize=16)
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

    # 创建新列：股息率减国债收益率
    data['股息率减国债收益率'] = data[y1_column] - data[y2_column]

    # 转换为历史分位数
    data['股息率减国债收益率（历史分位数）'] = calculate_historical_quantile_position(data['股息率减国债收益率'])

    # 调用绘图函数
    plot_data_with_index(data, x_column, '股息率减国债收益率（历史分位数）', index_column)

