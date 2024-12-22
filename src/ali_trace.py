import time
from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from util.util_ali import print_statistics,loadcsv_dask

ali_folder = "/home/data-7T/lzq/alibaba_block_traces_2020/"


# 1. 块重用时间(对于重复访问的块不去重,会记录多次)
def cal_blk_reuse_distance(trace_df, cache_file='last_access_cache.pkl'):
    """
    计算每个块的重用距离，并将每个块上次访问的索引存储到文件中。
    :param trace_df: DataFrame 包含 trace 数据
    :param cache_file: 保存块上次访问索引的文件路径
    """
    last_access = defaultdict(lambda: np.inf)  # 设置默认值为无穷大，表示从未访问过的块

    reuse_distances = []  # 计算结果

    for index, row in tqdm(trace_df.iterrows(), total=len(trace_df), desc="计算块重用距离"):
        blk_id_start = (row['device_id'], row['offset'] // 4096)  # 使用 (device_id, block_offset) 作为唯一标识，假设块大小为4KB
        blk_count = (row['length'] + 4095) // 4096  # 计算请求覆盖的块数量
        device_id = row['device_id']

        # 遍历请求覆盖的每一个块
        for i in range(blk_count):
            blk_id = (device_id, blk_id_start[1] + i)  # 每个块的唯一标识
            cur = index  # 数据集按照时间戳排序，cur表示当前访问的索引

            # 如果该块之前访问过，则计算重用距离
            if last_access[blk_id] != np.inf:
                reuse_distance = cur - last_access[blk_id]
                reuse_distances.append(reuse_distance)

            # 更新该块的最后访问索引
            last_access[blk_id] = cur

    # plot
    plot_reuse_distance_cdf(reuse_distances, "Block Reuse Distance CDF")
    return reuse_distances

# 绘制重用距离的 CDF 图
def plot_reuse_distance_cdf(reuse_distances, title):
    """
    绘制重用距离的 CDF 图
    :param reuse_distances: 重用距离列表
    :param title: 图表标题
    """
    sorted_distances = sorted(reuse_distances)
    cdf = [i / len(sorted_distances) for i in range(len(sorted_distances))]

    plt.figure(figsize=(12, 8))  # 统一大小为 12x8
    plt.plot(sorted_distances, cdf, marker='.', markersize=4, color='blue', alpha=0.7)  # 修改端点大小
    plt.xlabel('Reuse Distance')
    plt.ylabel('Percent')
    plt.title(title)
    plt.grid(True)
    plt.savefig('../SVG/1-Reuse_distance_CDF.svg', format='svg')
    plt.show()

# 2. 请求大小(length 4K)分布（CDF）
# 横轴 Request size (KiB) 从1起,最小是4K; 纵轴 Cumulative (%)     0%〜100% # 根据 AliCloud Read     AliCloud Write
def plot_req_size_cdf(trace_df, num_samples=1000):
    # 获取读写请求长度并排序
    read_lengths = trace_df[trace_df['opcode'] == 'R']['length'].sort_values().reset_index(drop=True)
    write_lengths = trace_df[trace_df['opcode'] == 'W']['length'].sort_values().reset_index(drop=True)

    # 对数据进行均匀采样，num_samples定义了采样点的数量
    if len(read_lengths) > num_samples:
        indices = np.linspace(0, len(read_lengths) - 1, num_samples, dtype=int)
        read_lengths = read_lengths.iloc[indices]
        read_cdf = np.linspace(0, 1, num_samples)
    else:
        read_cdf = np.arange(1, len(read_lengths) + 1) / len(read_lengths)

    if len(write_lengths) > num_samples:
        indices = np.linspace(0, len(write_lengths) - 1, num_samples, dtype=int)
        write_lengths = write_lengths.iloc[indices]
        write_cdf = np.linspace(0, 1, num_samples)
    else:
        write_cdf = np.arange(1, len(write_lengths) + 1) / len(write_lengths)

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.plot(read_lengths.values, read_cdf, linewidth=2, color='blue', alpha=0.9, label='Read')
    plt.plot(write_lengths.values, write_cdf, linewidth=2, color='red', alpha=0.9, label='Write')
    plt.xlabel('Request Size (Bytes)')
    plt.ylabel('CDF')
    plt.title('Request Size CDF')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(pad=0)
    plt.savefig('../SVG/2-Request_Size_CDF.svg', format='svg')
    plt.show()

def plot_req_freq(trace_df, time_unit='minute'):
    trace_df['timestamp'] = pd.to_datetime(trace_df['timestamp'])

    if time_unit == 'second':
        trace_df['time'] = trace_df['timestamp'].dt.floor('s')
    elif time_unit == 'minute':
        trace_df['time'] = trace_df['timestamp'].dt.floor('min')
    elif time_unit == 'hour':
        trace_df['time'] = trace_df['timestamp'].dt.floor('H')
    else:
        raise ValueError("Invalid time unit, choose 'second', 'minute', or 'hour'.")

    time_requests = trace_df.groupby('time').size() / 1000  # 转换为 kIOPS

    # 使用resample来计算每个时间间隔的均值和峰值
    mean_requests = time_requests.resample(time_unit[0].upper()).mean()
    # peak_requests = time_requests.resample(time_unit[0].upper()).max()

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(mean_requests.index, mean_requests.values, label='Mean kIOPS', color='dodgerblue', linestyle='-', linewidth=2)
    # ax.scatter(peak_requests.index, peak_requests.values, color='crimson', label='Peak kIOPS', s=40, edgecolor='black', zorder=5)
    ax.set_xlabel(f'Time ({time_unit.capitalize()})')
    ax.set_ylabel('Requests (kIOPS)')
    ax.set_title(f'Request Frequency Over Time ({time_unit.capitalize()})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('#f8f8f8')  # 设置轻微灰色的背景色
    plt.tight_layout()
    plt.savefig(f'../SVG/3-1-Request_Frequency_{time_unit}.svg', format='svg')
    plt.show()

    # 2统计各个磁盘的总访问量分布情况
    device_requests = trace_df['device_id'].value_counts().sort_index()
    plt.figure(figsize=(12, 8))
    device_requests.plot(kind='bar')
    plt.xlabel('Device ID')
    plt.ylabel('Number of Requests')
    plt.title('Total Access Distribution Across Devices')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks([])  # 这行代码移除了横轴的刻度标签
    plt.tight_layout()
    plt.savefig('../SVG/3-2-Device_Access_Distribution.svg', format='svg')
    plt.show()


    # 3统计最大访问量的设备的请求频率
    max_device_id = device_requests.idxmax()  # 获取访问次数最多的设备ID
    max_device_df = trace_df[trace_df['device_id'] == max_device_id]
    max_device_time_requests = max_device_df.groupby('time').size() / 1000  # 转换为 kIOPS

    plt.figure(figsize=(12, 8))
    plt.plot(max_device_time_requests.index, max_device_time_requests.values, marker='o', markersize=1, linestyle='-', color='green', linewidth=2, alpha=0.9)
    plt.xlabel(f'Time ({time_unit.capitalize()})')
    plt.ylabel('Number of Requests (kIOPS)')
    plt.title(f'Request Frequency for Most Active Device ID {max_device_id} ({time_unit.capitalize()})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'../SVG/3-3-Most_Active_Device_{max_device_id}_Frequency.svg', format='svg')
    plt.show()


# 参数
sampling_rate=1
trace_file = "/home/data-7T/lzq/alibaba_block_traces_2020/csv/dev_1.csv"
# db_file = "/home/data-7T/lzq/alibaba_block_traces_2020/trace_data.db"

# 主函数
def main():

# 0. load
    t1 = time.time()
    trace_df = loadcsv_dask(trace_file, "2GB", nrows=4600_0000, nhours=2)
    t2 = time.time()
    print(f"加载时间: {t2 - t1:.2f} 秒")
    
    print_statistics(trace_df) # 这里会进行时间戳格式转换
    print(trace_df)

    # 1. 分析块访问模式
    # sample_trace_df = load_trace_ali_panda(trace_file, nrows = 1000_0000, sampling_rate=1)
    # cal_blk_reuse_distance(sample_trace_df)

    # 2. 分析请求大小分布
    plot_req_size_cdf(trace_df)

    # 3. 请求数量的时间分布
    plot_req_freq(trace_df, time_unit='second')


if __name__ == "__main__":
    main()
