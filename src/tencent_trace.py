import time
from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from util.util_tencent import print_statistics, loadcsv_dask  # 假设有一个类似的工具库

# 腾讯数据集路径
tencent_folder = "/home/Data-7T-nvme/lzq/tencent_blk_trace/"
trace_files = [f"{tencent_folder}/{file}" for file in os.listdir(tencent_folder) if file.endswith('.tgz')]

# 1. 块重用时间
def cal_blk_reuse_distance(trace_df, cache_file='last_access_cache.pkl'):
    """
    计算每个块的重用距离，并将每个块上次访问的索引存储到文件中。
    :param trace_df: DataFrame 包含 trace 数据
    :param cache_file: 保存块上次访问索引的文件路径
    """
    last_access = defaultdict(lambda: np.inf)  # 设置默认值为无穷大，表示从未访问过的块
    reuse_distances = []  # 计算结果

    for index, row in tqdm(trace_df.iterrows(), total=len(trace_df), desc="计算块重用距离"):
        blk_id_start = (row['device_id'], row['offset'] // 4096)  # 假设块大小为4KB
        blk_count = (row['length'] + 4095) // 4096  # 计算请求覆盖的块数量

        for i in range(blk_count):
            blk_id = (row['device_id'], blk_id_start[1] + i)  # 每个块的唯一标识
            cur = index  # 数据集按照时间戳排序，cur表示当前访问的索引

            # 如果该块之前访问过，则计算重用距离
            if last_access[blk_id] != np.inf:
                reuse_distance = cur - last_access[blk_id]
                reuse_distances.append(reuse_distance)

            # 更新该块的最后访问索引
            last_access[blk_id] = cur

    plot_reuse_distance_cdf(reuse_distances, "Block Reuse Distance CDF")
    return reuse_distances

# 绘制重用距离的 CDF 图
def plot_reuse_distance_cdf(reuse_distances, title):
    """
    绘制重用距离的 CDF 图
    """
    sorted_distances = sorted(reuse_distances)
    cdf = [i / len(sorted_distances) for i in range(len(sorted_distances))]

    plt.figure(figsize=(12, 8))
    plt.plot(sorted_distances, cdf, marker='.', markersize=4, color='blue', alpha=0.7)
    plt.xlabel('Reuse Distance')
    plt.ylabel('Percent')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f'../SVG/{title.replace(" ", "_")}.svg', format='svg')
    plt.show()

# 请求大小分布（CDF）
def plot_request_size_distribution_cdf(trace_df, num_samples=1000):
    read_lengths = trace_df[trace_df['opcode'] == 'R']['length'].sort_values().reset_index(drop=True)
    write_lengths = trace_df[trace_df['opcode'] == 'W']['length'].sort_values().reset_index(drop=True)

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

    plt.figure(figsize=(12, 8))
    plt.plot(read_lengths.values, read_cdf, linewidth=2, color='blue', alpha=0.9, label='Read')
    plt.plot(write_lengths.values, write_cdf, linewidth=2, color='red', alpha=0.9, label='Write')
    plt.xlabel('Request Size (Bytes)')
    plt.ylabel('CDF')
    plt.title('Request Size CDF')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(pad=0)
    plt.savefig('../SVG/Request_Size_CDF.svg', format='svg')
    plt.show()

# 请求频率分析
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
    mean_requests = time_requests.resample(time_unit[0].upper()).mean()

    plt.figure(figsize=(14, 8))
    plt.plot(mean_requests.index, mean_requests.values, label='Mean kIOPS', color='dodgerblue', linestyle='-', linewidth=2)
    plt.xlabel(f'Time ({time_unit.capitalize()})')
    plt.ylabel('Requests (kIOPS)')
    plt.title(f'Request Frequency Over Time ({time_unit.capitalize()})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'../SVG/Request_Frequency_{time_unit}.svg', format='svg')
    plt.show()

# 主函数
def main():
    # 0. 加载数据
    t1 = time.time()
    trace_df = loadcsv_dask(trace_files[0], "2GB", nrows=None)  # 假设只处理第一个文件
    t2 = time.time()
    print(f"加载时间: {t2 - t1:.2f} 秒")

    print_statistics(trace_df)  # 输出统计信息

    # 1. 分析块访问模式
    cal_blk_reuse_distance(trace_df)

    # 2. 分析请求大小分布
    plot_request_size_distribution_cdf(trace_df)

    # 3. 请求数量的时间分布
    plot_req_freq(trace_df, time_unit='minute')

if __name__ == "__main__":
    main()
