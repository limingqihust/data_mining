import time
import pandas as pd
import sqlite3
import os
import dask.dataframe as dd
import math
import numpy as np
import dask
import matplotlib.pyplot as plt
from functorch.dim import t__setitem__
from torch.onnx.symbolic_opset11 import chunk
from tqdm import tqdm


from datetime import datetime

def unix_to_Date(microtimestamp):
    return datetime.utcfromtimestamp(microtimestamp / 1000000).strftime('%Y-%m-%d %H:%M:%S.%f')

def date_to_Unix(date_str):
    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
    return int(dt.timestamp() * 1000000)  # 转换为微秒

# # 示例用法
#     micro_timestamp = 1580486398672122  # 例如 2020-01-17 02:57:14 的微秒时间戳
#     date_string = unix_to_Date(micro_timestamp)
#     print(f"{micro_timestamp} -> {date_string}")
#
#     # 转换回微秒时间戳
#     date_time = "2020-01-31 16:00:00.00000"
#     new_microtimestamp = date_to_Unix(date_time)
#     print(f"{date_time} -> {new_microtimestamp}")

def add_point_lines(x, y, color='blue', alpha=0.7):
        """
        为指定点(x,y)添加参考线和标注
        
        参数:
        x, y: 点的坐标
        color: 线条颜色
        alpha: 透明度
        """
        # 画垂直于x轴的线
        plt.vlines(x=x, ymin=0, ymax=y, color=color, linestyle='--', alpha=alpha)
        
        # 画垂直于y轴的线
        plt.hlines(y=y, xmin=0, xmax=x, color=color, linestyle='--', alpha=alpha)
        
        # 标注点的坐标
        plt.plot(x, y, 'o', color=color, markersize=3)  # 在点的位置画一个圆点
        plt.annotate(f'({x:.1f}, {y:.2f})', 
                    xy=(x, y),             # 点的位置
                    xytext=(10, 10),       # 文本的偏移量
                    textcoords='offset points',  # 使用偏移坐标
                    color=color)


# load from csv ------------------------------------------------------------

# panda读取太慢了
def loadcsv_panda(file_path, nrows=None, nhours=None, sampling_rate=1, chunksize=1e5):

    """
    :param file_path: trace文件的路径
    :param nrows: 读取的行数，默认读取全部数据
    :param nhours: 读取的小时数，默认读取全部数据
    :param sampling_rate: 采样率，每隔多少行读取一行数据
    :param chunksize: 每次读取的行数，用于分块读取大文件
    """

    # 根据采样率计算跳过的行数
    def should_skip_line(line_num):
        # 每 1/sampling_rate 行读取一行
        return (line_num != 0) and (math.floor(line_num * sampling_rate) != line_num * sampling_rate)

    # 指定数据类型，减少内存占用
    dtypes = {
        "device_id": "int32",
        "opcode": "category",  # 使用类别类型更省内存
        "offset": "int64",
        "length": "int32",
        "timestamp": "int64"  # 原始的时间戳值
    }
    columns = ["device_id", "opcode", "offset", "length", "timestamp"]

    # 分块读取文件
    reader = pd.read_csv(
        file_path,
        names=columns,
        dtype=dtypes,
        nrows=nrows*sampling_rate,
        header=None,
        chunksize=int(chunksize),
        skiprows=lambda x: should_skip_line(x),  # 使用自定义跳行逻辑
        on_bad_lines='skip'
    )

    # 获取 nhours 时间范围限制
    start_time, end_time = None, None
    filtered_chunks = []
    chunk_count = 0  # 计数器初始化

    try:
        for chunk in reader:
            chunk_count += 1  # 更新计数器
            print(f"正在处理第 {chunk_count} 个数据块...")

            # 将时间戳转换为 datetime
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce', unit='us')
            chunk.dropna(subset=['timestamp'], inplace=True)  # 去除无效的时间戳

            if nhours is not None and start_time is None:
                start_time = chunk['timestamp'].iloc[0]
                end_time = start_time + pd.Timedelta(hours=nhours)

            # 过滤时间范围
            if nhours is not None:
                chunk = chunk[(chunk['timestamp'] >= start_time) & (chunk['timestamp'] <= end_time)]

            # 将过滤后的数据块添加到列表中
            filtered_chunks.append(chunk)

        # 合并所有过滤后的数据块
        if filtered_chunks:
            trace_df = pd.concat(filtered_chunks, ignore_index=True)
        else:
            trace_df = pd.DataFrame(columns=columns)

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return pd.DataFrame(columns=columns)

    # 计算基本特性
    print_statistics(trace_df)
    return trace_df

# 
def loadcsv_dask(file_path, blksize="1GB", nrows=None, start_time="2019-12-31 16:00:00", nhours=None):
    """
    使用dask并行加载CSV文件的函数。

    参数:
        file_path (str): CSV文件路径
        blksize (str): dask读取的块大小,默认"1GB" 
        nrows (int, optional): 读取的最大行数,默认None表示读取全部
        start_time (str, optional): 开始时间,格式"YYYY-MM-DD HH:MM:SS",默认"2019-12-31 16:00:00"
        nhours (int, optional): 读取的小时数,默认None表示不限制

    返回:
        pandas.DataFrame: 包含以下列的数据表:
            - device_id (int32): 设备ID
            - opcode (category): 操作类型,'R'表示读,'W'表示写
            - offset (int64): 偏移量
            - length (int32): 长度
            - timestamp (datetime64): 时间戳

    示例:
        >>> df = loadcsv_dask("trace.csv", blksize="2GB", nrows=1000000)
        >>> df = loadcsv_dask("trace.csv", start_time="2020-01-01", nhours=24)

    """
    load_start = time.time()
    dtypes = {
        "device_id": np.int32,
        "opcode": "category",
        "offset": np.int64,
        "length": np.int32,
        "timestamp": np.int64
    }
    columns = ["device_id", "opcode", "offset", "length", "timestamp"]

    # 使用dask读取，设置更大的分区
    df = dd.read_csv(
        file_path,
        names=columns,
        dtype=dtypes,
        header=None,
        blocksize=blksize,
        assume_missing=True,  # 提高性能
        on_bad_lines='skip'
    )

    # 时间过滤
    if start_time is not None:
        t1 = date_to_Unix(start_time)
        df = df[df['timestamp'] >= t1]
        if nhours is not None:
            t2 = t1 + nhours * 3600 * 10**6
            df = df[df['timestamp'] <= t2]

    # 行数限制
    if nrows is not None:
        df = df.head(n=nrows, compute=False)

    # 优化计算
    with dask.config.set(scheduler='processes', num_workers=32):  # 使用多进程
        trace_df = df.compute()

    load_finish = time.time()
    print(f"Dask加载时间: {load_finish - load_start:.2f} 秒")
    return trace_df

def print_statistics(trace_df):
    df = trace_df.copy()
    # 统计读写操作的数量
    total_reads = (df['opcode'] == 'R').sum()
    total_writes = (df['opcode'] == 'W').sum()
    read_ratio = total_reads / (total_reads + total_writes) if (total_reads + total_writes) > 0 else 0
    write_ratio = total_writes / (total_reads + total_writes) if (total_reads + total_writes) > 0 else 0

    # 转换单位
    length_min = df['length'].min() / 1024  # 转换为KB
    length_max = df['length'].max() / 1024
    offset_min = df['offset'].min() / (1024 * 1024)  # 转换为MB 
    offset_max = df['offset'].max() / (1024 * 1024)

    # 输出统计结果
    print(df)

    print(f"读写比例: 读 = {read_ratio:.2%}, 写 = {write_ratio:.2%}")
    print(f"请求大小: {length_min:.2f} ~ {length_max:.2f} KB")
    print(f"地址范围: {offset_min:.2f} ~ {offset_max:.2f} MB")
    print(f"时间范围: 从 {unix_to_Date(df['timestamp'].min())} 到 {unix_to_Date(df['timestamp'].max())}")


def sample_and_save_csv(trace_file, nrows, rate):
    # 计算采样后的行数
    sample_nrows = int(nrows * rate)

    # 从原始文件中读取前 nrows 行数据
    df = pd.read_csv(trace_file, nrows=nrows)

    # 根据指定的采样比例进行采样
    sampled_df = df.sample(n=sample_nrows) if rate < 1 else df

    # 构建新文件的名称，包含采样的行数和比例
    new_file_name = f"/home/data-7T/lzq/alibaba_block_traces_2020/io_traces_sampled_{nrows}_{rate:.6f}.csv"

    # 将采样后的数据保存到新文件
    sampled_df.to_csv(new_file_name, index=False)

    print(f"Sampled data saved to {new_file_name}")



def loadcsv_file(file_path, nrows=None, start_time="2019-12-31 16:00:00", nhours=None, buffer_size=8192*1024):
    """
    优化的文件指针读取方法，适用于大数据量
    :param buffer_size: 读取缓冲区大小，默认8MB
    """
    t1 = time.time()
    
    # 计算时间戳范围
    start_timestamp = None
    end_timestamp = None
    if start_time is not None:
        start_timestamp = pd.Timestamp(start_time).value // 10**3
        if nhours is not None:
            end_timestamp = start_timestamp + nhours * 3600 * 10**6

    # 使用列表而不是字典来存储数据，减少内存开销
    device_ids = []
    opcodes = []
    offsets = []
    lengths = []
    timestamps = []
    
    count = 0
    buffer = ""
    remainder = ""
    
    with open(file_path, 'r', buffering=buffer_size) as f:
        while True:
            chunk = f.read(buffer_size)
            if not chunk:
                break
                
            buffer = remainder + chunk
            lines = buffer.split('\n')
            remainder = lines[-1]
            
            for line in lines[:-1]:
                if nrows is not None and count >= nrows:
                    break
                    
                try:
                    values = line.strip().split(',')
                    if len(values) != 5:
                        continue
                        
                    device_id, opcode, offset, length, timestamp = values
                    timestamp = int(timestamp)
                    
                    if start_timestamp and timestamp < start_timestamp:
                        continue
                    if end_timestamp and timestamp > end_timestamp:
                        continue
                    
                    device_ids.append(int(device_id))
                    opcodes.append(opcode)
                    offsets.append(int(offset))
                    lengths.append(int(length))
                    timestamps.append(timestamp)
                    
                    count += 1
                    if count % 1000000 == 0:  # 减少打印频率
                        print(f"已读取 {count//1000000}M 行...")
                        
                except (ValueError, IndexError):
                    continue
            
            if nrows is not None and count >= nrows:
                break
    
    # 使用字典构造DataFrame，更高效
    trace_df = pd.DataFrame({
        'device_id': np.array(device_ids, dtype=np.int32),
        'opcode': pd.Categorical(opcodes),
        'offset': np.array(offsets, dtype=np.int64),
        'length': np.array(lengths, dtype=np.int32),
        'timestamp': np.array(timestamps, dtype=np.int64)
    })
    
    t2 = time.time()
    print(f"文件指针读取时间: {t2 - t1:.2f} 秒")
    return trace_df

def compare_loading_methods(file_path, nrows=100000000):  # 增加默认测试行数
    """
    优化的性能比较方法
    """
    print("\n性能比较开始...")
    results = []
    
    # 测试文件指针方法
    print("\n1. 测试文件指针方法:")
    t1 = time.time()
    df1 = loadcsv_file(file_path, nrows=nrows, buffer_size=16*1024*1024)  # 使用16MB缓冲区
    t2 = time.time()
    time_file = t2 - t1
    mem1 = df1.memory_usage(deep=True).sum() / 1024 / 1024
    results.append(('文件指针', time_file, mem1))
    print(df1)
    del df1  # 释放内存
    
    # 测试dask方法
    print("\n2. 测试dask方法:")
    t1 = time.time()
    # 854757459
    df2 = loadcsv_dask(file_path, "900MB", nrows=nrows)  # 增加块大小
    t2 = time.time()
    time_dask = t2 - t1
    mem2 = df2.memory_usage(deep=True).sum() / 1024 / 1024
    results.append(('Dask', time_dask, mem2))
    print(df2)

    # 比较结果
    print("\n性能比较结果:")
    print(f"{'方法':<15} {'加载时间(秒)':<15} {'内存使用(MB)':<15} {'相对速度':<15}")
    print("-" * 60)
    min_time = min(r[1] for r in results)
    for method, time_taken, mem_used in results:
        print(f"{method:<15} {time_taken:<15.2f} {mem_used:<15.2f} {min_time/time_taken:<15.2f}x")



def main():
    trace_file = "/home/Data-7T-nvme/lzq/alibaba_block_traces_2020/csvfix/device_1.csv"
    
    # 比较不同方法的性能
    compare_loading_methods(trace_file, nrows=10000000)  # 测试前1000万行
    compare_loading_methods(trace_file, nrows=100000000)  # 测试一亿行, dev1 总共22462799 dask 18s > file 29s

if __name__ == "__main__":
    main()
