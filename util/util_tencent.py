import pandas as pd
import dask.dataframe as dd
import os

def loadcsv_dask(file_path, blksize, nrows=None, start_time=None, nhours=None):
    """
    使用 Dask 从文件中读取 trace 数据，并根据时间范围筛选。
    :param file_path: trace 文件的路径
    :param blksize: 块大小
    :param nrows: 读取的行数
    :param start_time: 起始时间
    :param nhours: 读取的小时数
    """
    # 设置数据类型和列名
    dtypes = {
        "Timestamp": "int64",  # Unix 时间戳
        "Offset": "int64",      # 偏移量
        "Size": "int32",        # 请求大小（以扇区为单位）
        "IOType": "int32",    # 操作类型
        "VolumeID": "int32"     # 卷 ID
    }
    columns = ["Timestamp", "Offset", "Size", "IOType", "VolumeID"]

    # 使用 Dask 读取 CSV 文件
    df = dd.read_csv(
        file_path,
        names=columns,
        dtype=dtypes,
        header=None,
        blocksize=blksize,
        encoding='us-ascii'  # 指定编码为 us-ascii
    )

    # 计算前 n 行数据，如果指定
    if nrows is not None:
        df = df.head(n=nrows, compute=False)

    # 如果指定了起始时间和小时数，转换为 UNIX 时间戳（微秒），并筛选
    if start_time is not None:
        start_timestamp = pd.Timestamp(start_time).value // 10**3  # 转换为微秒时间戳
        if nhours is not None:
            end_timestamp = start_timestamp + nhours * 3600 * 10**6  # 计算结束时间的微秒时间戳
            df = df[(df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp)]

    # 计算结果并转换为 Pandas DataFrame
    trace_df = df.compute()
    return trace_df

def print_statistics(trace_df):
    """
    打印 trace 数据的基本统计信息
    :param trace_df: DataFrame 包含 trace 数据
    """
    total_reads = (trace_df['IOType'] == 1).sum()  # 根据 IOType 判断读请求
    total_writes = (trace_df['IOType'] == 0).sum()  # 根据 IOType 判断写请求
    read_ratio = total_reads / (total_reads + total_writes) if (total_reads + total_writes) > 0 else 0
    write_ratio = total_writes / (total_reads + total_writes) if (total_reads + total_writes) > 0 else 0
    print(trace_df)
    # 输出统计结果
    print(f"总读请求: {total_reads}, 总写请求: {total_writes}")
    print(f"读写比例: 读 = {read_ratio:.2%}, 写 = {write_ratio:.2%}")
    print(f"请求大小范围: {trace_df['Size'].min()} ~ {trace_df['Size'].max()} 扇区")
    print(f"地址范围: {trace_df['Offset'].min()} ~ {trace_df['Offset'].max()} 字节")
    print(f"时间范围: 从 {trace_df['Timestamp'].min()} 到 {trace_df['Timestamp'].max()}")
