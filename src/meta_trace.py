from sys import meta_path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



meta_folder = "/home/data-7T/lzq/meta_cachelib_baleen"
path_data0 = meta_folder + "/storage/20230325/Region5/"
# data0 ： storage_0.1.tar.gz的20230325 (0.1% traces (1 sample each)	15 MB	76 MB)
'''例子：
`310 262144 4325376 1572074461.57806 1 24 3 5`
这个示例中的追踪行表示块ID为310的请求，从字节偏移262144开始，大小为4325376字节，时间为1572074461.57806，操作类型为GET_TEMP（即临时读取）。操作类型（`op_name`）的映射包括：

- GET_TEMP = 1，GET_PERM = 2
- PUT_TEMP = 3，PUT_PERM = 4
- GET_NOT_INIT = 5，PUT_NOT_INIT = 6
- UNKNOWN = 100
`PUT_OPS` 包括 [4, 3, 6]，`GET_OPS` 包括 [2, 1, 5]。
'''



# --- load  trace ---
# 读取 meta *.trace 文件
def read_save2csv(file_path, output_path="../data-csv/data.npy"):
    # 读取数据并解析字段
    data = np.loadtxt(file_path, comments='#', dtype={'names': ('block_id', 'io_offset', 'io_size', 'time', 'op_name', 'user_namespace', 'user_name', 'rs_shard_id', 'op_count', 'host_name'),
                                                      'formats': ('i8', 'i8', 'i8', 'f8', 'i4', 'S10', 'S10', 'i4', 'i4', 'S10')})
    # 保存为 .npy 格式
    np.save(output_path, data)
    print(f"Data saved to {output_path}")

    return data

# 分析重用距离
def analyze_reuse_distance(traces):
    last_access = {}
    reuse_distances = []

    for idx, trace in enumerate(traces):
        block_id = trace[0]
        if block_id in last_access:
            reuse_distance = idx - last_access[block_id]
            reuse_distances.append(reuse_distance)
        last_access[block_id] = idx

    # 绘制重用距离分布
    plt.figure(figsize=(10, 6))
    plt.hist(reuse_distances, bins=50, color='lightblue', edgecolor='black')
    plt.title('Reuse Distance Distribution')
    plt.xlabel('Reuse Distance')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.show()



if __name__ == '__main__':
    meta_trace = read_save2csv(path_data0 + "full_0_0.1.trace")
    print(meta_trace[:5])  # 显示前5条数据

    analyze_reuse_distance(meta_trace)

    # io_size_distribution, op_counts = analyze_load(meta_trace[:5])
    # show(io_size_distribution, op_counts)