# 数据分析工具集

用于分析不同来源的块级 I/O 访问跟踪数据，包括阿里巴巴、腾讯和 Meta 的数据集。

## 目录结构

```
.
├── src/                    # 源代码目录
│   ├── ali_trace.py       # 阿里巴巴数据集分析
│   ├── tencent_trace.py   # 腾讯数据集分析
│   ├── meta_trace.py      # Meta数据集分析
│   └── per-dev.py         # 按照每个设备进行分析
├── util/                   # 工具函数
│   ├── util_ali.py        # 阿里数据集工具函数
│   ├── util_tencent.py    # 腾讯数据集工具函数
│   ├── temp.py            # 临时工具函数
│   ├── split.cpp          # 对ali的csv按设备进行数据分割
│   └── split_thread.cpp   # 多线程对ali的csv按设备进行数据分割
└── SVG/                    # 输出的统计图表
```

## 安装依赖

```bash
pip install pandas numpy matplotlib dask
```

## 函数说明

默认属性：

- chunksize：4KB

### 数据加载函数

#### [load](src/per-dev.ipynb#L10)

参数：`(device_id, start_hour=0, duration_hours=2)`

默认写死:

- 块大小(加载的 csv 最大大小)：900MB
- 数据文件路径：`/home/Data-7T-nvme/lzq/alibaba_block_traces_2020/csvfix/device_{device_id}.csv`
- 基准时间：`2020-01-01 00:00:00`

返回 DataFrame：`{device_id}, 时间范围: {基准时间+start_hour} 以后的{duration_hours}小时`

```python
# 示例：加载设备9从凌晨开始的2小时数据
df = load(device_id=9, start_hour=0, duration_hours=1)
```

#### [loadcsv_dask](util/util_ali.py#L138)

参数：`(file_path, blksize="1GB", nrows=None, start_time="2019-12-31 16:00:00", nhours=None)`

load 的 子函数，调用 Dask 加载指定时间范围的跟踪数据。

### 分析函数

#### [plot_req_size_cdf(trace_df)](src/per-dev.ipynb#L20)

画图函数，请求大小分布, CDF
带 P99 的标注线
输出文件：`SVG/dev-{dev_id}/1-Req_Size_CDF_{time_range}.svg`

#### [plot_req_freq(trace_df, time_interval_seconds=60)](src/per-dev.ipynb#L26)

画 IOPS 图。峰值 IOSP = 每秒的请求数;
平均 IOPS = 每秒的请求数 / 时间间隔
time*interval_seconds 默认 60 秒
输出文件：`SVG/dev-{dev_id}/2-Request_Frequency*{time_range}.svg`

```python
# 输入：预处理后的DataFrame，时间间隔（秒）
# 输出：SVG/dev-{dev_id}/2-Request_Frequency_{time_range}.svg
plot_req_freq(df, time_interval_seconds=60)
```

#### [plot_reuse_distance_cdf(trace_df)](src/per-dev.ipynb#L32)

画图函数，重用距离(重用次数)分布图。
输出文件：`SVG/dev-{dev_id}/3-1-reuse_times_cdf_separate_{time_range}.svg`

```python
# 输入：预处理后的DataFrame
# 输出：SVG/dev-{dev_id}/3-1-reuse_times_cdf_separate_{time_range}.svg
plot_reuse_distance_cdf(df)
```

#### [plot_unique_reuse_cdf(trace_df)](src/per-dev.ipynb#L44)

画图函数，唯一空间访问模式图。

#### [plot_access_space(trace_df)](src/per-dev.ipynb#L38)

生成空间访问模式图。

```python
# 输入：预处理后的DataFrame
# 输出：SVG/dev-{dev_id}/4-2-access_pattern_space_{time_range}.svg
plot_access_space(df)
```

## 输入数据格式

所有分析函数需要的 DataFrame 格式：

- timestamp: 时间戳(微秒)
- device_id: 设备 ID
- offset: 偏移量
- length: 请求大小
- opcode: 操作类型(R/W)

## 常用分析流程

1. 基础分析流程：

```python
# 加载数据并生成所有分析图表
df = load(device_id=9, start_hour=0, duration_hours=2)
```

2. 自定义分析流程：

```python
# 加载数据
df = load_trace_dask("/path/to/trace.csv")

# 生成特定分析图表
plot_req_size_cdf(df)
plot_req_freq(df, time_interval_seconds=300)  # 5分钟间隔
plot_reuse_distance_cdf(df)
plot_access_space(df)
```

## 注意事项

- 所有图表自动保存在`SVG/dev-{device_id}/`目录下
- 文件名包含时间范围信息（格式：MMDD_HH-MMDD_HH）
- 大文件建议使用`load_trace_dask`
- 默认块大小(chunk_size)为 4KB
