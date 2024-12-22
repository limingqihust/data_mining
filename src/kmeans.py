import sys
sys.path.append('../')  # 如果util模块在上级目录
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

trace_path = "../../device_7.csv"

def DataPreProcess():
    df = pd.read_csv(trace_path, names=["device_id", "opcode", "offset", "length", "timestamp"])
    df['operation_type'] = df['opcode'].map({'R': 1, 'W': 0})
    df['timestamp_diff'] = df['timestamp'].diff().fillna(0)
    
    window_size = 3600 * 1000 * 1000 
    # window_size = 36000 * 1000 
    # window_size = 14400 * 1000 # 14400s 4h 为一个窗口    
    # window_size = 7200 * 1000 # 7200s 2h 为一个窗口
    # window_size = 3600 * 1000 # 3600s 1h 为一个窗口
    # window_size = 1800 * 1000 # 1800s 0.5h 为一个窗口
    # window_size = 600 * 1000 # 600s 10min 为一个窗口


    df['time_window'] = (df['timestamp'] // window_size).astype('int64')
    # print(df)
    

    features = []

    # 提取每个窗口的特征
    for _, group in df.groupby('time_window'):
        avg_io_count = len(group)  # 每个时间窗口的IO数量
        avg_io_size = group['length'].mean()  # 平均IO大小
        io_size_std = group['length'].std() if group['length'].size > 1 else 0  # IO大小的标准差
        read_write_ratio = group['operation_type'].mean()  # 读写比例
        avg_response_time = group['timestamp_diff'].mean()  # 平均响应时间（假设用timestamp_diff作为响应时间）
        
        # 请求间隔分布特征：请求间隔的均值和标准差
        interval_mean = group['timestamp_diff'].mean()
        interval_std = group['timestamp_diff'].std() if group['timestamp_diff'].size > 1 else 0
        
        features.append([avg_io_count, avg_io_size, io_size_std, read_write_ratio, avg_response_time, interval_mean, interval_std])

    # 将提取的特征转化为DataFrame
    feature_df = pd.DataFrame(features, columns=['avg_io_count', 'avg_io_size', 'io_size_std', 'read_write_ratio', 'avg_response_time', 'interval_mean', 'interval_std'])

    # 查看提取的特征
    print("print feature_df")
    print(feature_df)
    return feature_df


def StandardProcess(feature_df):
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    # 查看标准化后的数据
    print("print X_scaled")
    print(X_scaled)
    return X_scaled


def GetK(X_scaled):
    # 计算不同k值下的SSE（误差平方和），用以选择最优的k
    sse = []
    k_range = range(2, 12)  # 从2到11的聚类数
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

    # 绘制肘部法则图
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.show()

def MyKMeans(k):
    # 使用K-Means进行聚类
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)

    # 将聚类结果加入原数据
    feature_df['cluster'] = kmeans.labels_
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 绘制聚类结果
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=feature_df['cluster'], cmap='viridis')
    plt.colorbar(label='Cluster')
    plt.title('K-Means Clustering of IO Trace Data')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()




if __name__ == '__main__':
    feature_df = DataPreProcess()
    X_scaled = StandardProcess(feature_df)
    # GetK(X_scaled)
    k = 6
    MyKMeans(k)