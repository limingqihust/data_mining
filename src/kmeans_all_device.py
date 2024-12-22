import sys
sys.path.append('../')  # 如果util模块在上级目录
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# trace_path = "../../device_.csv"
file_paths = [f"../../device_{i}.csv" for i in range(1, 30)]

statistics = []



def DataPreProcess():

    for file_path in file_paths:
        df = pd.read_csv(file_path, names=["device_id", "opcode", "offset", "length", "timestamp"])
        df['operation_type'] = df['opcode'].map({'R': 1, 'W': 0})
        df['timestamp_diff'] = df['timestamp'].diff().fillna(0)
        io_count = len(df)
        io_size_avg = df['length'].mean()
        io_size_std = df['length'].std() if df['length'].size > 1 else 0
        read_write_ratio = df['operation_type'].mean()
        response_time_avg = df['timestamp_diff'].mean()          
        response_time_std = df['timestamp_diff'].std() if df['timestamp_diff'].size > 1 else 0
        
        statistics.append({
            "file": file_path,
            "io_count": io_count,
            "io_size_avg": io_size_avg,
            "io_size_std": io_size_std,
            "read_write_ratio": read_write_ratio,
            "response_time_avg": response_time_avg,
            "response_time_std": response_time_std,
        })
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