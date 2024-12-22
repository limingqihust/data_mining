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

    # for file_path in file_paths:
    #     df = pd.read_csv(file_path, names=["device_id", "opcode", "offset", "length", "timestamp"])
    #     df['operation_type'] = df['opcode'].map({'R': 1, 'W': 0})
    #     df['timestamp_diff'] = df['timestamp'].diff().fillna(0)
    #     io_count = len(df)
    #     io_size_avg = df['length'].mean()
    #     io_size_std = df['length'].std() if df['length'].size > 1 else 0
    #     read_write_ratio = df['operation_type'].mean()
    #     response_time_avg = df['timestamp_diff'].mean()          
    #     response_time_std = df['timestamp_diff'].std() if df['timestamp_diff'].size > 1 else 0
        
    #     statistics.append({
    #         "file": file_path,
    #         "io_count": io_count,
    #         "io_size_avg": io_size_avg,
    #         "io_size_std": io_size_std,
    #         "read_write_ratio": read_write_ratio,
    #         "response_time_avg": response_time_avg,
    #         "response_time_std": response_time_std,
    #     })
    # stats_df = pd.DataFrame(statistics)
    # stats_df.to_csv('./stats_df.csv', index=False)
    stats_df = pd.read_csv('./stats_df.csv')

    print("print stats_df:")
    print(stats_df)

    # 标准化
    features = stats_df[["io_count", "io_size_avg", "io_size_std", "read_write_ratio", "response_time_avg", "response_time_std"]]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA降维
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    loading_matrix = pd.DataFrame(
        pca.components_,
        columns=["io_count", "io_size_avg", "io_size_std", "read_write_ratio", "response_time_avg", "response_time_std"],
        index=['PCA1', 'PCA2']
    )

    print("主成分载荷矩阵：")
    print(loading_matrix)

    pca_df = pd.DataFrame(features_pca, columns=["PCA1", "PCA2"])

    # 肘部法则确定k
    # inertia = []
    # K = range(1, 11)
    # for k in K:
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(features_pca)
    #     inertia.append(kmeans.inertia_)

    # 绘制肘部法图像
    # plt.figure(figsize=(8, 6))
    # plt.plot(K, inertia, 'bo-', markersize=8)
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method for Optimal k')
    # plt.show()


    # 绘制聚类结果
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    pca_df['cluster'] = kmeans.fit_predict(features_pca)


    # 绘制每个device
    plt.figure(figsize=(6, 4))
    for cluster in range(optimal_k):
        cluster_data = pca_df[pca_df['cluster'] == cluster]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], s=75, label=f"Cluster {cluster}")

    # 绘制聚类中心
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        marker='*', s=300, label='Centroids'
    )




    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('KMeans Clustering Results (PCA Reduced)')
    plt.legend()
    plt.show()





    return stats_df


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
    # X_scaled = StandardProcess(feature_df)
    # # GetK(X_scaled)
    # k = 6
    # MyKMeans(k)