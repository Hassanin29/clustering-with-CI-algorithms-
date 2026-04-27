import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================================
# 1. تحميل البيانات
# ============================================
data_path = r"D:\ci\code\cluster_project\Mall_Customers.csv"
df = pd.read_csv(data_path)

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# 2. تشغيل K-Means
# ============================================
def run_kmeans(n_clusters=5, n_init=30, seed=42):
    """تشغيل K-Means عدة مرات وأخذ أفضل نتيجة"""
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
    kmeans.fit(X_scaled)
    
    # حساب SSE
    sse = kmeans.inertia_
    
    # المراكز
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    
    return centroids, sse

def visualize_clusters(centroids, title="K-Means Clustering"):
    """رسم النتائج"""
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_scaled)
    clusters = kmeans.labels_
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolors='black')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title(title)
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.show()

# ============================================
# 3. التشغيل
# ============================================
if __name__ == "__main__":
    centroids, sse = run_kmeans()
    print(f"K-Means Final SSE: {sse:.4f}")
    visualize_clusters(centroids)