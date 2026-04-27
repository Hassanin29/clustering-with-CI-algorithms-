"""
K-Means++ Clustering Algorithm
===============================
نسخة محسنة من K-Means تستخدم تهيئة ذكية للمراكز الابتدائية
بدل ما تختار المراكز عشوائياً، بتختار مراكز متباعدة عن بعض
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# 2. دوال الـ Clustering الأساسية
# ============================================
def assign_clusters(data, centroids):
    """توزيع النقاط على أقرب مركز"""
    distances = np.sqrt(((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=1)

def calculate_sse(data, centroids):
    """حساب Sum of Squared Errors"""
    clusters = assign_clusters(data, centroids)
    sse = 0.0
    for i in range(len(centroids)):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - centroids[i]) ** 2)
    return sse

# ============================================
# 3. خوارزمية K-Means++
# ============================================
def initialize_centroids_kmeans_pp(data, k):
    """
    تهيئة المراكز بطريقة K-Means++
    
    الفكرة:
    1. اختيار أول مركز عشوائياً
    2. حساب المسافة من كل نقطة لأقرب مركز
    3. اختيار المركز التالي باحتمال يتناسب مع مربع المسافة
       (النقاط البعيدة عن المراكز الحالية لها فرصة أكبر)
    4. تكرار حتى الحصول على K مراكز
    """
    n_samples = len(data)
    
    # الخطوة 1: اختيار أول مركز عشوائياً
    first_centroid_idx = np.random.randint(n_samples)
    centroids = [data[first_centroid_idx].copy()]
    
    # الخطوة 2-4: اختيار باقي المراكز
    for _ in range(1, k):
        # حساب المسافة من كل نقطة لأقرب مركز موجود
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            # المسافة لأقرب مركز
            min_dist = float('inf')
            for centroid in centroids:
                dist = np.sqrt(np.sum((data[i] - centroid) ** 2))
                min_dist = min(min_dist, dist)
            distances[i] = min_dist ** 2  # مربع المسافة
        
        # اختيار المركز الجديد باحتمال يتناسب مع مربع المسافة
        probabilities = distances / distances.sum()
        next_centroid_idx = np.random.choice(n_samples, p=probabilities)
        centroids.append(data[next_centroid_idx].copy())
    
    return np.array(centroids)

def run_kmeans_pp(n_clusters=5, max_iters=100, seed=42):
    """
    تشغيل K-Means++ مع تهيئة ذكية
    
    Parameters:
    -----------
    n_clusters : int
        عدد المجموعات
    max_iters : int
        أقصى عدد للتكرارات
    seed : int
        بذرة العشوائية
    
    Returns:
    --------
    centroids : numpy array
        المراكز النهائية (بالمقياس الأصلي)
    sse : float
        مجموع مربعات الأخطاء
    history : list
        تاريخ التحسن عبر التكرارات
    """
    np.random.seed(seed)
    
    # تهيئة ذكية باستخدام K-Means++
    centroids_scaled = initialize_centroids_kmeans_pp(X_scaled, n_clusters)
    
    history = []
    
    for iteration in range(max_iters):
        # حساب SSE الحالي
        current_sse = calculate_sse(X_scaled, centroids_scaled)
        history.append(current_sse)
        
        # توزيع النقاط
        clusters = assign_clusters(X_scaled, centroids_scaled)
        
        # تحديث المراكز
        new_centroids = np.zeros_like(centroids_scaled)
        for i in range(n_clusters):
            cluster_points = X_scaled[clusters == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # لو المجموعة فاضية، نعيد تهيئة المركز عشوائياً
                new_centroids[i] = X_scaled[np.random.randint(len(X_scaled))]
        
        # التحقق من التوقف
        if np.allclose(centroids_scaled, new_centroids, rtol=1e-4):
            break
        
        centroids_scaled = new_centroids
    
    # التحويل للمقياس الأصلي
    centroids_original = scaler.inverse_transform(centroids_scaled)
    final_sse = calculate_sse(X_scaled, centroids_scaled)
    
    return centroids_original, final_sse, history

def run_multiple_init(n_clusters=5, n_init=10, max_iters=100):
    """
    تشغيل K-Means++ عدة مرات واختيار أفضل نتيجة
    بيستخدم np.random.seed من بره
    """
    best_centroids = None
    best_sse = float('inf')
    best_history = None
    
    for i in range(n_init):
        centroids, sse, history = run_kmeans_pp(
            n_clusters=n_clusters, 
            max_iters=max_iters, 
            seed=np.random.randint(10000)  # عشوائي داخلياً
        )
        
        if sse < best_sse:
            best_sse = sse
            best_centroids = centroids
            best_history = history
    
    return best_centroids, best_sse, best_history

def visualize_clusters(centroids, title="K-Means++ Clustering"):
    """رسم النتائج"""
    clusters = assign_clusters(X_scaled, scaler.transform(centroids))
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, 
               edgecolors='black', linewidths=2, zorder=5)
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title(title)
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_with_standard_kmeans(n_clusters=5):
    """مقارنة مع K-Means العادي"""
    from sklearn.cluster import KMeans
    
    # K-Means العادي
    kmeans_standard = KMeans(n_clusters=n_clusters, n_init=30, random_state=42)
    kmeans_standard.fit(X_scaled)
    standard_sse = kmeans_standard.inertia_
    
    # K-Means++
    kmeans_pp = KMeans(n_clusters=n_clusters, init='k-means++', n_init=30, random_state=42)
    kmeans_pp.fit(X_scaled)
    pp_sse = kmeans_pp.inertia_
    
    print("="*50)
    print("K-Means vs K-Means++ Comparison")
    print("="*50)
    print(f"Standard K-Means SSE: {standard_sse:.2f}")
    print(f"K-Means++ SSE:        {pp_sse:.2f}")
    print(f"Improvement:          {standard_sse - pp_sse:.2f}")
    
    # رسم المقارنة
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # رسم K-Means العادي
    standard_labels = kmeans_standard.labels_
    standard_centroids = scaler.inverse_transform(kmeans_standard.cluster_centers_)
    ax1.scatter(X[:, 0], X[:, 1], c=standard_labels, cmap='viridis', alpha=0.6)
    ax1.scatter(standard_centroids[:, 0], standard_centroids[:, 1], 
               c='red', marker='X', s=200, edgecolors='black')
    ax1.set_title(f'Standard K-Means\nSSE: {standard_sse:.2f}')
    ax1.set_xlabel('Annual Income (k$)')
    ax1.set_ylabel('Spending Score (1-100)')
    ax1.grid(True, alpha=0.3)
    
    # رسم K-Means++
    pp_labels = kmeans_pp.labels_
    pp_centroids = scaler.inverse_transform(kmeans_pp.cluster_centers_)
    ax2.scatter(X[:, 0], X[:, 1], c=pp_labels, cmap='viridis', alpha=0.6)
    ax2.scatter(pp_centroids[:, 0], pp_centroids[:, 1], 
               c='red', marker='X', s=200, edgecolors='black')
    ax2.set_title(f'K-Means++\nSSE: {pp_sse:.2f}')
    ax2.set_xlabel('Annual Income (k$)')
    ax2.set_ylabel('Spending Score (1-100)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================
# 4. التشغيل للتجربة
# ============================================
if __name__ == "__main__":
    # تشغيل K-Means++ عدة مرات
    centroids, sse, history = run_multiple_init(n_clusters=5, n_init=10)
    
    print(f"K-Means++ Final SSE: {sse:.2f}")
    print(f"Number of iterations: {len(history)}")
    
    # عرض المراكز
    print("\nFinal Centroids (Original Scale):")
    for i, c in enumerate(centroids):
        print(f"  Cluster {i+1}: Income=${c[0]:.2f}k, Spending Score={c[1]:.2f}")
    
    visualize_clusters(centroids)
    
    # مقارنة مع K-Means العادي
    compare_with_standard_kmeans()