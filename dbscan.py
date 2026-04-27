"""
DBSCAN - Density-Based Spatial Clustering of Applications with Noise
====================================================================
خوارزمية تجميع تعتمد على كثافة النقاط
- لا تحتاج لتحديد عدد المجموعات مسبقاً
- تكتشف مجموعات بأشكال عشوائية
- تحدد النقاط الشاذة (Outliers) تلقائياً
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import deque

# ============================================
# 1. تحميل البيانات
# ============================================
data_path = r"D:\ci\code\cluster_project\Mall_Customers.csv"
df = pd.read_csv(data_path)

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# 2. دوال المسافة
# ============================================
def euclidean_distance(point1, point2):
    """حساب المسافة الإقليدية بين نقطتين"""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def get_neighbors(data, point_idx, eps):
    """
    الحصول على جميع النقاط داخل نطاق eps من نقطة معينة
    
    Parameters:
    -----------
    data : numpy array
        مجموعة البيانات
    point_idx : int
        فهرس النقطة
    eps : float
        نصف القطر
    
    Returns:
    --------
    neighbors : list
        قائمة بفهارس النقاط المجاورة
    """
    neighbors = []
    point = data[point_idx]
    
    for i in range(len(data)):
        if i != point_idx:
            dist = euclidean_distance(point, data[i])
            if dist <= eps:
                neighbors.append(i)
    
    return neighbors

# ============================================
# 3. خوارزمية DBSCAN
# ============================================
class DBSCAN:
    """
    DBSCAN Clustering Algorithm
    
    Parameters:
    -----------
    eps : float
        المسافة القصوى بين نقطتين لاعتبارهما جيران
    min_samples : int
        أقل عدد من النقاط لتكوين مجموعة
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = None
        self.noise_mask_ = None
        
    def fit(self, data):
        """
        تشغيل خوارزمية DBSCAN
        
        تصنيف النقاط:
        - Core Point: نقطة لها min_samples جيران على الأقل
        - Border Point: نقطة ليست Core لكنها جارة لـ Core Point
        - Noise Point: نقطة ليست Core ولا Border
        """
        n_samples = len(data)
        
        # تهيئة التسميات
        # -1 = غير مصنفة بعد
        # -2 = ضوضاء (Noise)
        self.labels_ = np.full(n_samples, -1, dtype=int)
        self.core_sample_indices_ = []
        
        cluster_id = 0
        
        for point_idx in range(n_samples):
            # تخطي النقاط المصنفة مسبقاً
            if self.labels_[point_idx] != -1:
                continue
            
            # البحث عن الجيران
            neighbors = get_neighbors(data, point_idx, self.eps)
            
            # التحقق من Core Point
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -2  # ضوضاء مؤقتة
                continue
            
            # بدء مجموعة جديدة
            cluster_id += 1
            self.labels_[point_idx] = cluster_id
            self.core_sample_indices_.append(point_idx)
            
            # توسيع المجموعة
            self._expand_cluster(data, point_idx, neighbors, cluster_id)
        
        # تحويل الضوضاء المؤقتة إلى ضوضاء نهائية
        self.labels_[self.labels_ == -2] = -1
        
        # تخزين معلومات المجموعات
        self.n_clusters_ = cluster_id
        self.noise_mask_ = self.labels_ == -1
        
        return self
    
    def _expand_cluster(self, data, point_idx, neighbors, cluster_id):
        """
        توسيع المجموعة بإضافة جميع النقاط المتصلة
        
        باستخدام BFS (Breadth-First Search) من خلال طابور
        """
        # استخدام deque كطابور (Queue)
        queue = deque(neighbors)
        visited = {point_idx}
        
        while queue:
            current_idx = queue.popleft()
            
            if current_idx in visited:
                continue
            visited.add(current_idx)
            
            # إذا كانت النقطة ضوضاء، ضمها للمجموعة
            if self.labels_[current_idx] == -2:
                self.labels_[current_idx] = cluster_id
            
            # إذا كانت النقطة مصنفة مسبقاً، تخطى
            if self.labels_[current_idx] != -1:
                continue
            
            # إضافة النقطة للمجموعة
            self.labels_[current_idx] = cluster_id
            
            # البحث عن جيران النقطة
            current_neighbors = get_neighbors(data, current_idx, self.eps)
            
            # إذا كانت Core Point، أضف جيرانها للطابور
            if len(current_neighbors) >= self.min_samples:
                self.core_sample_indices_.append(current_idx)
                queue.extend(current_neighbors)
    
    def get_cluster_statistics(self, data):
        """حساب إحصائيات كل مجموعة"""
        stats = {}
        
        for cluster_id in range(1, self.n_clusters_ + 1):
            cluster_mask = self.labels_ == cluster_id
            cluster_points = data[cluster_mask]
            
            stats[f'Cluster {cluster_id}'] = {
                'Size': len(cluster_points),
                'Percentage': 100 * len(cluster_points) / len(data),
                'Mean': cluster_points.mean(axis=0),
                'Std': cluster_points.std(axis=0)
            }
        
        return stats

def find_optimal_eps(data, k=4):
    """
    إيجاد قيمة eps المناسبة باستخدام k-distance graph
    
    الفكرة:
    1. حساب المسافة لكل نقطة إلى k-th أقرب جار
    2. ترتيب المسافات تصاعدياً
    3. eps = المسافة عند نقطة "الكوع" في الرسم البياني
    """
    from sklearn.neighbors import NearestNeighbors
    
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # المسافة إلى k-th جار (آخر عمود)
    k_distances = distances[:, -1]
    k_distances.sort()
    
    return k_distances


def run_dbscan(eps=0.5, min_samples=5):
    """
    تشغيل DBSCAN
    DBSCAN مش بيحتاج seed لأنه deterministic
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X_scaled)
    
    # حساب SSE كبديل للمقارنة
    sse = 0
    for cluster_id in range(1, dbscan.n_clusters_ + 1):
        cluster_mask = dbscan.labels_ == cluster_id
        cluster_points = X_scaled[cluster_mask]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            sse += np.sum((cluster_points - centroid) ** 2)
    
    return dbscan.labels_, dbscan.n_clusters_, np.sum(dbscan.noise_mask_), sse


def visualize_dbscan(labels, n_clusters, n_noise, title="DBSCAN Clustering"):
    """رسم نتائج DBSCAN"""
    plt.figure(figsize=(12, 5))
    
    # رسم 1: المجموعات
    plt.subplot(1, 2, 1)
    
    # تحويل التسميات للرسم
    unique_labels = set(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(sorted(unique_labels), colors):
        if label == -1:
            # الضوضاء بالأسود
            color = [0, 0, 0, 1]
            label_name = 'Noise'
        else:
            label_name = f'Cluster {label}'
        
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], c=[color], label=label_name, alpha=0.6, s=50)
    
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title(f'{title}\nClusters: {n_clusters}, Noise: {n_noise}')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid(True, alpha=0.3)
    
    # رسم 2: k-distance graph
    plt.subplot(1, 2, 2)
    k_distances = find_optimal_eps(X_scaled)
    plt.plot(k_distances)
    plt.xlabel('Points (sorted by distance)')
    plt.ylabel(f'Distance to 4th nearest neighbor')
    plt.title('K-Distance Graph\n(Find optimal eps at the "elbow")')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_different_parameters():
    """تجربة قيم مختلفة من eps و min_samples"""
    eps_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    min_samples_values = [3, 5, 7, 10]
    
    fig, axes = plt.subplots(len(eps_values), len(min_samples_values), 
                             figsize=(15, 12))
    
    for i, eps in enumerate(eps_values):
        for j, min_samples in enumerate(min_samples_values):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(X_scaled)
            
            ax = axes[i, j]
            
            # رسم النتائج
            unique_labels = set(dbscan.labels_)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(sorted(unique_labels), colors):
                if label == -1:
                    color = [0, 0, 0, 1]
                mask = dbscan.labels_ == label
                ax.scatter(X[mask, 0], X[mask, 1], c=[color], alpha=0.6, s=20)
            
            ax.set_title(f'eps={eps}, min_samples={min_samples}\n'
                        f'Clusters: {dbscan.n_clusters_}')
            ax.set_xlabel('Income')
            ax.set_ylabel('Spending')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================
# 4. التشغيل للتجربة
# ============================================
if __name__ == "__main__":
    # تجربة مع قيم مختلفة
    print("Testing DBSCAN with different parameters...")
    
    for eps in [0.3, 0.4, 0.5]:
        for min_samples in [3, 5, 7]:
            labels, n_clusters, n_noise, sse = run_dbscan(eps=eps, min_samples=min_samples)
            print(f"eps={eps}, min_samples={min_samples}: "
                  f"Clusters={n_clusters}, Noise={n_noise}")
    
    # أفضل إعدادات للبيانات دي
    print("\n" + "="*50)
    print("Best Configuration Found:")
    best_labels, n_clusters, n_noise, best_sse = run_dbscan(eps=0.4, min_samples=5)
    print(f"Clusters: {n_clusters}")
    print(f"Noise points: {n_noise} ({100*n_noise/len(X):.1f}%)")
    print(f"Equivalent SSE: {best_sse:.2f}")
    
    visualize_dbscan(best_labels, n_clusters, n_noise)
    
    # عرض تأثير تغيير المعاملات
    # test_different_parameters()  # Uncomment to see parameter effects