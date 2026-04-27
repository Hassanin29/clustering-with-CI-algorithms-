import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data_path = r"D:\ci\code\cluster_project\Mall_Customers.csv"
df = pd.read_csv(data_path)

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)





N_CLUSTERS = 5
N_ANTS = 50
MAX_ITER = 100
EVAPORATION_RATE = 0.1
N_FEATURES = 2

LOWER_BOUND = -3.0
UPPER_BOUND = 3.0




def assign_clusters(data, centroids):
    distances = np.sqrt(((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=1)

def calculate_sse(data, centroids):
    clusters = assign_clusters(data, centroids)
    sse = 0.0
    for i in range(len(centroids)):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - centroids[i]) ** 2)
    return sse






def run_aco(seed=42):
    np.random.seed(seed)
    
    # تهيئة الفيرمونات (Probability Distribution لكل بعد)
    pheromones = np.ones((N_CLUSTERS * N_FEATURES, 50))  # 50 مستوى لكل بعد
    
    best_solution = None
    best_sse = float('inf')
    history = []
    
    for iteration in range(MAX_ITER):
        solutions = []
        scores = []
        
        # بناء الحلول بواسطة النمل
        for ant in range(N_ANTS):
            # اختيار قيم من الفيرمونات
            solution = np.zeros(N_CLUSTERS * N_FEATURES)
            for dim in range(N_CLUSTERS * N_FEATURES):
                probs = pheromones[dim] / pheromones[dim].sum()
                idx = np.random.choice(50, p=probs)
                # تحويل index إلى قيمة بين LOWER_BOUND و UPPER_BOUND
                solution[dim] = LOWER_BOUND + (idx / 49.0) * (UPPER_BOUND - LOWER_BOUND)
            
            centroids = solution.reshape(N_CLUSTERS, N_FEATURES)
            sse = calculate_sse(X_scaled, centroids)
            
            solutions.append(solution)
            scores.append(sse)
            
            if sse < best_sse:
                best_sse = sse
                best_solution = solution.copy()
        
        # تبخير الفيرمونات
        pheromones *= (1 - EVAPORATION_RATE)
        
        # إضافة فيرمونات جديدة بناءً على أفضل الحلول
        sorted_indices = np.argsort(scores)[:5]  # أفضل 5 نمل
        
        for idx in sorted_indices:
            solution = solutions[idx]
            quality = 1.0 / (1.0 + scores[idx])
            
            for dim in range(N_CLUSTERS * N_FEATURES):
                bin_idx = int((solution[dim] - LOWER_BOUND) / 
                             (UPPER_BOUND - LOWER_BOUND) * 49)
                bin_idx = np.clip(bin_idx, 0, 49)
                pheromones[dim][bin_idx] += quality
        
        history.append(best_sse)
    
    best_centroids = best_solution.reshape(N_CLUSTERS, N_FEATURES)
    best_centroids_original = scaler.inverse_transform(best_centroids)
    
    return best_centroids_original, best_sse, history

def visualize_clusters(centroids, title="ACO Clustering"):
    clusters = assign_clusters(X_scaled, scaler.transform(centroids))
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolors='black')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title(title)
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    centroids, sse, history = run_aco()
    print(f"ACO Final SSE: {sse:.4f}")
    visualize_clusters(centroids)