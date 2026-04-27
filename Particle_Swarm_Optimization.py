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
N_PARTICLES = 50
MAX_ITER = 100
W = 0.7          # Inertia weight
C1 = 1.5         # Cognitive coefficient
C2 = 1.5         # Social coefficient
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







def run_pso(seed=42):
    np.random.seed(seed)
    
    # تهيئة الجسيمات
    positions = np.random.uniform(LOWER_BOUND, UPPER_BOUND, 
                                 (N_PARTICLES, N_CLUSTERS * N_FEATURES))
    velocities = np.random.uniform(-1, 1, (N_PARTICLES, N_CLUSTERS * N_FEATURES))
    
    # أفضل موقع لكل جسيم
    pbest_positions = positions.copy()
    pbest_scores = np.array([calculate_sse(X_scaled, p.reshape(N_CLUSTERS, N_FEATURES)) 
                            for p in positions])
    
    # أفضل موقع عام
    gbest_idx = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    
    history = []
    
    for iteration in range(MAX_ITER):
        for i in range(N_PARTICLES):
            # تحديث السرعة
            r1, r2 = np.random.random(2)
            velocities[i] = (W * velocities[i] + 
                           C1 * r1 * (pbest_positions[i] - positions[i]) +
                           C2 * r2 * (gbest_position - positions[i]))
            
            # تحديث الموقع
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], LOWER_BOUND, UPPER_BOUND)
            
            # تقييم الحل الجديد
            centroids = positions[i].reshape(N_CLUSTERS, N_FEATURES)
            current_score = calculate_sse(X_scaled, centroids)
            
            # تحديث pbest
            if current_score < pbest_scores[i]:
                pbest_scores[i] = current_score
                pbest_positions[i] = positions[i].copy()
                
                # تحديث gbest
                if current_score < gbest_score:
                    gbest_score = current_score
                    gbest_position = positions[i].copy()
        
        history.append(gbest_score)
    
    best_centroids = gbest_position.reshape(N_CLUSTERS, N_FEATURES)
    best_centroids_original = scaler.inverse_transform(best_centroids)
    
    return best_centroids_original, gbest_score, history

def visualize_clusters(centroids, title="PSO Clustering"):
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
    centroids, sse, history = run_pso()
    print(f"PSO Final SSE: {sse:.4f}")
    visualize_clusters(centroids)