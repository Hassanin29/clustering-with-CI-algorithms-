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
POP_SIZE = 50
MAX_ITER = 100
F = 0.8          # Differential weight
CR = 0.9         # Crossover probability
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






def run_de(seed=42):
    np.random.seed(seed)
    
    # تهيئة المجتمع
    population = np.random.uniform(LOWER_BOUND, UPPER_BOUND, 
                                  (POP_SIZE, N_CLUSTERS * N_FEATURES))
    scores = np.array([calculate_sse(X_scaled, ind.reshape(N_CLUSTERS, N_FEATURES)) 
                      for ind in population])
    
    history = []
    
    for iteration in range(MAX_ITER):
        for i in range(POP_SIZE):
            # اختيار ثلاث أفراد عشوائيين
            candidates = [j for j in range(POP_SIZE) if j != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Mutation
            mutant = population[a] + F * (population[b] - population[c])
            mutant = np.clip(mutant, LOWER_BOUND, UPPER_BOUND)
            
            # Crossover
            cross_points = np.random.random(N_CLUSTERS * N_FEATURES) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, len(cross_points))] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Selection
            trial_score = calculate_sse(X_scaled, trial.reshape(N_CLUSTERS, N_FEATURES))
            
            if trial_score < scores[i]:
                population[i] = trial
                scores[i] = trial_score
        
        history.append(np.min(scores))
    
    best_idx = np.argmin(scores)
    best_centroids = population[best_idx].reshape(N_CLUSTERS, N_FEATURES)
    best_centroids_original = scaler.inverse_transform(best_centroids)
    
    return best_centroids_original, scores[best_idx], history

def visualize_clusters(centroids, title="DE Clustering"):
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
    centroids, sse, history = run_de()
    print(f"DE Final SSE: {sse:.4f}")
    visualize_clusters(centroids)