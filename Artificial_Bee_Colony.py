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
POP_SIZE = 50          # عدد مصادر الطعام = عدد النحل العامل = عدد النحل المراقب
MAX_ITER = 100
LIMIT = 10             # حد الإهمال
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

def fitness(sse):
    """تحويل SSE إلى Fitness (كلما قل SSE زاد الـ Fitness)"""
    return 1.0 / (1.0 + sse)





def run_abc(seed=42):
    np.random.seed(seed)
    
    # تهيئة مصادر الطعام
    food_sources = np.random.uniform(LOWER_BOUND, UPPER_BOUND, 
                                    (POP_SIZE, N_CLUSTERS * N_FEATURES))
    trials = np.zeros(POP_SIZE)  # عداد الإهمال
    
    # حساب الـ Fitness الأولي
    fitness_values = np.array([fitness(calculate_sse(X_scaled, 
                            fs.reshape(N_CLUSTERS, N_FEATURES))) for fs in food_sources])
    
    history = []
    
    for iteration in range(MAX_ITER):
        # مرحلة النحل العامل (Employed Bees)
        for i in range(POP_SIZE):
            # اختيار شريك عشوائي
            k = np.random.choice([j for j in range(POP_SIZE) if j != i])
            
            # اختيار بعد عشوائي للتعديل
            j = np.random.randint(N_CLUSTERS * N_FEATURES)
            
            # إنتاج حل جديد
            phi = np.random.uniform(-1, 1)
            new_solution = food_sources[i].copy()
            new_solution[j] += phi * (food_sources[i][j] - food_sources[k][j])
            new_solution = np.clip(new_solution, LOWER_BOUND, UPPER_BOUND)
            
            # تقييم الحل الجديد
            new_sse = calculate_sse(X_scaled, new_solution.reshape(N_CLUSTERS, N_FEATURES))
            new_fitness = fitness(new_sse)
            
            # Greedy Selection
            if new_fitness > fitness_values[i]:
                food_sources[i] = new_solution
                fitness_values[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1
        
        # حساب الاحتمالات للنحل المراقب
        probabilities = fitness_values / fitness_values.sum()
        
        # مرحلة النحل المراقب (Onlooker Bees)
        for i in range(POP_SIZE):
            if np.random.random() < probabilities[i]:
                k = np.random.choice([j for j in range(POP_SIZE) if j != i])
                j = np.random.randint(N_CLUSTERS * N_FEATURES)
                
                phi = np.random.uniform(-1, 1)
                new_solution = food_sources[i].copy()
                new_solution[j] += phi * (food_sources[i][j] - food_sources[k][j])
                new_solution = np.clip(new_solution, LOWER_BOUND, UPPER_BOUND)
                
                new_sse = calculate_sse(X_scaled, new_solution.reshape(N_CLUSTERS, N_FEATURES))
                new_fitness = fitness(new_sse)
                
                if new_fitness > fitness_values[i]:
                    food_sources[i] = new_solution
                    fitness_values[i] = new_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1
        
        # مرحلة النحل الكشاف (Scout Bees)
        for i in range(POP_SIZE):
            if trials[i] >= LIMIT:
                food_sources[i] = np.random.uniform(LOWER_BOUND, UPPER_BOUND, 
                                                   N_CLUSTERS * N_FEATURES)
                fitness_values[i] = fitness(calculate_sse(X_scaled, 
                                           food_sources[i].reshape(N_CLUSTERS, N_FEATURES)))
                trials[i] = 0
        
        best_sse = 1.0 / fitness_values.max() - 1.0
        history.append(best_sse)
    
    best_idx = np.argmax(fitness_values)
    best_centroids = food_sources[best_idx].reshape(N_CLUSTERS, N_FEATURES)
    best_centroids_original = scaler.inverse_transform(best_centroids)
    best_sse = 1.0 / fitness_values[best_idx] - 1.0
    
    return best_centroids_original, best_sse, history

def visualize_clusters(centroids, title="ABC Clustering"):
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
    centroids, sse, history = run_abc()
    print(f"ABC Final SSE: {sse:.4f}")
    visualize_clusters(centroids)