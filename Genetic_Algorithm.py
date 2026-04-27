import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


data_path = r"D:\ci\code\cluster_project\Mall_Customers.csv"
df = pd.read_csv(data_path)

# اختيار الـ Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# تطبيع البيانات (مهم جداً للـ Clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




N_CLUSTERS = 5          # عدد المجموعات
POP_SIZE = 50           # حجم المجتمع
MAX_GEN = 100           # عدد الأجيال
CROSSOVER_RATE = 0.8    # نسبة التزاوج
MUTATION_RATE = 0.1     # نسبة الطفرة
N_FEATURES = 2          # عدد الخصائص (Annual Income, Spending Score)

# حدود البحث (بعد التطبيع بتكون حوالي -2 لـ 2)
LOWER_BOUND = -3.0
UPPER_BOUND = 3.0





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

def fitness_function(centroids_flat, data):
    """دالة الـ Fitness (كلما قلت الـ SSE كلما كان الحل أفضل)"""
    centroids = centroids_flat.reshape(N_CLUSTERS, N_FEATURES)
    return -calculate_sse(data, centroids)  # سالب لأننا نريد تقليل الـ SSE





def initialize_population():
    """إنشاء مجتمع عشوائي"""
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, 
                            (POP_SIZE, N_CLUSTERS * N_FEATURES))

def selection(population, fitness):
    """اختيار الآباء باستخدام Tournament Selection"""
    selected = []
    for _ in range(POP_SIZE):
        idx = np.random.choice(len(population), 3, replace=False)
        winner = idx[np.argmax(fitness[idx])]
        selected.append(population[winner].copy())
    return np.array(selected)

def crossover(parents):
    """التزاوج باستخدام One-Point Crossover"""
    offspring = []
    for i in range(0, len(parents), 2):
        p1, p2 = parents[i], parents[i+1]
        if np.random.random() < CROSSOVER_RATE:
            point = np.random.randint(1, len(p1))
            c1 = np.concatenate([p1[:point], p2[point:]])
            c2 = np.concatenate([p2[:point], p1[point:]])
        else:
            c1, c2 = p1.copy(), p2.copy()
        offspring.extend([c1, c2])
    return np.array(offspring)

def mutation(offspring):
    """الطفرة باستخدام Gaussian Mutation"""
    for i in range(len(offspring)):
        if np.random.random() < MUTATION_RATE:
            noise = np.random.normal(0, 0.5, len(offspring[i]))
            offspring[i] += noise
            offspring[i] = np.clip(offspring[i], LOWER_BOUND, UPPER_BOUND)
    return offspring






def run_genetic_algorithm(seed=42):
    np.random.seed(seed)
    
    population = initialize_population()
    best_fitness_history = []
    
    for gen in range(MAX_GEN):
        # حساب الـ Fitness
        fitness = np.array([fitness_function(ind, X_scaled) for ind in population])
        
        # تسجيل أفضل حل
        best_idx = np.argmax(fitness)
        best_fitness_history.append(-fitness[best_idx])  # SSE موجب
        
        # Selection
        parents = selection(population, fitness)
        
        # Crossover
        offspring = crossover(parents)
        
        # Mutation
        offspring = mutation(offspring)
        
        # Elitism: الاحتفاظ بأفضل حل
        population = offspring
        population[0] = population[best_idx].copy()
    
    # النتيجة النهائية
    final_fitness = np.array([fitness_function(ind, X_scaled) for ind in population])
    best_idx = np.argmax(final_fitness)
    best_centroids = population[best_idx].reshape(N_CLUSTERS, N_FEATURES)
    
    # إعادة المراكز للمقياس الأصلي
    best_centroids_original = scaler.inverse_transform(best_centroids)
    
    return best_centroids_original, best_fitness_history[-1], best_fitness_history

def visualize_clusters(centroids, title="GA Clustering"):
    """رسم النتائج"""
    clusters = assign_clusters(X_scaled, 
                              scaler.transform(centroids))
    
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
    centroids, sse, history = run_genetic_algorithm()
    print(f"GA Final SSE: {sse:.4f}")
    visualize_clusters(centroids)