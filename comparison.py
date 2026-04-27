import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tabulate import tabulate

# Import all algorithms
from Genetic_Algorithm import run_genetic_algorithm
from Particle_Swarm_Optimization import run_pso
from Differential_Evolution import run_de
from Artificial_Bee_Colony import run_abc
from Ant_Colony_Optimization import run_aco
from kmeans import run_kmeans

# ============================================
# دالة المقارنة الرئيسية
# ============================================
def run_comparison_analysis(n_runs=10, show_plots=True):
    """
    تشغيل مقارنة شاملة بين جميع الخوارزميات
    
    Parameters:
    -----------
    n_runs : int
        عدد مرات التشغيل لكل خوارزمية
    show_plots : bool
        هل نعرض الرسوم البيانية؟
    
    Returns:
    --------
    results : dict
        قاموس يحتوي على نتائج جميع الخوارزميات
    """
    
    algorithms = {
        'K-Means': run_kmeans,
        'GA': run_genetic_algorithm,
        'PSO': run_pso,
        'DE': run_de,
        'ABC': run_abc,
        'ACO': run_aco
    }
    
    # ============================================
    # 1. تهيئة النتائج
    # ============================================
    results = {name: {'sse': [], 'time': [], 'history': []} for name in algorithms}
    
    print("\n" + "="*70)
    print("🚀 STARTING COMPREHENSIVE COMPARISON")
    print("="*70)
    
    for name, algo_func in algorithms.items():
        print(f"\n{'='*50}")
        print(f"Running {name} for {n_runs} runs...")
        print('='*50)
        
        for run in range(n_runs):
            seed = 42 + run
            
            start_time = time.time()
            
            if name == 'K-Means':
                centroids, sse = algo_func(n_clusters=5, seed=seed)
                history = [sse]
            else:
                centroids, sse, history = algo_func(seed=seed)
            
            end_time = time.time()
            
            results[name]['sse'].append(sse)
            results[name]['time'].append(end_time - start_time)
            results[name]['history'].append(history)
            
            if (run + 1) % 5 == 0:
                print(f"  Completed {run + 1}/{n_runs} runs")
    
    # ============================================
    # 2. حساب الإحصائيات
    # ============================================
    stats = []
    for name in algorithms:
        sse_array = np.array(results[name]['sse'])
        time_array = np.array(results[name]['time'])
        
        stats.append([
            name,
            f"{np.mean(sse_array):.4f}",
            f"{np.std(sse_array):.4f}",
            f"{np.min(sse_array):.4f}",
            f"{np.max(sse_array):.4f}",
            f"{np.mean(time_array):.4f}",
            f"{np.std(time_array):.4f}"
        ])
    
    # ============================================
    # 3. عرض النتائج في الـ Console
    # ============================================
    headers = ['Algorithm', 'Mean SSE', 'Std SSE', 'Min SSE', 'Max SSE', 'Mean Time (s)', 'Std Time (s)']
    print("\n" + "="*100)
    print("📊 COMPARISON RESULTS")
    print("="*100)
    print(tabulate(stats, headers=headers, tablefmt='grid'))
    
    # تحديد أفضل خوارزمية
    best_algo = min(algorithms.keys(), key=lambda x: np.mean(results[x]['sse']))
    print(f"\n🏆 Best Algorithm (Lowest SSE): {best_algo}")
    
    # ============================================
    # 4. رسم النتائج (اختياري)
    # ============================================
    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 4.1 Boxplot لـ SSE
        ax1 = axes[0, 0]
        sse_data = [results[name]['sse'] for name in algorithms]
        bp1 = ax1.boxplot(sse_data, labels=list(algorithms.keys()), patch_artist=True)
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('SSE')
        ax1.set_title('Algorithm Performance (SSE) - Lower is Better')
        ax1.grid(True, alpha=0.3)
        
        # 4.2 Bar plot لوقت التنفيذ
        ax2 = axes[0, 1]
        time_means = [np.mean(results[name]['time']) for name in algorithms]
        bars = ax2.bar(list(algorithms.keys()), time_means, color=colors, alpha=0.7)
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Average Execution Time')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 4.3 Convergence curves
        ax3 = axes[1, 0]
        for name in algorithms:
            if name != 'K-Means':
                # متوسط history لكل الخوارزميات
                avg_history = np.mean([h[:100] if len(h) > 100 else h for h in results[name]['history']], axis=0)
                ax3.plot(avg_history, label=name, linewidth=2)
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('SSE')
        ax3.set_title('Average Convergence Curves')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4.4 Summary Table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # إنشاء جدول ملخص
        table_data = []
        for name in algorithms:
            sse_mean = np.mean(results[name]['sse'])
            sse_std = np.std(results[name]['sse'])
            time_mean = np.mean(results[name]['time'])
            table_data.append([name, f"{sse_mean:.4f}±{sse_std:.4f}", f"{time_mean:.4f}s"])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Algorithm', 'SSE (Mean±Std)', 'Time'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.4, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # تلوين صف أفضل خوارزمية
        best_idx = list(algorithms.keys()).index(best_algo)
        for j in range(3):
            table[(best_idx + 1, j)].set_facecolor('#90EE90')
        
        plt.tight_layout()
        plt.show()
    
    return results, stats, best_algo

# ============================================
# دالة للحصول على ملخص سريع
# ============================================
def get_quick_summary(n_runs=5):
    """دالة سريعة للحصول على ملخص بدون رسوم"""
    algorithms = {
        'K-Means': run_kmeans,
        'GA': run_genetic_algorithm,
        'PSO': run_pso,
        'DE': run_de,
        'ABC': run_abc,
        'ACO': run_aco
    }
    
    summary = {}
    
    for name, algo_func in algorithms.items():
        sse_list = []
        time_list = []
        
        for run in range(n_runs):
            start_time = time.time()
            
            if name == 'K-Means':
                _, sse = algo_func(n_clusters=5, seed=42+run)
            else:
                _, sse, _ = algo_func(seed=42+run)
                
            time_list.append(time.time() - start_time)
            sse_list.append(sse)
        
        summary[name] = {
            'sse_mean': np.mean(sse_list),
            'sse_std': np.std(sse_list),
            'time_mean': np.mean(time_list)
        }
    
    return summary

# ============================================
# التشغيل المباشر (للتجربة فقط)
# ============================================
if __name__ == "__main__":
    # تشغيل سريع للتجربة
    print("Running quick test (5 runs each)...")
    results, stats, best = run_comparison_analysis(n_runs=5, show_plots=False)
    print(f"\n✅ Quick test completed! Best: {best}")