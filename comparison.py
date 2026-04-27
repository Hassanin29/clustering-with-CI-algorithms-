"""
Comparison Module - مقارنة شاملة بين جميع خوارزميات التجميع
============================================================
يقارن بين:
- Classical: K-Means, K-Means++, DBSCAN
- Evolutionary: GA, PSO, DE, ABC, ACO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler

# Import all algorithms
from Genetic_Algorithm import run_genetic_algorithm
from Particle_Swarm_Optimization import run_pso
from Differential_Evolution import run_de
from Artificial_Bee_Colony import run_abc
from Ant_Colony_Optimization import run_aco
from kmeans import run_kmeans
from kmeans_plus_plus import run_kmeans_pp, run_multiple_init
from dbscan import run_dbscan

# ============================================
# تحميل البيانات
# ============================================
data_path = r"D:\ci\code\cluster_project\Mall_Customers.csv"
df = pd.read_csv(data_path)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# دوال مساعدة لاستخراج SSE من النتائج المختلفة
# ============================================
def extract_sse(result, algo_name):
    """
    استخراج SSE من نتيجة أي خوارزمية
    
    Parameters:
    -----------
    result : tuple
        نتيجة الخوارزمية (تختلف من خوارزمية لأخرى)
    algo_name : str
        اسم الخوارزمية
    
    Returns:
    --------
    sse : float
        قيمة SSE
    extra_info : dict or None
        معلومات إضافية (لـ DBSCAN)
    """
    if algo_name == 'DBSCAN':
        # DBSCAN: (labels, n_clusters, n_noise, sse)
        labels, n_clusters, n_noise, sse = result
        return sse, {'clusters': n_clusters, 'noise': n_noise}
    else:
        # باقي الخوارزميات: (centroids, sse, history)
        centroids, sse, history = result
        return sse, None

def safe_run(algo_func, algo_name, seed):
    """
    تشغيل آمن للخوارزمية مع معالجة الأخطاء
    
    Parameters:
    -----------
    algo_func : function
        دالة الخوارزمية
    algo_name : str
        اسم الخوارزمية
    seed : int
        بذرة العشوائية
    
    Returns:
    --------
    sse : float or None
        قيمة SSE (None إذا فشلت)
    exec_time : float
        وقت التنفيذ
    extra_info : dict or None
        معلومات إضافية
    """
    try:
        start_time = time.time()
        result = algo_func(seed=seed)
        exec_time = time.time() - start_time
        
        sse, extra_info = extract_sse(result, algo_name)
        return sse, exec_time, extra_info
        
    except Exception as e:
        print(f"  ⚠️ Error in {algo_name} (seed={seed}): {str(e)}")
        return None, 0, None

# ============================================
# دالة المقارنة الرئيسية
# ============================================
def run_comparison_analysis(n_runs=10, show_plots=True, verbose=True):
    """
    تشغيل مقارنة شاملة بين جميع الخوارزميات
    
    Parameters:
    -----------
    n_runs : int
        عدد مرات التشغيل لكل خوارزمية (افتراضي 10)
    show_plots : bool
        هل نعرض الرسوم البيانية؟ (افتراضي True)
    verbose : bool
        هل نطبع تفاصيل أثناء التشغيل؟ (افتراضي True)
    
    Returns:
    --------
    results : dict
        قاموس يحتوي على نتائج جميع الخوارزميات
    stats : list
        إحصائيات منسقة للعرض
    best_algo : str
        اسم أفضل خوارزمية
    """
    
    # تعريف الخوارزميات
    algorithms = {
        'K-Means': {
            'func': lambda s: run_kmeans(n_clusters=5, seed=s),
            'type': 'Classical',
            'color': '#3498db'
        },
        'K-Means++': {
            'func': lambda s: run_multiple_init(n_clusters=5, n_init=10, seed=s),
            'type': 'Classical',
            'color': '#2ecc71'
        },
        'DBSCAN': {
            'func': lambda s: run_dbscan(eps=0.4, min_samples=5, seed=s),
            'type': 'Classical',
            'color': '#e74c3c'
        },
        'GA': {
            'func': run_genetic_algorithm,
            'type': 'Evolutionary',
            'color': '#f39c12'
        },
        'PSO': {
            'func': run_pso,
            'type': 'Evolutionary',
            'color': '#9b59b6'
        },
        'DE': {
            'func': run_de,
            'type': 'Evolutionary',
            'color': '#1abc9c'
        },
        'ABC': {
            'func': run_abc,
            'type': 'Evolutionary',
            'color': '#34495e'
        },
        'ACO': {
            'func': run_aco,
            'type': 'Evolutionary',
            'color': '#e67e22'
        }
    }
    
    # ============================================
    # 1. تهيئة النتائج
    # ============================================
    results = {}
    for name in algorithms:
        results[name] = {
            'sse': [],
            'time': [],
            'extra': []
        }
    
    if verbose:
        print("\n" + "="*70)
        print("🚀 STARTING COMPREHENSIVE COMPARISON")
        print("="*70)
        print(f"📊 Algorithms: {len(algorithms)}")
        print(f"🔄 Runs per algorithm: {n_runs}")
        print(f"📈 Total runs: {len(algorithms) * n_runs}")
        print("="*70)
    
    # ============================================
    # 2. تشغيل كل الخوارزميات
    # ============================================
    for name, algo_info in algorithms.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"🔄 Running {name} ({algo_info['type']})...")
            print('='*50)
        
        algo_func = algo_info['func']
        successful_runs = 0
        
        for run in range(n_runs):
            seed = 42 + run
            
            sse, exec_time, extra_info = safe_run(algo_func, name, seed)
            
            if sse is not None:
                results[name]['sse'].append(sse)
                results[name]['time'].append(exec_time)
                if extra_info:
                    results[name]['extra'].append(extra_info)
                successful_runs += 1
            
            # طباعة تقدم كل 5 تشغيلات
            if verbose and (run + 1) % 5 == 0:
                print(f"  ✅ Completed {run + 1}/{n_runs} runs ({successful_runs} successful)")
        
        if verbose and successful_runs > 0:
            print(f"  📊 {name}: {successful_runs}/{n_runs} successful runs")
    
    # ============================================
    # 3. حساب الإحصائيات
    # ============================================
    stats = []
    
    for name in algorithms:
        if results[name]['sse']:
            sse_array = np.array(results[name]['sse'])
            time_array = np.array(results[name]['time'])
            
            algo_type = algorithms[name]['type']
            
            row = [
                name,
                algo_type,
                f"{np.mean(sse_array):.4f}",
                f"{np.std(sse_array):.4f}",
                f"{np.min(sse_array):.4f}",
                f"{np.max(sse_array):.4f}",
                f"{np.mean(time_array):.4f}",
                f"{np.std(time_array):.4f}"
            ]
            
            # إضافة معلومات DBSCAN
            if name == 'DBSCAN' and results[name]['extra']:
                avg_clusters = np.mean([e['clusters'] for e in results[name]['extra']])
                avg_noise = np.mean([e['noise'] for e in results[name]['extra']])
                row.append(f"Clusters: {avg_clusters:.1f}")
                row.append(f"Noise: {avg_noise:.1f}")
            else:
                row.append("-")
                row.append("-")
            
            stats.append(row)
    
    # ============================================
    # 4. عرض النتائج
    # ============================================
    headers = ['Algorithm', 'Type', 'Mean SSE', 'Std SSE', 'Min SSE', 'Max SSE', 
               'Mean Time (s)', 'Std Time (s)', 'Extra 1', 'Extra 2']
    
    if verbose:
        print("\n" + "="*110)
        print("📊 COMPARISON RESULTS")
        print("="*110)
        print(tabulate(stats, headers=headers, tablefmt='grid'))
    
    # تحديد أفضل خوارزمية (أقل SSE)
    valid_algos = {n: r for n, r in results.items() if r['sse']}
    if valid_algos:
        best_algo = min(valid_algos.keys(), 
                       key=lambda x: np.mean(valid_algos[x]['sse']))
        
        if verbose:
            print(f"\n🏆 Best Algorithm (Lowest SSE): {best_algo}")
            print(f"   Mean SSE: {np.mean(results[best_algo]['sse']):.4f}")
            print(f"   Mean Time: {np.mean(results[best_algo]['time']):.4f}s")
    else:
        best_algo = None
    
    # ============================================
    # 5. رسم النتائج
    # ============================================
    if show_plots:
        plot_comparison_results(results, algorithms, best_algo, stats)
    
    return results, stats, best_algo

# ============================================
# دوال الرسم
# ============================================
def plot_comparison_results(results, algorithms, best_algo, stats):
    """رسم نتائج المقارنة"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    valid_names = [n for n in algorithms if results[n]['sse']]
    
    # ============================================
    # 1. Boxplot لـ SSE
    # ============================================
    ax1 = axes[0, 0]
    sse_data = [results[n]['sse'] for n in valid_names]
    bp = ax1.boxplot(sse_data, labels=valid_names, patch_artist=True)
    
    for patch, name in zip(bp['boxes'], valid_names):
        patch.set_facecolor(algorithms[name]['color'])
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('SSE (Lower is Better)')
    ax1.set_title(f'Algorithm Performance (SSE)\n🏆 Best: {best_algo}')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # ============================================
    # 2. Bar plot لوقت التنفيذ
    # ============================================
    ax2 = axes[0, 1]
    time_means = [np.mean(results[n]['time']) for n in valid_names]
    bars = ax2.bar(valid_names, time_means, 
                   color=[algorithms[n]['color'] for n in valid_names], alpha=0.7)
    
    # إضافة القيم فوق الأعمدة
    for bar, val in zip(bars, time_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Average Execution Time')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # ============================================
    # 3. Convergence Curves (للخوارزميات التطورية فقط)
    # ============================================
    ax3 = axes[0, 2]
    for name in valid_names:
        if algorithms[name]['type'] == 'Evolutionary':
            # تشغيل مرة واحدة لرسم منحنى التقارب
            try:
                result = algorithms[name]['func'](seed=42)
                centroids, sse, history = result
                ax3.plot(history, label=name, color=algorithms[name]['color'], linewidth=2)
            except:
                pass
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('SSE')
    ax3.set_title('Convergence Curves (Evolutionary)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ============================================
    # 4. Scatter: SSE vs Time
    # ============================================
    ax4 = axes[1, 0]
    for name in valid_names:
        sse_mean = np.mean(results[name]['sse'])
        time_mean = np.mean(results[name]['time'])
        
        ax4.scatter(time_mean, sse_mean, 
                   c=algorithms[name]['color'], s=200, 
                   label=name, edgecolors='black', linewidths=1, zorder=5)
        
        # إضافة اسم الخوارزمية
        ax4.annotate(name, (time_mean, sse_mean), 
                    xytext=(10, 10), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Average Time (seconds)')
    ax4.set_ylabel('Average SSE')
    ax4.set_title('Time vs Quality Trade-off')
    ax4.legend(loc='upper right', fontsize='small')
    ax4.grid(True, alpha=0.3)
    
    # ============================================
    # 5. Bar plot للمقارنة النسبية
    # ============================================
    ax5 = axes[1, 1]
    
    if valid_names:
        # تطبيع القيم للمقارنة
        sse_means = [np.mean(results[n]['sse']) for n in valid_names]
        min_sse = min(sse_means)
        max_sse = max(sse_means)
        
        if max_sse != min_sse:
            normalized_sse = [(s - min_sse) / (max_sse - min_sse) * 100 for s in sse_means]
        else:
            normalized_sse = [0] * len(sse_means)
        
        bars = ax5.barh(valid_names, normalized_sse, 
                       color=[algorithms[n]['color'] for n in valid_names], alpha=0.7)
        
        # إضافة القيم
        for bar, val, orig in zip(bars, normalized_sse, sse_means):
            ax5.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{orig:.2f}', va='center', fontsize=9)
        
        ax5.set_xlabel('Normalized SSE % (Lower is Better)')
        ax5.set_title('Relative Performance Comparison')
        ax5.grid(True, alpha=0.3)
    
    # ============================================
    # 6. Summary Table
    # ============================================
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')
    
    # إنشاء جدول ملخص
    table_data = []
    for name in valid_names:
        sse_mean = np.mean(results[name]['sse'])
        sse_std = np.std(results[name]['sse'])
        time_mean = np.mean(results[name]['time'])
        
        row = [name, f"{sse_mean:.2f}±{sse_std:.2f}", f"{time_mean:.3f}s"]
        
        if name == best_algo:
            row.append("🏆 BEST")
        else:
            row.append("")
        
        table_data.append(row)
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Algorithm', 'SSE (Mean±Std)', 'Time', 'Status'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.35, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # تلوين صف أفضل خوارزمية بالأخضر
    if best_algo:
        best_idx = valid_names.index(best_algo)
        for j in range(4):
            table[(best_idx + 1, j)].set_facecolor('#90EE90')
    
    ax6.set_title('Summary Table', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# ============================================
# دوال مساعدة للمقارنات السريعة
# ============================================
def compare_classical_only(n_runs=10):
    """مقارنة الخوارزميات الكلاسيكية فقط"""
    algorithms = {
        'K-Means': lambda s: run_kmeans(n_clusters=5, seed=s),
        'K-Means++': lambda s: run_multiple_init(n_clusters=5, n_init=10, seed=s),
        'DBSCAN': lambda s: run_dbscan(eps=0.4, min_samples=5, seed=s)
    }
    
    results = {}
    for name, func in algorithms.items():
        sse_list = []
        time_list = []
        
        print(f"\n🔄 Testing {name}...")
        for run in range(n_runs):
            sse, exec_time, _ = safe_run(func, name, 42+run)
            if sse is not None:
                sse_list.append(sse)
                time_list.append(exec_time)
        
        results[name] = {
            'sse_mean': np.mean(sse_list),
            'sse_std': np.std(sse_list),
            'time_mean': np.mean(time_list)
        }
        
        print(f"  ✅ {name}: SSE={results[name]['sse_mean']:.4f}±{results[name]['sse_std']:.4f}")
    
    return results

def compare_evolutionary_only(n_runs=10):
    """مقارنة الخوارزميات التطورية فقط"""
    algorithms = {
        'GA': run_genetic_algorithm,
        'PSO': run_pso,
        'DE': run_de,
        'ABC': run_abc,
        'ACO': run_aco
    }
    
    results = {}
    for name, func in algorithms.items():
        sse_list = []
        time_list = []
        
        print(f"\n🔄 Testing {name}...")
        for run in range(n_runs):
            sse, exec_time, _ = safe_run(func, name, 42+run)
            if sse is not None:
                sse_list.append(sse)
                time_list.append(exec_time)
        
        results[name] = {
            'sse_mean': np.mean(sse_list),
            'sse_std': np.std(sse_list),
            'time_mean': np.mean(time_list)
        }
        
        print(f"  ✅ {name}: SSE={results[name]['sse_mean']:.4f}±{results[name]['sse_std']:.4f}")
    
    return results

def get_quick_summary(n_runs=5):
    """ملخص سريع لجميع الخوارزميات (بدون رسوم)"""
    results, stats, best_algo = run_comparison_analysis(
        n_runs=n_runs, 
        show_plots=False, 
        verbose=True
    )
    
    print("\n" + "="*70)
    print("📋 QUICK SUMMARY")
    print("="*70)
    
    for row in stats:
        name, algo_type, mean_sse, std_sse, min_sse, max_sse, mean_time, std_time = row[:8]
        print(f"{name:12} ({algo_type:12}): SSE={mean_sse}±{std_sse}, Time={mean_time}s")
    
    print(f"\n🏆 Best: {best_algo}")
    
    return results

# ============================================
# دوال اختبار DBSCAN بمعاملات مختلفة
# ============================================
def test_dbscan_parameters():
    """اختبار DBSCAN بقيم مختلفة من eps و min_samples"""
    eps_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    min_samples_values = [3, 5, 7, 10]
    
    print("\n" + "="*70)
    print("🔵 DBSCAN PARAMETER TESTING")
    print("="*70)
    
    results = {}
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            key = f"eps={eps}, min={min_samples}"
            
            sse_list = []
            clusters_list = []
            noise_list = []
            
            for run in range(5):  # 5 تشغيلات لكل إعداد
                try:
                    labels, n_clusters, n_noise, sse = run_dbscan(
                        eps=eps, min_samples=min_samples, seed=42+run
                    )
                    sse_list.append(sse)
                    clusters_list.append(n_clusters)
                    noise_list.append(n_noise)
                except:
                    pass
            
            if sse_list:
                results[key] = {
                    'sse_mean': np.mean(sse_list),
                    'clusters_mean': np.mean(clusters_list),
                    'noise_mean': np.mean(noise_list),
                    'noise_pct': 100 * np.mean(noise_list) / len(X)
                }
                
                print(f"  {key:25} | SSE={results[key]['sse_mean']:8.2f} | "
                      f"Clusters={results[key]['clusters_mean']:.1f} | "
                      f"Noise={results[key]['noise_pct']:.1f}%")
    
    return results

# ============================================
# التشغيل المباشر
# ============================================
if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("🧬 CLUSTERING ALGORITHMS COMPARISON TOOL")
    print("="*70)
    print("\nOptions:")
    print("  1. Full comparison (all algorithms)")
    print("  2. Quick summary (5 runs, no plots)")
    print("  3. Classical algorithms only")
    print("  4. Evolutionary algorithms only")
    print("  5. DBSCAN parameter testing")
    print("  0. Exit")
    
    try:
        choice = input("\n📌 Enter your choice (0-5): ").strip()
    except:
        choice = "1"
    
    if choice == "1":
        # مقارنة كاملة
        results, stats, best = run_comparison_analysis(n_runs=10, show_plots=True)
        
    elif choice == "2":
        # ملخص سريع
        get_quick_summary(n_runs=5)
        
    elif choice == "3":
        # كلاسيكية فقط
        results = compare_classical_only(n_runs=10)
        
    elif choice == "4":
        # تطورية فقط
        results = compare_evolutionary_only(n_runs=10)
        
    elif choice == "5":
        # اختبار DBSCAN
        results = test_dbscan_parameters()
        
    elif choice == "0":
        print("👋 Goodbye!")
        sys.exit(0)
        
    else:
        print("❌ Invalid choice. Running default comparison...")
        results, stats, best = run_comparison_analysis(n_runs=5, show_plots=True)
    
    print("\n✅ Done!")