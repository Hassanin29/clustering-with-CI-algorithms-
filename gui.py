import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import time
import threading
from sklearn.preprocessing import StandardScaler

# Import all algorithms
from Genetic_Algorithm import run_genetic_algorithm, X, scaler as ga_scaler
from Particle_Swarm_Optimization import run_pso
from Differential_Evolution import run_de
from Artificial_Bee_Colony import run_abc
from Ant_Colony_Optimization import run_aco
from kmeans import run_kmeans

# ============================================
# شرح الخوارزميات (نفس الكود السابق)
# ============================================
ALGORITHM_EXPLANATIONS = {
    "K-Means": """
⭐ K-MEANS CLUSTERING ⭐
─────────────────────────────────────────
📌 الفكرة الأساسية:
• تقسيم البيانات إلى K مجموعات
• كل مجموعة لها مركز (Centroid)

🔧 طريقة العمل:
1️⃣ اختيار K مراكز عشوائية
2️⃣ توزيع النقاط على أقرب مركز
3️⃣ تحديث المراكز (متوسط النقاط)
4️⃣ تكرار الخطوات حتى الاستقرار

✅ المميزات: بسيط وسريع
❌ العيوب: قد يعلق في حلول ضعيفة
─────────────────────────────────────────
    """,
    
    "Genetic Algorithm (GA)": """
🧬 GENETIC ALGORITHM (GA) 🧬
─────────────────────────────────────────
📌 الفكرة الأساسية:
• محاكاة نظرية التطور والانتخاب الطبيعي
• البقاء للأصلح (Survival of the Fittest)

🔧 طريقة العمل:
1️⃣ إنشاء مجتمع عشوائي من الحلول (Chromosomes)
2️⃣ تقييم كل حل باستخدام Fitness Function
3️⃣ اختيار أفضل الحلول كآباء (Selection)
4️⃣ تزاوج الآباء لإنتاج أبناء جدد (Crossover)
5️⃣ إضافة طفرات عشوائية (Mutation)
6️⃣ تكرار العملية لعدة أجيال

🧩 في مشروعنا:
• كل Chromosome = مراكز الـ Clusters
• الـ Fitness = -SSE (كلما قل الـ SSE زاد الـ Fitness)
• التزاوج = One-Point Crossover
• الطفرة = Gaussian Mutation
─────────────────────────────────────────
    """,
    
    "Particle Swarm Optimization (PSO)": """
🐦 PARTICLE SWARM OPTIMIZATION (PSO) 🐦
─────────────────────────────────────────
📌 الفكرة الأساسية:
• محاكاة سلوك أسراب الطيور أو أسراب الأسماك
• كل طائر (Particle) يبحث عن الطعام ويشارك المعلومات

🔧 طريقة العمل:
1️⃣ مجموعة من الجسيمات تتحرك في فضاء البحث
2️⃣ كل جسيم يتذكر أفضل موقع زاره (pbest)
3️⃣ السرب كله يتذكر أفضل موقع عام (gbest)
4️⃣ الجسيمات تحدث سرعتها وموقعها بناءً على:
   • خبرتها الشخصية (pbest)
   • خبرة المجموعة (gbest)

🧩 في مشروعنا:
• كل Particle = مجموعة مراكز للـ Clusters
• السرعة = مقدار التغيير في المراكز
• الـ Fitness = -SSE
─────────────────────────────────────────
    """,
    
    "Differential Evolution (DE)": """
🔬 DIFFERENTIAL EVOLUTION (DE) 🔬
─────────────────────────────────────────
📌 الفكرة الأساسية:
• خوارزمية تطورية تستخدم الفروق بين الحلول
• لا تحتاج لعمليات Selection معقدة

🔧 طريقة العمل:
1️⃣ لكل حل في المجتمع:
   • اختيار 3 حلول عشوائية مختلفة (a, b, c)
   • إنشاء Mutant = a + F × (b - c)
2️⃣ Crossover بين الحل الأصلي والـ Mutant
3️⃣ اختيار الأفضل بين Trial والأصلي (Greedy)

🧩 في مشروعنا:
• F = 0.8 (معامل التفاضل)
• CR = 0.9 (احتمالية التزاوج)
• نقارن مباشرة باستخدام SSE
─────────────────────────────────────────
    """,
    
    "Artificial Bee Colony (ABC)": """
🐝 ARTIFICIAL BEE COLONY (ABC) 🐝
─────────────────────────────────────────
📌 الفكرة الأساسية:
• محاكاة سلوك النحل في البحث عن الرحيق
• ثلاثة أنواع من النحل: عامل، مراقب، كشاف

🔧 طريقة العمل:
1️⃣ النحل العامل (Employed Bees):
   • يستكشف مناطق جديدة قريبة من مصدر طعامه
2️⃣ النحل المراقب (Onlooker Bees):
   • يختار أفضل المصادر بناءً على جودتها
3️⃣ النحل الكشاف (Scout Bees):
   • يبحث عن مصادر جديدة إذا لم يتحسن المصدر

🧩 في مشروعنا:
• كل مصدر طعام = مجموعة مراكز Clusters
• جودة المصدر = 1/(1+SSE)
• الـ Limit = 10 (عدد المحاولات قبل الهجر)
─────────────────────────────────────────
    """,
    
    "Ant Colony Optimization (ACO)": """
🐜 ANT COLONY OPTIMIZATION (ACO) 🐜
─────────────────────────────────────────
📌 الفكرة الأساسية:
• محاكاة سلوك النمل في البحث عن أقصر طريق
• استخدام الفيرمونات للتواصل

🔧 طريقة العمل:
1️⃣ النمل يتحرك عشوائياً ويترك أثر فيرمون
2️⃣ النمل الآخر يتبع المسارات ذات الفيرمون الأعلى
3️⃣ تبخير الفيرمونات مع الزمن
4️⃣ تقوية المسارات الجيدة بفيرمونات إضافية

🧩 في مشروعنا:
• تم تكييف ACO للـ Clustering
• كل نملة تبني مجموعة مراكز
• الفيرمونات = توزيع احتمالي لكل بعد
• الجودة = 1/(1+SSE)
─────────────────────────────────────────
    """
}

# ============================================
# الواجهة الرئيسية
# ============================================
class ClusteringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧬 Evolutionary Clustering Algorithms 🧬")
        self.root.geometry("1400x900")
        
        # تحميل البيانات
        self.data_path = r"D:\ci\code\cluster_project\Mall_Customers.csv"
        self.df = pd.read_csv(self.data_path)
        self.X = self.df[['Annual Income (k$)', 'Spending Score (1-100)']].values
        
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # متغيرات
        self.current_algorithm = None
        self.is_running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # الإطار الرئيسي
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # العنوان
        title_label = ttk.Label(main_frame, text="🧬 Evolutionary Clustering Algorithms 🧬", 
                               font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # ====== الإطار الأيسر: الأزرار والتحكم ======
        left_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # أزرار الخوارزميات
        ttk.Label(left_frame, text="Select Algorithm:", font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=5)
        
        algorithms = [
            ("⭐ K-Means (Baseline)", self.run_kmeans_gui),
            ("🧬 Genetic Algorithm (GA)", self.run_ga_gui),
            ("🐦 Particle Swarm Optimization (PSO)", self.run_pso_gui),
            ("🔬 Differential Evolution (DE)", self.run_de_gui),
            ("🐝 Artificial Bee Colony (ABC)", self.run_abc_gui),
            ("🐜 Ant Colony Optimization (ACO)", self.run_aco_gui)
        ]
        
        for i, (name, command) in enumerate(algorithms):
            btn = ttk.Button(left_frame, text=name, command=command, width=35)
            btn.grid(row=i+1, column=0, pady=3)
        
        # زر المقارنة
        ttk.Separator(left_frame, orient='horizontal').grid(row=len(algorithms)+1, column=0, sticky=(tk.W, tk.E), pady=10)
        compare_btn = ttk.Button(left_frame, text="📊 Compare All Algorithms 📊", 
                                command=self.run_comparison_gui, width=35)
        compare_btn.grid(row=len(algorithms)+2, column=0, pady=5)
        
        # إعدادات
        ttk.Separator(left_frame, orient='horizontal').grid(row=len(algorithms)+3, column=0, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(left_frame, text="Settings:", font=('Arial', 12, 'bold')).grid(row=len(algorithms)+4, column=0, pady=5)
        
        ttk.Label(left_frame, text="Number of Clusters:").grid(row=len(algorithms)+5, column=0, sticky=tk.W)
        self.n_clusters_var = tk.StringVar(value="5")
        n_clusters_spin = ttk.Spinbox(left_frame, from_=2, to=10, textvariable=self.n_clusters_var, width=10)
        n_clusters_spin.grid(row=len(algorithms)+5, column=0, sticky=tk.E, pady=2)
        
        ttk.Label(left_frame, text="Comparison Runs:").grid(row=len(algorithms)+6, column=0, sticky=tk.W)
        self.n_runs_var = tk.StringVar(value="10")
        n_runs_spin = ttk.Spinbox(left_frame, from_=5, to=30, textvariable=self.n_runs_var, width=10)
        n_runs_spin.grid(row=len(algorithms)+6, column=0, sticky=tk.E, pady=2)
        
        # زر الإيقاف
        ttk.Separator(left_frame, orient='horizontal').grid(row=len(algorithms)+7, column=0, sticky=(tk.W, tk.E), pady=10)
        self.stop_btn = ttk.Button(left_frame, text="🛑 Stop", command=self.stop_execution, state='disabled')
        self.stop_btn.grid(row=len(algorithms)+8, column=0, pady=5)
        
        # ====== الإطار الأوسط: الرسم البياني ======
        middle_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
        middle_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=middle_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # ====== الإطار الأيمن: الشرح والنتائج ======
        right_frame = ttk.LabelFrame(main_frame, text="Algorithm Explanation & Results", padding="10")
        right_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # منطقة النص للشرح والنتائج
        self.text_area = scrolledtext.ScrolledText(right_frame, width=50, height=35, 
                                                   font=('Consolas', 10), wrap=tk.WORD)
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, length=400, variable=self.progress_var)
        self.progress_bar.grid(row=2, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # تكوين الأوزان
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
    # ============================================
    # دوال التشغيل الأساسية
    # ============================================
    def stop_execution(self):
        self.is_running = False
        self.status_var.set("Stopping...")
        
    def update_progress(self, value):
        self.progress_var.set(value)
        self.root.update_idletasks()
        
    def display_explanation(self, algo_name):
        self.text_area.delete(1.0, tk.END)
        if algo_name in ALGORITHM_EXPLANATIONS:
            self.text_area.insert(tk.END, ALGORITHM_EXPLANATIONS[algo_name])
        self.text_area.see(tk.END)
        
    def plot_results(self, centroids, sse, history, algo_name):
        """رسم النتائج"""
        self.ax1.clear()
        self.ax2.clear()
        
        # رسم الـ Clusters
        from Genetic_Algorithm import assign_clusters
        clusters = assign_clusters(self.X_scaled, self.scaler.transform(centroids))
        
        scatter = self.ax1.scatter(self.X[:, 0], self.X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        self.ax1.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolors='black')
        self.ax1.set_xlabel('Annual Income (k$)')
        self.ax1.set_ylabel('Spending Score (1-100)')
        self.ax1.set_title(f'{algo_name} - SSE: {sse:.4f}')
        self.ax1.grid(True, alpha=0.3)
        
        # رسم الـ Convergence
        if len(history) > 1:
            self.ax2.plot(history, 'b-', linewidth=2)
            self.ax2.set_xlabel('Iteration')
            self.ax2.set_ylabel('SSE')
            self.ax2.set_title('Convergence Curve')
            self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def append_results(self, algo_name, sse, exec_time):
        """إضافة النتائج للنص"""
        self.text_area.insert(tk.END, f"\n\n{'='*50}\n")
        self.text_area.insert(tk.END, f"✅ RESULTS for {algo_name}\n")
        self.text_area.insert(tk.END, f"{'='*50}\n")
        self.text_area.insert(tk.END, f"📊 Final SSE: {sse:.4f}\n")
        self.text_area.insert(tk.END, f"⏱️ Execution Time: {exec_time:.4f} seconds\n")
        self.text_area.see(tk.END)
        
    # ============================================
    # تشغيل الخوارزميات الفردية
    # ============================================
    def run_algorithm_thread(self, algo_func, algo_name):
        """تشغيل الخوارزمية في Thread منفصل"""
        self.is_running = True
        self.stop_btn.config(state='normal')
        self.status_var.set(f"Running {algo_name}...")
        self.display_explanation(algo_name)
        self.update_progress(10)
        
        start_time = time.time()
        
        try:
            if algo_name == "K-Means":
                centroids, sse = algo_func(n_clusters=int(self.n_clusters_var.get()))
                history = [sse]
            else:
                centroids, sse, history = algo_func()
            
            self.update_progress(90)
            exec_time = time.time() - start_time
            
            # تحديث في main thread
            self.root.after(0, lambda: self.plot_results(centroids, sse, history, algo_name))
            self.root.after(0, lambda: self.append_results(algo_name, sse, exec_time))
            
            self.update_progress(100)
            self.status_var.set(f"✅ {algo_name} completed! SSE: {sse:.4f}, Time: {exec_time:.2f}s")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set(f"❌ Error in {algo_name}")
            
        finally:
            self.is_running = False
            self.stop_btn.config(state='disabled')
            
    def run_kmeans_gui(self):
        if self.is_running:
            return
        threading.Thread(target=self.run_algorithm_thread, 
                        args=(run_kmeans, "K-Means"), daemon=True).start()
        
    def run_ga_gui(self):
        if self.is_running:
            return
        threading.Thread(target=self.run_algorithm_thread, 
                        args=(run_genetic_algorithm, "Genetic Algorithm (GA)"), daemon=True).start()
        
    def run_pso_gui(self):
        if self.is_running:
            return
        threading.Thread(target=self.run_algorithm_thread, 
                        args=(run_pso, "Particle Swarm Optimization (PSO)"), daemon=True).start()
        
    def run_de_gui(self):
        if self.is_running:
            return
        threading.Thread(target=self.run_algorithm_thread, 
                        args=(run_de, "Differential Evolution (DE)"), daemon=True).start()
        
    def run_abc_gui(self):
        if self.is_running:
            return
        threading.Thread(target=self.run_algorithm_thread, 
                        args=(run_abc, "Artificial Bee Colony (ABC)"), daemon=True).start()
        
    def run_aco_gui(self):
        if self.is_running:
            return
        threading.Thread(target=self.run_algorithm_thread, 
                        args=(run_aco, "Ant Colony Optimization (ACO)"), daemon=True).start()
        
    # ============================================
    # المقارنة الشاملة (تم إصلاحها)
    # ============================================
    def run_comparison_gui(self):
        """تشغيل مقارنة شاملة - نسخة مبسطة بدون استدعاء comparison.py"""
        if self.is_running:
            return
            
        self.is_running = True
        self.stop_btn.config(state='normal')
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "="*60 + "\n")
        self.text_area.insert(tk.END, "📊 COMPREHENSIVE COMPARISON 📊\n")
        self.text_area.insert(tk.END, "="*60 + "\n\n")
        
        def run_comparison():
            algorithms = {
                'K-Means': run_kmeans,
                'GA': run_genetic_algorithm,
                'PSO': run_pso,
                'DE': run_de,
                'ABC': run_abc,
                'ACO': run_aco
            }
            
            results = {name: {'sse': [], 'time': []} for name in algorithms}
            n_runs = int(self.n_runs_var.get())
            
            for i, (name, algo_func) in enumerate(algorithms.items()):
                if not self.is_running:
                    break
                    
                self.root.after(0, lambda n=name, idx=i: self.status_var.set(
                    f"Running {n} ({idx+1}/{len(algorithms)})..."))
                self.update_progress((i / len(algorithms)) * 100)
                
                self.root.after(0, lambda n=name: self.text_area.insert(tk.END, f"🔄 Testing {n}...\n"))
                self.root.after(0, lambda: self.text_area.see(tk.END))
                
                for run in range(n_runs):
                    if not self.is_running:
                        break
                        
                    start_time = time.time()
                    
                    try:
                        if name == 'K-Means':
                            _, sse = algo_func(n_clusters=int(self.n_clusters_var.get()), 
                                              seed=42+run)
                        else:
                            _, sse, _ = algo_func(seed=42+run)
                            
                        exec_time = time.time() - start_time
                        
                        results[name]['sse'].append(sse)
                        results[name]['time'].append(exec_time)
                        
                    except Exception as e:
                        print(f"Error in {name} run {run}: {e}")
                
                if self.is_running:
                    self.root.after(0, lambda n=name: self.text_area.insert(tk.END, 
                        f"   ✅ Completed {n_runs} runs\n"))
                    self.root.after(0, lambda: self.text_area.see(tk.END))
            
            if self.is_running:
                # عرض النتائج
                self.root.after(0, lambda: self.text_area.insert(tk.END, "\n" + "="*60 + "\n"))
                self.root.after(0, lambda: self.text_area.insert(tk.END, "📈 RESULTS SUMMARY 📈\n"))
                self.root.after(0, lambda: self.text_area.insert(tk.END, "="*60 + "\n\n"))
                
                # إيجاد أفضل خوارزمية
                best_algo = min(results.keys(), key=lambda x: np.mean(results[x]['sse']))
                
                summary_lines = []
                for name in algorithms:
                    if results[name]['sse']:
                        sse_mean = np.mean(results[name]['sse'])
                        sse_std = np.std(results[name]['sse'])
                        time_mean = np.mean(results[name]['time'])
                        
                        status = "🏆 BEST" if name == best_algo else ""
                        summary_lines.append(
                            f"{name:10} | SSE: {sse_mean:8.4f} ± {sse_std:6.4f} | "
                            f"Time: {time_mean:6.4f}s {status}\n"
                        )
                
                for line in summary_lines:
                    self.root.after(0, lambda l=line: self.text_area.insert(tk.END, l))
                
                self.root.after(0, lambda: self.text_area.insert(tk.END, f"\n✅ Best Algorithm: {best_algo}\n"))
                self.root.after(0, lambda: self.text_area.see(tk.END))
                
                # رسم المقارنة
                self.root.after(0, lambda: self.plot_comparison_results(results, algorithms.keys(), best_algo))
                
                self.status_var.set(f"✅ Comparison completed! Best: {best_algo}")
                self.update_progress(100)
            
            self.is_running = False
            self.stop_btn.config(state='disabled')
            
        threading.Thread(target=run_comparison, daemon=True).start()
    
    def plot_comparison_results(self, results, algo_names, best_algo):
        """رسم نتائج المقارنة"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Boxplot للـ SSE
        sse_data = [results[name]['sse'] for name in algo_names if results[name]['sse']]
        labels = [name for name in algo_names if results[name]['sse']]
        
        if sse_data:
            bp = self.ax1.boxplot(sse_data, labels=labels, patch_artist=True)
            
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
            for patch, color in zip(bp['boxes'], colors[:len(labels)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            self.ax1.set_ylabel('SSE')
            self.ax1.set_title(f'Algorithm Performance - Best: {best_algo}')
            self.ax1.grid(True, alpha=0.3)
        
        # Bar plot للوقت
        time_means = [np.mean(results[name]['time']) for name in labels]
        bars = self.ax2.bar(labels, time_means, color=colors[:len(labels)], alpha=0.7)
        self.ax2.set_ylabel('Time (seconds)')
        self.ax2.set_title('Average Execution Time')
        self.ax2.tick_params(axis='x', rotation=45)
        self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()

# ============================================
# تشغيل البرنامج
# ============================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ClusteringGUI(root)
    root.mainloop()