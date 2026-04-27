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
from kmeans_plus_plus import run_kmeans_pp, run_multiple_init
from dbscan import run_dbscan, DBSCAN

# ============================================
# شرح الخوارزميات (محدث)
# ============================================
ALGORITHM_EXPLANATIONS = {
    "K-Means": """
⭐ K-MEANS CLUSTERING ⭐
─────────────────────────────────────────
📌 الفكرة الأساسية:
• تقسيم البيانات إلى K مجموعات
• كل مجموعة لها مركز (Centroid)
• أبسط وأسرع خوارزمية تجميع

🔧 طريقة العمل:
1️⃣ اختيار K مراكز عشوائية
2️⃣ توزيع النقاط على أقرب مركز
3️⃣ تحديث المراكز (متوسط النقاط)
4️⃣ تكرار الخطوات حتى الاستقرار

✅ المميزات:
• بسيط وسريع جداً
• سهل الفهم والتنفيذ

❌ العيوب:
• حساس للمراكز الابتدائية
• قد يعلق في حلول ضعيفة
• يفترض أن المجموعات كروية الشكل
─────────────────────────────────────────
    """,
    
    "K-Means++": """
⭐ K-MEANS++ CLUSTERING ⭐
─────────────────────────────────────────
📌 الفكرة الأساسية:
• نسخة محسنة من K-Means
• تستخدم تهيئة ذكية للمراكز الابتدائية
• المراكز تختار متباعدة عن بعض

🔧 طريقة العمل:
1️⃣ اختيار أول مركز عشوائياً
2️⃣ حساب المسافة لكل نقطة لأقرب مركز
3️⃣ اختيار المركز التالي باحتمال يتناسب مع
   مربع المسافة (النقاط البعيدة فرصتها أكبر)
4️⃣ تكرار حتى الحصول على K مراكز
5️⃣ تشغيل K-Means العادي بهذه المراكز

✅ المميزات:
• نتائج أفضل من K-Means العادي
• تقارب أسرع
• أقل عرضة للحلول الضعيفة

❌ العيوب:
• لا يزال يحتاج تحديد K
• لا يضمن الحل الأمثل عالمياً
─────────────────────────────────────────
    """,

    "DBSCAN": """
🔵 DBSCAN CLUSTERING 🔵
─────────────────────────────────────────
📌 الفكرة الأساسية:
• تجميع يعتمد على كثافة النقاط
• لا يحتاج تحديد عدد المجموعات!
• يكتشف مجموعات بأي شكل
• يكتشف النقاط الشاذة تلقائياً

🔧 طريقة العمل:
1️⃣ لكل نقطة، ابحث عن جيرانها داخل eps
2️⃣ إذا كان عدد الجيران ≥ min_samples:
   • هذه Core Point → ابدأ مجموعة جديدة
   • مدد المجموعة لكل الجيران المتصلين
3️⃣ إذا كانت النقطة جارة لـ Core:
   • هذه Border Point → انضم للمجموعة
4️⃣ باقي النقاط = Noise (ضوضاء/شاذة)

📊 أنواع النقاط:
• Core Point: لها min_samples جار على الأقل
• Border Point: جارة لـ Core لكن ليست Core
• Noise Point: ليست Core ولا Border

✅ المميزات:
• لا يحتاج تحديد K
• يكتشف مجموعات بأشكال عشوائية
• مقاوم للقيم الشاذة

❌ العيوب:
• حساس لاختيار eps و min_samples
• لا يعمل جيداً مع البيانات عالية الأبعاد
• صعوبة مع الكثافات المتفاوتة
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
        self.is_running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # الإطار الرئيسي
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # العنوان
        title_label = ttk.Label(main_frame, text="🧬 Evolutionary & Classical Clustering Algorithms 🧬", 
                               font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # ====== الإطار الأيسر: الأزرار والتحكم ======
        left_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # القسم 1: خوارزميات تقليدية
        ttk.Label(left_frame, text="📊 Classical Algorithms", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=5)
        
        classical_algorithms = [
            ("⭐ K-Means (Standard)", self.run_kmeans_gui),
            ("⭐ K-Means++ (Smart Init)", self.run_kmeans_pp_gui),
            ("🔵 DBSCAN (Density-Based)", self.run_dbscan_gui)
        ]
        
        for i, (name, command) in enumerate(classical_algorithms):
            btn = ttk.Button(left_frame, text=name, command=command, width=35)
            btn.grid(row=i+1, column=0, pady=3)
        
        # فاصل
        ttk.Separator(left_frame, orient='horizontal').grid(row=4, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # القسم 2: خوارزميات تطورية
        ttk.Label(left_frame, text="🧬 Evolutionary Algorithms", 
                 font=('Arial', 12, 'bold')).grid(row=5, column=0, pady=5)
        
        evolutionary_algorithms = [
            ("🧬 Genetic Algorithm (GA)", self.run_ga_gui),
            ("🐦 Particle Swarm Optimization (PSO)", self.run_pso_gui),
            ("🔬 Differential Evolution (DE)", self.run_de_gui),
            ("🐝 Artificial Bee Colony (ABC)", self.run_abc_gui),
            ("🐜 Ant Colony Optimization (ACO)", self.run_aco_gui)
        ]
        
        for i, (name, command) in enumerate(evolutionary_algorithms):
            btn = ttk.Button(left_frame, text=name, command=command, width=35)
            btn.grid(row=i+6, column=0, pady=3)
        
        # فاصل
        ttk.Separator(left_frame, orient='horizontal').grid(row=11, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # زر المقارنة
        compare_btn = ttk.Button(left_frame, text="📊 Compare All Algorithms 📊", 
                                command=self.run_comparison_gui, width=35)
        compare_btn.grid(row=12, column=0, pady=5)
        
        # فاصل
        ttk.Separator(left_frame, orient='horizontal').grid(row=13, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # إعدادات
        ttk.Label(left_frame, text="⚙️ Settings:", 
                 font=('Arial', 12, 'bold')).grid(row=14, column=0, pady=5)
        
        # عدد المجموعات
        settings_frame = ttk.Frame(left_frame)
        settings_frame.grid(row=15, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(settings_frame, text="Number of Clusters:").grid(row=0, column=0, sticky=tk.W)
        self.n_clusters_var = tk.StringVar(value="5")
        n_clusters_spin = ttk.Spinbox(settings_frame, from_=2, to=10, 
                                      textvariable=self.n_clusters_var, width=8)
        n_clusters_spin.grid(row=0, column=1, sticky=tk.E, padx=5)
        
        # عدد مرات المقارنة
        ttk.Label(settings_frame, text="Comparison Runs:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.n_runs_var = tk.StringVar(value="10")
        n_runs_spin = ttk.Spinbox(settings_frame, from_=5, to=30, 
                                  textvariable=self.n_runs_var, width=8)
        n_runs_spin.grid(row=1, column=1, sticky=tk.E, padx=5)
        
        # إعدادات DBSCAN
        ttk.Label(settings_frame, text="DBSCAN eps:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.dbscan_eps_var = tk.StringVar(value="0.4")
        eps_spin = ttk.Spinbox(settings_frame, from_=0.1, to=2.0, increment=0.1,
                              textvariable=self.dbscan_eps_var, width=8)
        eps_spin.grid(row=2, column=1, sticky=tk.E, padx=5)
        
        ttk.Label(settings_frame, text="DBSCAN Min Samples:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.dbscan_min_samples_var = tk.StringVar(value="5")
        min_samples_spin = ttk.Spinbox(settings_frame, from_=2, to=20,
                                      textvariable=self.dbscan_min_samples_var, width=8)
        min_samples_spin.grid(row=3, column=1, sticky=tk.E, padx=5)
        
        # زر الإيقاف
        ttk.Separator(left_frame, orient='horizontal').grid(row=16, column=0, sticky=(tk.W, tk.E), pady=10)
        self.stop_btn = ttk.Button(left_frame, text="🛑 Stop Execution", 
                                   command=self.stop_execution, state='disabled')
        self.stop_btn.grid(row=17, column=0, pady=5)
        
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
        self.status_var = tk.StringVar(value="✅ Ready - Select an algorithm to begin")
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
        """إيقاف التنفيذ"""
        self.is_running = False
        self.status_var.set("⏹️ Stopping...")
        self.stop_btn.config(state='disabled')
        
    def update_progress(self, value):
        """تحديث شريط التقدم"""
        self.progress_var.set(value)
        self.root.update_idletasks()
        
    def display_explanation(self, algo_name):
        """عرض شرح الخوارزمية"""
        self.text_area.delete(1.0, tk.END)
        if algo_name in ALGORITHM_EXPLANATIONS:
            self.text_area.insert(tk.END, ALGORITHM_EXPLANATIONS[algo_name])
        self.text_area.see(tk.END)
        
    def plot_clustering_results(self, centroids, sse, history, algo_name, labels=None):
        """رسم نتائج التجميع"""
        self.ax1.clear()
        self.ax2.clear()
        
        # رسم الـ Clusters
        from Genetic_Algorithm import assign_clusters
        
        if labels is None:
            clusters = assign_clusters(self.X_scaled, self.scaler.transform(centroids))
        else:
            clusters = labels
        
        # رسم النقاط
        unique_labels = set(clusters)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(sorted(unique_labels), colors):
            if label == -1:
                # الضوضاء (Noise) في DBSCAN
                color = [0, 0, 0, 1]
                label_name = 'Noise'
            else:
                label_name = f'Cluster {label}'
            
            mask = clusters == label
            self.ax1.scatter(self.X[mask, 0], self.X[mask, 1], 
                           c=[color], label=label_name, alpha=0.6, s=50)
        
        # رسم المراكز (لو موجودة)
        if centroids is not None:
            self.ax1.scatter(centroids[:, 0], centroids[:, 1], 
                           c='red', marker='X', s=200, edgecolors='black', 
                           linewidths=2, zorder=5, label='Centroids')
        
        self.ax1.set_xlabel('Annual Income (k$)')
        self.ax1.set_ylabel('Spending Score (1-100)')
        self.ax1.set_title(f'{algo_name}\nSSE: {sse:.4f}' if sse else f'{algo_name}')
        self.ax1.legend(loc='upper right', fontsize='small')
        self.ax1.grid(True, alpha=0.3)
        
        # رسم الـ Convergence (لو موجود)
        if history and len(history) > 1:
            self.ax2.plot(history, 'b-', linewidth=2)
            self.ax2.set_xlabel('Iteration')
            self.ax2.set_ylabel('SSE')
            self.ax2.set_title('Convergence Curve')
            self.ax2.grid(True, alpha=0.3)
        else:
            self.ax2.text(0.5, 0.5, 'No convergence data\n(DBSCAN is non-iterative)', 
                         ha='center', va='center', fontsize=12)
            self.ax2.set_title('Convergence Curve')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def append_results(self, algo_name, sse, exec_time, additional_info=None):
        """إضافة النتائج للنص"""
        self.text_area.insert(tk.END, f"\n\n{'='*50}\n")
        self.text_area.insert(tk.END, f"✅ RESULTS: {algo_name}\n")
        self.text_area.insert(tk.END, f"{'='*50}\n")
        
        if sse is not None:
            self.text_area.insert(tk.END, f"📊 SSE: {sse:.4f}\n")
        self.text_area.insert(tk.END, f"⏱️ Execution Time: {exec_time:.4f} seconds\n")
        
        if additional_info:
            for key, value in additional_info.items():
                self.text_area.insert(tk.END, f"📌 {key}: {value}\n")
        
        self.text_area.see(tk.END)
        
    # ============================================
    # تشغيل الخوارزميات الفردية
    # ============================================
    def run_algorithm_thread(self, algo_func, algo_name, **kwargs):
        """تشغيل الخوارزمية في Thread منفصل"""
        if self.is_running:
            messagebox.showwarning("Busy", "Please wait for the current algorithm to finish.")
            return
            
        self.is_running = True
        self.stop_btn.config(state='normal')
        self.status_var.set(f"⏳ Running {algo_name}...")
        self.display_explanation(algo_name)
        self.update_progress(10)
        
        def task():
            start_time = time.time()
            
            try:
                result = algo_func(**kwargs)
                self.update_progress(90)
                exec_time = time.time() - start_time
                
                # تحديث في main thread
                self.root.after(0, lambda: self._handle_result(result, algo_name, exec_time))
                
            except Exception as e:
                self.root.after(0, lambda: self._handle_error(algo_name, str(e)))
                
            finally:
                self.is_running = False
                self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                self.update_progress(100)
        
        threading.Thread(target=task, daemon=True).start()
    
    def _handle_result(self, result, algo_name, exec_time):
        """معالجة نتيجة الخوارزمية"""
        if algo_name == "DBSCAN":
            labels, n_clusters, n_noise, sse = result
            self.plot_clustering_results(None, sse, [], algo_name, labels=labels)
            self.append_results(algo_name, sse, exec_time, {
                'Clusters Found': n_clusters,
                'Noise Points': f'{n_noise} ({100*n_noise/len(self.X):.1f}%)'
            })
            self.status_var.set(f"✅ {algo_name}: {n_clusters} clusters, {n_noise} noise points")
        else:
            # التعامل مع النتائج اللي بترجع قيمتين أو تلاتة
            if len(result) == 3:
                centroids, sse, history = result
            elif len(result) == 2:
                centroids, sse = result
                history = []  # تاريخ فارغ للخوارزميات اللي مش بترجع history
            
            self.plot_clustering_results(centroids, sse, history, algo_name)
            self.append_results(algo_name, sse, exec_time)
            self.status_var.set(f"✅ {algo_name}: SSE={sse:.4f}, Time={exec_time:.2f}s")
        
    def _handle_error(self, algo_name, error_msg):
        """معالجة الأخطاء"""
        messagebox.showerror("Error", f"Error in {algo_name}:\n{error_msg}")
        self.status_var.set(f"❌ Error in {algo_name}")
        self.text_area.insert(tk.END, f"\n❌ ERROR: {error_msg}\n")
    
    # ============================================
    # أزرار الخوارزميات الكلاسيكية
    # ============================================
    def run_kmeans_gui(self):
        self.run_algorithm_thread(
            run_kmeans, "K-Means",
            n_clusters=int(self.n_clusters_var.get())
        )
        
    def run_kmeans_pp_gui(self):
        self.run_algorithm_thread(
            run_multiple_init, "K-Means++",
            n_clusters=int(self.n_clusters_var.get()),
            n_init=10
        )
        
    def run_dbscan_gui(self):
        self.run_algorithm_thread(
            run_dbscan, "DBSCAN",
            eps=float(self.dbscan_eps_var.get()),
            min_samples=int(self.dbscan_min_samples_var.get())
        )
    
    # ============================================
    # أزرار الخوارزميات التطورية
    # ============================================
    def run_ga_gui(self):
        self.run_algorithm_thread(run_genetic_algorithm, "Genetic Algorithm (GA)")
        
    def run_pso_gui(self):
        self.run_algorithm_thread(run_pso, "Particle Swarm Optimization (PSO)")
        
    def run_de_gui(self):
        self.run_algorithm_thread(run_de, "Differential Evolution (DE)")
        
    def run_abc_gui(self):
        self.run_algorithm_thread(run_abc, "Artificial Bee Colony (ABC)")
        
    def run_aco_gui(self):
        self.run_algorithm_thread(run_aco, "Ant Colony Optimization (ACO)")
    
    # ============================================
    # المقارنة الشاملة
    # ============================================
    def run_comparison_gui(self):
        """تشغيل مقارنة شاملة بين جميع الخوارزميات"""
        if self.is_running:
            messagebox.showwarning("Busy", "Please wait for the current algorithm to finish.")
            return
            
        self.is_running = True
        self.stop_btn.config(state='normal')
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "="*60 + "\n")
        self.text_area.insert(tk.END, "📊 COMPREHENSIVE ALGORITHM COMPARISON 📊\n")
        self.text_area.insert(tk.END, "="*60 + "\n\n")
        self.text_area.insert(tk.END, "🔄 Running all algorithms...\n\n")
        self.text_area.see(tk.END)
        
        def comparison_task():
            # تعريف الخوارزميات بدون lambda - مباشرة
            algorithms = {
                'K-Means': {
                    'func': run_kmeans,
                    'kwargs': {'n_clusters': 5}
                },
                'K-Means++': {
                    'func': run_multiple_init,
                    'kwargs': {'n_clusters': 5, 'n_init': 10}
                },
                'DBSCAN': {
                    'func': run_dbscan,
                    'kwargs': {
                        'eps': float(self.dbscan_eps_var.get()),
                        'min_samples': int(self.dbscan_min_samples_var.get())
                    }
                },
                'GA': {
                    'func': run_genetic_algorithm,
                    'kwargs': {}
                },
                'PSO': {
                    'func': run_pso,
                    'kwargs': {}
                },
                'DE': {
                    'func': run_de,
                    'kwargs': {}
                },
                'ABC': {
                    'func': run_abc,
                    'kwargs': {}
                },
                'ACO': {
                    'func': run_aco,
                    'kwargs': {}
                }
            }
            
            results = {name: {'sse': [], 'time': [], 'extra': []} for name in algorithms}
            n_runs = int(self.n_runs_var.get())
            
            for i, (name, algo_info) in enumerate(algorithms.items()):
                if not self.is_running:
                    break
                    
                progress = (i / len(algorithms)) * 100
                self.root.after(0, lambda n=name, idx=i: self.status_var.set(
                    f"⏳ Running {n} ({idx+1}/{len(algorithms)})..."))
                self.update_progress(progress)
                
                self.root.after(0, lambda n=name: self.text_area.insert(tk.END, f"🔄 Testing {n}...\n"))
                self.root.after(0, lambda: self.text_area.see(tk.END))
                
                algo_func = algo_info['func']
                algo_kwargs = algo_info['kwargs']
                
                for run in range(n_runs):
                    if not self.is_running:
                        break
                        
                    start_time = time.time()
                    
                    try:
                        # استدعاء الدالة بالـ seed
                        np.random.seed(42 + run)
                        result = algo_func(**algo_kwargs)
                        
                        if name == 'DBSCAN':
                            # DBSCAN: 4 قيم
                            labels, n_clusters, n_noise, sse = result
                            results[name]['extra'].append({'clusters': n_clusters, 'noise': n_noise})
                        elif name == 'K-Means':
                            # K-Means: 3 قيم (centroids, sse, history)
                            centroids, sse, history = result
                        else:
                            # باقي الخوارزميات: 3 قيم
                            centroids, sse, history = result
                        
                        exec_time = time.time() - start_time
                        results[name]['sse'].append(sse)
                        results[name]['time'].append(exec_time)
                        
                    except Exception as e:
                        import traceback
                        error_msg = traceback.format_exc()
                        # اختصار رسالة الخطأ
                        short_error = str(e)[:200]
                        self.root.after(0, lambda n=name, r=run, err=short_error: 
                            self.text_area.insert(tk.END, f"  ⚠️ Error in {n} run {r}: {err}\n"))
                        print(f"Error in {name} run {run}:\n{error_msg}")  # للـ console
                
                if self.is_running and results[name]['sse']:
                    sse_mean = np.mean(results[name]['sse'])
                    sse_std = np.std(results[name]['sse'])
                    time_mean = np.mean(results[name]['time'])
                    
                    self.root.after(0, lambda n=name, m=sse_mean, s=sse_std, t=time_mean: 
                        self.text_area.insert(tk.END, 
                            f"   ✅ {n}: SSE={m:.4f}±{s:.4f}, Time={t:.4f}s\n"))
                    self.root.after(0, lambda: self.text_area.see(tk.END))
            
            if self.is_running:
                self.root.after(0, lambda: self._show_comparison_results(results, algorithms))
            
            self.is_running = False
            self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
            self.update_progress(100)
        
        threading.Thread(target=comparison_task, daemon=True).start()

    def _show_comparison_results(self, results, algorithms):
        """عرض نتائج المقارنة"""
        self.text_area.insert(tk.END, "\n" + "="*60 + "\n")
        self.text_area.insert(tk.END, "📈 FINAL COMPARISON RESULTS 📈\n")
        self.text_area.insert(tk.END, "="*60 + "\n\n")
        
        # إيجاد أفضل خوارزمية من حيث SSE
        valid_algos = {n: r for n, r in results.items() if r['sse']}
        if valid_algos:
            best_algo = min(valid_algos.keys(), 
                        key=lambda x: np.mean(valid_algos[x]['sse']))
            
            # جدول ملخص
            for name in algorithms:
                if results[name]['sse']:
                    sse_mean = np.mean(results[name]['sse'])
                    sse_std = np.std(results[name]['sse'])
                    time_mean = np.mean(results[name]['time'])
                    
                    status = "🏆 BEST" if name == best_algo else ""
                    
                    extra_info = ""
                    if name == 'DBSCAN' and results[name]['extra']:
                        avg_clusters = np.mean([e['clusters'] for e in results[name]['extra']])
                        avg_noise = np.mean([e['noise'] for e in results[name]['extra']])
                        extra_info = f" | Clusters: {avg_clusters:.1f}, Noise: {avg_noise:.1f}"
                    
                    self.text_area.insert(tk.END, 
                        f"  {name:12} | SSE: {sse_mean:8.4f} ± {sse_std:6.4f} | "
                        f"Time: {time_mean:6.4f}s{extra_info} {status}\n")
            
            self.text_area.insert(tk.END, f"\n🏆 Best Algorithm: {best_algo}\n")
            self.text_area.see(tk.END)
            
            # رسم المقارنة
            self._plot_comparison_charts(results, algorithms, best_algo)
            
            self.status_var.set(f"✅ Comparison completed! Best: {best_algo}")
        else:
            self.text_area.insert(tk.END, "❌ No valid results.\n")
            self.status_var.set("❌ Comparison failed")
    
    def _plot_comparison_charts(self, results, algorithms, best_algo):
        """رسم مخططات المقارنة"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Boxplot للـ SSE
        valid_algos = [n for n in algorithms if results[n]['sse']]
        sse_data = [results[n]['sse'] for n in valid_algos]
        
        if sse_data:
            bp = self.ax1.boxplot(sse_data, labels=valid_algos, patch_artist=True)
            
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
            for patch, color in zip(bp['boxes'], colors[:len(valid_algos)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            self.ax1.set_ylabel('SSE (Lower is Better)')
            self.ax1.set_title(f'Algorithm Performance Comparison\n🏆 Best: {best_algo}')
            self.ax1.tick_params(axis='x', rotation=45)
            self.ax1.grid(True, alpha=0.3)
        
        # Bar plot للوقت
        time_means = [np.mean(results[n]['time']) for n in valid_algos]
        bars = self.ax2.bar(valid_algos, time_means, color=colors[:len(valid_algos)], alpha=0.7)
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