import matplotlib.pyplot as plt
import numpy as np

# --- 1. 数据定义 ---
# X轴：核心/进程数
cores = [1, 2, 4, 8, 16, 32]
# 将X轴标签设为字符串，以确保它们是等间距的分类，而不是对数刻度
core_labels = [str(c) for c in cores]

# Y轴：执行时间 (ms)
# 注意：对于单点性能（如串行、CUDA），我们将数据点扩展到所有核心数，以绘制一条水平线
performance_data = {
    "Sequential (SOA)": {
        "times": [1913] * len(cores),
        "color": "#d62728",
        "marker": "s"
    },
    "Auto-Vectorization": {
        "times": [386] * len(cores),
        "color": "#ffdd57",
        "marker": "s"
    },
    "MPI": {
        "times": [1905, 1672, 848, 447, 245, 146],
        "color": "#1f77b4",
        "marker": "o"
    },
    "Pthread": {
        "times": [1964, 1638, 834, 425, 224, 135],
        "color": "#2ca02c",
        "marker": "o"
    },
    "OpenMP": {
        "times": [1905, 1634, 833, 425, 219, 154],
        "color": "#8c564b",
        "marker": "o"
    },
    "CUDA": {
        "times": [1.37] * len(cores),
        "color": "#9467bd",
        "marker": "s"
    },
    "OpenACC": {
        "times": [2.00] * len(cores),
        "color": "#e377c2",
        "marker": "s"
    },
    "Triton": {
        "times": [106.05] * len(cores),
        "color": "#7f7f7f",
        "marker": "s"
    }
}


# --- 2. 绘图设置 ---
# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(12, 7))

# 设置背景色为浅灰色，类似图片风格
fig.patch.set_facecolor('#f0f0f0')
ax.set_facecolor('#ffffff')

# 绘制网格线
ax.grid(axis='y', linestyle='--', color='gray', alpha=0.6)
ax.set_axisbelow(True) # 让网格线在图形下方

# --- 3. 绘制数据曲线 ---
for name, data in performance_data.items():
    ax.plot(core_labels, data["times"], label=name, marker=data["marker"], color=data["color"], linewidth=2)

# --- 4. 设置图表属性 ---
# 设置标题和轴标签
ax.set_title("CSC4005 Project 1 PartC Baseline Performance on 4K-RGB jpg", fontsize=16, pad=20)
ax.set_xlabel("Number of Cores/Processors per task", fontsize=12, labelpad=15)
ax.set_ylabel("Execution Time (ms)", fontsize=12, labelpad=15)

# 设置Y轴范围
ax.set_ylim(0, 2500)

# 设置图例
# ncol参数让图例水平排列，bbox_to_anchor将其精确定位在图表下方
legend = ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=5, # 每行显示5个图例项
    frameon=False, # 不显示图例边框
    fontsize=10
)

# 调整布局以防止标签重叠
plt.tight_layout()
# 为底部的图例和文字留出更多空间
plt.subplots_adjust(bottom=0.2)


# --- 5. 显示图表 ---
plt.show()