import numpy as np
import pandas as pd

#这是我们根据词频做的AHP矩阵计算代码

# ==================== 1. 词频数据（基于 PDF 解析结果）====================
# 需要过滤的无意义词
stop_english = {"al", "et", "cid", "https", "crossref", "doi", "china"}  # china 作为地理词可忽略
english_freq = {k: v for k, v in english_freq_raw.items() if k not in stop_english}

# ==================== 2. 关键词 → 维度映射 ====================
# 根据论文表2、3、4的定义进行归类
mapping = {
    # 产业基础 (B) : 企业水平、产业经济、人力资源、技术研发等
    "产业基础": [
        "产业", "发展", "技术", "系统", "无人机", "航空器", "飞行器", "设计", "算法",
        "资源", "实现", "安全", "经济", "研究", "数据", "分析",  # 中文
        "aircraft", "evtol", "uav", "technology", "design", "system", "development",
        "research", "data", "analysis", "aviation", "safety"
    ],
    # 产业布局 (L) : 基础设施、空间分布、空域、城市、飞行运行等
    "产业布局": [
        "空域", "城市", "空间", "规划", "飞行", "运行", "场景", "环境", "基础设施",
        "low-altitude", "urban", "mobility", "transportation", "flight", "operations",
        "vertiport", "uam", "infrastructure", "air"
    ],
    # 产业协同 (S) : 协同、政策、管理、优化、风险、应用、创新等
    "产业协同": [
        "协同", "优化", "风险", "管理", "政策", "应用", "活动", "领域",
        # 英文中无明显对应词，保留少量相关的
        "analysis"  # 分析可支撑协同决策
    ]
}

# 注意：去除重复归类（例如 safety 已放入基础，不再放布局）
# 构建中文词频字典（直接使用）
# 构建英文词频字典（直接使用）

# ==================== 3. 计算各维度的总词频 ====================
def get_total_freq(dimension_keywords):
    total = 0
    for kw in dimension_keywords:
        # 中文词频
        if kw in chinese_freq:
            total += chinese_freq[kw]
        # 英文词频
        if kw in english_freq:
            total += english_freq[kw]
    return total

dimensions = ["产业基础", "产业布局", "产业协同"]
freqs = []
for dim in dimensions:
    kw_list = mapping[dim]
    f = get_total_freq(kw_list)
    freqs.append(f)
    print(f"{dim} 总词频: {f}")

# ==================== 4. 构造 AHP 判断矩阵 ====================
# 使用词频比构造，并限制在 1/9 ~ 9 之间
n = len(dimensions)
A = np.ones((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            A[i, j] = 1
        else:
            ratio = freqs[i] / freqs[j]
            # 映射到 Saaty 1-9 标度（取最接近的整数标度）
            if ratio >= 1:
                if ratio < 1.2:
                    val = 1
                elif ratio < 1.5:
                    val = 2
                elif ratio < 1.8:
                    val = 3
                elif ratio < 2.2:
                    val = 4
                elif ratio < 2.8:
                    val = 5
                elif ratio < 3.5:
                    val = 6
                elif ratio < 4.5:
                    val = 7
                elif ratio < 6.0:
                    val = 8
                else:
                    val = 9
            else:
                inv_ratio = 1 / ratio
                if inv_ratio < 1.2:
                    val = 1
                elif inv_ratio < 1.5:
                    val = 1/2
                elif inv_ratio < 1.8:
                    val = 1/3
                elif inv_ratio < 2.2:
                    val = 1/4
                elif inv_ratio < 2.8:
                    val = 1/5
                elif inv_ratio < 3.5:
                    val = 1/6
                elif inv_ratio < 4.5:
                    val = 1/7
                elif inv_ratio < 6.0:
                    val = 1/8
                else:
                    val = 1/9
            A[i, j] = val
    for j in range(i):
        A[i, j] = 1 / A[j, i]

print("\n判断矩阵 A（维度间）:")
print(np.round(A, 3))

# ==================== 5. 计算权重（几何平均法）====================
# 计算每行几何平均
geom_mean = np.exp(np.mean(np.log(A), axis=1))
weights = geom_mean / np.sum(geom_mean)
print("\n维度权重 (几何平均法):")
for dim, w in zip(dimensions, weights):
    print(f"{dim}: {w:.4f}")

# ==================== 6. 一致性检验 ====================
# 计算最大特征值 λ_max
lambda_max = 0
for i in range(n):
    row_sum = np.dot(A[i, :], weights)
    lambda_max += row_sum / weights[i]
lambda_max /= n

CI = (lambda_max - n) / (n - 1)
# n=3 时的 RI 值 (这是我们根据Saaty表导出的)
RI_dict = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45}
RI = RI_dict[n]
CR = CI / RI

print(f"\n一致性检验:")
print(f"λ_max = {lambda_max:.4f}")
print(f"CI = {CI:.4f}")
print(f"RI = {RI}")
print(f"CR = {CR:.4f}")
if CR < 0.1:
    print("CR < 0.1，判断矩阵一致性通过。")
else:
    print("CR >= 0.1，判断矩阵不一致，需要调整。")
