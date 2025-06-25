import numpy as np
import matplotlib.pyplot as plt
from time import time

def generate_samples(n_samples):
    """生成满足p(x) = 1/(2√x)分布的随机数"""
    u = np.random.uniform(0, 1, n_samples)
    x = u ** 2  # 逆变换公式x = u²
    return x

def calculate_integrand(x):
    """计算被积函数f(x) = x^(-1/2)/(e^x + 1)"""
    return x ** (-0.5) / (np.exp(x) + 1)

def calculate_weighted_function(x):
    """计算加权函数f(x)/p(x) = 2/(e^x + 1)"""
    return 2 / (np.exp(x) + 1)

def estimate_integral(n_samples=1000000, show_progress=False):
    """估计积分值并返回中间结果"""
    start_time = time()
    
    # 生成随机样本
    x = generate_samples(n_samples)
    
    # 计算加权函数值
    weighted_values = calculate_weighted_function(x)
    
    # 计算积分估计值
    integral_estimate = np.mean(weighted_values)
    
    # 计算运行时间
    end_time = time()
    run_time = end_time - start_time
    
    if show_progress:
        print(f"样本量: {n_samples}")
        print(f"运行时间: {run_time:.4f} 秒")
    
    return integral_estimate, weighted_values, x

def estimate_error(weighted_values):
    """估计积分结果的统计误差"""
    mean_f = np.mean(weighted_values)
    mean_f2 = np.mean(weighted_values ** 2)
    var_f = mean_f2 - mean_f ** 2
    sigma = np.sqrt(var_f) / np.sqrt(len(weighted_values))
    return sigma

def plot_sample_distribution(x, n_bins=100):
    """绘制样本分布直方图并与理论分布对比"""
    x_range = np.linspace(0, 1, 1000)
    theoretical_pdf = 1/(2 * np.sqrt(x_range))  # 理论概率密度函数
    
    plt.figure(figsize=(10, 6))
    plt.hist(x, bins=n_bins, density=True, alpha=0.7, label='样本分布')
    plt.plot(x_range, theoretical_pdf, 'r-', linewidth=2, label='理论分布 p(x)')
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('随机样本分布与理论概率密度对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sample_distribution.png')
    plt.close()

def plot_convergence(weighted_values, step=1000):
    """绘制积分估计值的收敛过程"""
    n_samples = len(weighted_values)
    indices = np.arange(step, n_samples+1, step)
    cumulative_means = []
    
    for i in indices:
        cumulative_means.append(np.mean(weighted_values[:i]))
    
    plt.figure(figsize=(12, 6))
    plt.plot(indices, cumulative_means, 'b-', alpha=0.7)
    plt.axhline(y=0.84, color='r', linestyle='--', label='预期值 0.84')
    plt.xlabel('样本数量')
    plt.ylabel('积分估计值')
    plt.title('积分估计值的收敛过程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('convergence.png')
    plt.close()

def main():
    n_samples = 1000000  # 样本数量
    
    # 估计积分值
    integral_estimate, weighted_values, x = estimate_integral(n_samples, show_progress=True)
    
    # 估计统计误差
    sigma = estimate_error(weighted_values)
    
    # 输出结果
    print(f"积分估计值: {integral_estimate:.6f}")
    print(f"统计误差: {sigma:.6f}")
    print(f"与预期值0.84的偏差: {abs(integral_estimate - 0.84):.6f}")
    
    # 绘制样本分布
    plot_sample_distribution(x)
    print("样本分布直方图已保存为'sample_distribution.png'")
    
    # 绘制收敛过程（使用前10万个样本以提高绘图效率）
    plot_convergence(weighted_values[:100000])
    print("收敛过程图已保存为'convergence.png'")

if __name__ == "__main__":
    main()
