import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

class NuclearChainReaction:
    """模拟核链式反应的类"""
    
    def __init__(self, size=100, initial_fissile=0.3, initial_neutrons=10, 
                 fission_prob=0.5, absorption_prob=0.1, reproduction=2.5, 
                 diffusion=0.2, max_neutrons=10000):
        """
        初始化链式反应模拟参数
        
        参数:
            size: 模拟网格的大小
            initial_fissile: 初始可裂变物质的比例
            initial_neutrons: 初始中子数量
            fission_prob: 中子撞击可裂变物质引发裂变的概率
            absorption_prob: 中子被吸收的概率
            reproduction: 每次裂变产生的平均中子数
            diffusion: 中子扩散的概率
            max_neutrons: 最大中子数，用于防止计算过载
        """
        self.size = size
        self.fission_prob = fission_prob
        self.absorption_prob = absorption_prob
        self.reproduction = reproduction
        self.diffusion = diffusion
        self.max_neutrons = max_neutrons
        
        # 初始化可裂变物质分布（1表示可裂变，0表示不可裂变）
        self.fissile = np.random.rand(size, size) < initial_fissile
        
        # 初始化中子分布
        self.neutrons = np.zeros((size, size), dtype=int)
        for _ in range(initial_neutrons):
            x, y = np.random.randint(0, size, 2)
            self.neutrons[x, y] += 1
            
        # 记录模拟数据
        self.time_steps = []
        self.total_fissile = []
        self.total_neutrons = []
        
    def simulate_step(self):
        """模拟一个时间步的链式反应"""
        new_neutrons = np.zeros_like(self.neutrons)
        fissions = 0
        
        # 遍历网格中的每个位置
        for i in range(self.size):
            for j in range(self.size):
                # 获取当前位置的中子数
                n = self.neutrons[i, j]
                if n == 0:
                    continue
                    
                # 计算裂变和吸收的中子数
                for _ in range(n):
                    # 中子可能引发裂变
                    if self.fissile[i, j] and np.random.rand() < self.fission_prob:
                        fissions += 1
                        self.fissile[i, j] = False  # 消耗可裂变物质
                        
                        # 产生新的中子（泊松分布模拟随机性）
                        new_n = np.random.poisson(self.reproduction)
                        
                        # 扩散新产生的中子
                        for _ in range(new_n):
                            if np.random.rand() < self.diffusion:
                                # 扩散到随机相邻位置
                                dx, dy = np.random.choice([-1, 0, 1], 2)
                                ni, nj = i + dx, j + dy
                                # 确保在网格范围内
                                if 0 <= ni < self.size and 0 <= nj < self.size:
                                    new_neutrons[ni, nj] += 1
                            else:
                                # 留在原地
                                new_neutrons[i, j] += 1
                    
                    # 中子可能被吸收
                    elif np.random.rand() < self.absorption_prob:
                        continue
                        
                    # 否则中子继续存在并可能扩散
                    else:
                        if np.random.rand() < self.diffusion:
                            dx, dy = np.random.choice([-1, 0, 1], 2)
                            ni, nj = i + dx, j + dy
                            if 0 <= ni < self.size and 0 <= nj < self.size:
                                new_neutrons[ni, nj] += 1
                        else:
                            new_neutrons[i, j] += 1
        
        # 更新中子分布
        self.neutrons = new_neutrons
        
        # 限制最大中子数以防止计算过载
        if np.sum(self.neutrons) > self.max_neutrons:
            scale = self.max_neutrons / np.sum(self.neutrons)
            self.neutrons = (self.neutrons * scale).astype(int)
        
        # 记录数据
        self.time_steps.append(len(self.time_steps))
        self.total_fissile.append(np.sum(self.fissile))
        self.total_neutrons.append(np.sum(self.neutrons))
        
        return fissions
    
    def simulate(self, steps=100):
        """
        模拟多个时间步
        
        参数:
            steps: 模拟的时间步数
        """
        for _ in range(steps):
            self.simulate_step()
            
    def visualize(self):
        """可视化模拟结果"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 创建自定义颜色映射
        cmap = LinearSegmentedColormap.from_list('nuclear_cmap', 
                                                 [(0, 'white'), (0.5, 'yellow'), (1, 'red')])
        
        # 显示最终的可裂变物质分布
        im1 = ax1.imshow(self.fissile, cmap='binary', interpolation='nearest')
        ax1.set_title('可裂变物质分布')
        ax1.set_xlabel('X 坐标')
        ax1.set_ylabel('Y 坐标')
        plt.colorbar(im1, ax=ax1, label='可裂变物质 (1=是, 0=否)')
        
        # 显示最终的中子分布
        im2 = ax2.imshow(self.neutrons, cmap=cmap, interpolation='nearest', 
                        norm=plt.Normalize(0, np.max(self.neutrons)))
        ax2.set_title('中子分布')
        ax2.set_xlabel('X 坐标')
        ax2.set_ylabel('Y 坐标')
        plt.colorbar(im2, ax=ax2, label='中子数')
        
        # 绘制随时间变化的趋势图
        ax3.plot(self.time_steps, self.total_fissile, 'b-', label='可裂变物质')
        ax3.plot(self.time_steps, self.total_neutrons, 'r-', label='中子数')
        ax3.set_title('链式反应随时间的演化')
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('数量')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def animate(self, steps=100, interval=200):
        """
        创建链式反应的动画
        
        参数:
            steps: 动画的步数
            interval: 帧间隔时间（毫秒）
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 创建自定义颜色映射
        cmap = LinearSegmentedColormap.from_list('nuclear_cmap', 
                                                 [(0, 'white'), (0.5, 'yellow'), (1, 'red')])
        
        # 初始化图像
        im1 = ax1.imshow(self.fissile, cmap='binary', interpolation='nearest')
        ax1.set_title('可裂变物质分布')
        ax1.set_xlabel('X 坐标')
        ax1.set_ylabel('Y 坐标')
        plt.colorbar(im1, ax=ax1, label='可裂变物质 (1=是, 0=否)')
        
        im2 = ax2.imshow(self.neutrons, cmap=cmap, interpolation='nearest', 
                        norm=plt.Normalize(0, 10))  # 固定颜色范围以便于观察
        ax2.set_title('中子分布')
        ax2.set_xlabel('X 坐标')
        ax2.set_ylabel('Y 坐标')
        plt.colorbar(im2, ax=ax2, label='中子数')
        
        # 裂变数和时间步的文本显示
        text = fig.text(0.5, 0.01, f'时间步: 0, 裂变数: 0', ha='center')
        
        def update(frame):
            """更新动画帧"""
            fissions = self.simulate_step()
            im1.set_data(self.fissile)
            im2.set_data(self.neutrons)
            im2.set_norm(plt.Normalize(0, max(10, np.max(self.neutrons))))  # 动态调整颜色范围
            text.set_text(f'时间步: {frame+1}, 裂变数: {fissions}')
            return im1, im2, text
        
        # 创建动画
        ani = FuncAnimation(fig, update, frames=steps, interval=interval, 
                           blit=True, repeat=False)
        
        plt.tight_layout()
        plt.show()
        
        return ani

# 示例：运行链式反应模拟
def run_simulation():
    """运行链式反应模拟并显示结果"""
    # 创建模拟对象
    reactor = NuclearChainReaction(
        size=50, 
        initial_fissile=0.4, 
        initial_neutrons=20,
        fission_prob=0.6,
        absorption_prob=0.1,
        reproduction=2.4,
        diffusion=0.3
    )
    
    # 运行模拟
    reactor.simulate(50)
    
    # 可视化结果
    reactor.visualize()
    
    # 也可以创建动画（取消下面的注释）
    # reactor.animate(50)

if __name__ == "__main__":
    run_simulation()
