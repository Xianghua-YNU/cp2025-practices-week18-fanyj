import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class HydrogenAtomSimulation:
    """氢原子电子云模拟类"""
    
    def __init__(self, a=5.29e-2, D_max=1.1, r0=0.25):
        """
        初始化氢原子模拟参数
        
        参数:
        a: 玻尔半径 (nm)，默认值为 5.29e-2 nm
        D_max: 概率密度最大值，用于可视化
        r0: 初始半径 (nm)，用于生成电子位置
        """
        self.a = a          # 玻尔半径
        self.D_max = D_max  # 概率密度最大值
        self.r0 = r0        # 初始半径
        
    def probability_density(self, r):
        """
        计算氢原子基态(n=1, l=0, m=0)的电子概率密度
        
        参数:
        r: 距离原子核的距离 (nm)
        
        返回:
        概率密度值
        """
        return (4 * r**2 / self.a**3) * np.exp(-2 * r / self.a)
    
    def generate_electron_positions(self, num_electrons=10000):
        """
        生成电子的随机位置，位置分布符合氢原子基态的概率密度
        
        参数:
        num_electrons: 要生成的电子数量
        
        返回:
        x, y, z: 电子的三维坐标数组
        """
        # 生成符合概率密度分布的随机半径
        r = np.zeros(num_electrons)
        for i in range(num_electrons):
            while True:
                # 在[0, 5*a]范围内随机选择r
                r_candidate = np.random.uniform(0, 5 * self.a)
                # 计算该r处的概率密度
                p = self.probability_density(r_candidate)
                # 接受/拒绝采样
                if np.random.uniform(0, self.D_max) < p:
                    r[i] = r_candidate
                    break
        
        # 生成随机角度
        theta = np.random.uniform(0, np.pi, num_electrons)  # 极角
        phi = np.random.uniform(0, 2*np.pi, num_electrons)  # 方位角
        
        # 转换为笛卡尔坐标
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        return x, y, z
    
    def plot_3d_scatter(self, x, y, z, title="氢原子电子云分布（三维散点图）", filename=None):
        """
        绘制电子云的三维散点图
        
        参数:
        x, y, z: 电子的三维坐标数组
        title: 图表标题
        filename: 保存图像的文件名，如果为None则不保存
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 计算每个点到原点的距离
        r = np.sqrt(x**2 + y**2 + z**2)
        # 根据距离设置颜色映射
        norm = colors.Normalize(vmin=0, vmax=np.max(r))
        cmap = cm.get_cmap('viridis')
        
        # 绘制散点图
        scatter = ax.scatter(x, y, z, c=r, cmap=cmap, s=1, alpha=0.6)
        
        # 添加颜色条和标签
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('距离原子核的距离 (nm)')
        
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')
        ax.set_title(title)
        
        # 设置坐标轴比例相等
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_2d_density(self, x, y, z, title="氢原子电子云密度投影图", filename=None):
        """
        绘制电子云的二维密度投影图
        
        参数:
        x, y, z: 电子的三维坐标数组
        title: 图表标题
        filename: 保存图像的文件名，如果为None则不保存
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # XY平面投影
        hist_xy, xedges, yedges = np.histogram2d(x, y, bins=50, density=True)
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        im_xy = axes[0].pcolormesh(X, Y, hist_xy.T, shading='auto', cmap='viridis')
        axes[0].set_xlabel('X (nm)')
        axes[0].set_ylabel('Y (nm)')
        axes[0].set_title('XY平面投影')
        plt.colorbar(im_xy, ax=axes[0])
        
        # XZ平面投影
        hist_xz, xedges, zedges = np.histogram2d(x, z, bins=50, density=True)
        X, Z = np.meshgrid(xedges[:-1], zedges[:-1])
        im_xz = axes[1].pcolormesh(X, Z, hist_xz.T, shading='auto', cmap='viridis')
        axes[1].set_xlabel('X (nm)')
        axes[1].set_ylabel('Z (nm)')
        axes[1].set_title('XZ平面投影')
        plt.colorbar(im_xz, ax=axes[1])
        
        # YZ平面投影
        hist_yz, yedges, zedges = np.histogram2d(y, z, bins=50, density=True)
        Y, Z = np.meshgrid(yedges[:-1], zedges[:-1])
        im_yz = axes[2].pcolormesh(Y, Z, hist_yz.T, shading='auto', cmap='viridis')
        axes[2].set_xlabel('Y (nm)')
        axes[2].set_ylabel('Z (nm)')
        axes[2].set_title('YZ平面投影')
        plt.colorbar(im_yz, ax=axes[2])
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_radial_distribution(self, x, y, z, title="氢原子电子云径向分布图", filename=None):
        """
        绘制电子云的径向分布函数图
        
        参数:
        x, y, z: 电子的三维坐标数组
        title: 图表标题
        filename: 保存图像的文件名，如果为None则不保存
        """
        # 计算每个点到原点的距离
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # 理论径向分布函数
        r_theory = np.linspace(0, np.max(r) * 1.1, 1000)
        D_theory = self.probability_density(r_theory)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制模拟结果的直方图
        hist, bins, _ = ax.hist(r, bins=50, density=True, alpha=0.6, label='模拟结果')
        
        # 绘制理论曲线
        ax.plot(r_theory, D_theory, 'r-', label='理论分布')
        
        ax.set_xlabel('距离原子核的距离 (nm)')
        ax.set_ylabel('概率密度')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def run_simulation(self, num_electrons=10000, plot_types=['3d_scatter', '2d_density', 'radial'], 
                      output_dir=None):
        """
        运行完整的模拟并生成所有可视化图表
        
        参数:
        num_electrons: 要生成的电子数量
        plot_types: 要生成的图表类型列表，可选值为 '3d_scatter', '2d_density', 'radial'
        output_dir: 保存图像的目录，如果为None则不保存图像
        """
        print(f"正在模拟氢原子电子云分布，参数设置：a={self.a} nm, D_max={self.D_max}, r0={self.r0} nm")
        print(f"生成 {num_electrons} 个电子位置...")
        
        # 生成电子位置
        x, y, z = self.generate_electron_positions(num_electrons)
        
        # 根据选择的图表类型生成图表
        if '3d_scatter' in plot_types:
            filename = f"{output_dir}/hydrogen_cloud_3d.png" if output_dir else None
            self.plot_3d_scatter(x, y, z, filename=filename)
        
        if '2d_density' in plot_types:
            filename = f"{output_dir}/hydrogen_cloud_2d.png" if output_dir else None
            self.plot_2d_density(x, y, z, filename=filename)
        
        if 'radial' in plot_types:
            filename = f"{output_dir}/hydrogen_radial_distribution.png" if output_dir else None
            self.plot_radial_distribution(x, y, z, filename=filename)
        
        print("模拟完成!")

def main():
    """主函数，用于演示氢原子电子云模拟"""
    # 创建模拟对象
    simulation = HydrogenAtomSimulation()
    
    # 运行模拟并生成所有图表
    simulation.run_simulation(num_electrons=50000)
    
    # 演示不同参数对电子云分布的影响
    print("\n演示不同玻尔半径(a)对电子云分布的影响:")
    a_values = [4.29e-2, 5.29e-2, 6.29e-2]  # 不同的玻尔半径值
    
    for a in a_values:
        print(f"\n使用玻尔半径 a = {a} nm 进行模拟...")
        sim = HydrogenAtomSimulation(a=a)
        sim.run_simulation(num_electrons=20000, plot_types=['3d_scatter'])

if __name__ == "__main__":
    main()    
