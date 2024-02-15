import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


# 定义三维高斯分布函数
def gaussian_3d_two(x, y, params1, params2):
    x01, y01, sigma_x1, sigma_y1, rho1 = params1
    x02, y02, sigma_x2, sigma_y2, rho2 = params2

    Z1 = np.exp(-((x - x01) ** 2 / (2 * sigma_x1 ** 2) +
                  (y - y01) ** 2 / (2 * sigma_y1 ** 2) -
                  (2 * rho1 * (x - x01) * (y - y01)) / (2 * sigma_x1 * sigma_y1))) / (
                     2 * np.pi * sigma_x1 * sigma_y1 * np.sqrt(1 - rho1 ** 2))

    Z2 = np.exp(-((x - x02) ** 2 / (2 * sigma_x2 ** 2) +
                  (y - y02) ** 2 / (2 * sigma_y2 ** 2) -
                  (2 * rho2 * (x - x02) * (y - y02)) / (2 * sigma_x2 * sigma_y2))) / (
                     2 * np.pi * sigma_x2 * sigma_y2 * np.sqrt(1 - rho2 ** 2))

    return Z1, Z2


# 设置两组参数，以减少重叠区域
params1 = [0, 0, 1, 1, 0]  # 中心在原点，较小的范围
params2 = [2, 2, 1.5, 1.5, 0]  # 中心更偏移，稍微大一些的范围

# 创建网格
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# 计算两个分布
Z1, Z2 = gaussian_3d_two(X, Y, params1, params2)

# 计算重叠区域
overlap = np.minimum(Z1, Z2)
overlap[overlap < 0.01] = np.nan  # 小于阈值的部分视为非重叠区域

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制两个高斯分布，使用不同的颜色
surf1 = ax.plot_surface(X, Y, Z1, cmap='Blues', alpha=0.5)
surf2 = ax.plot_surface(X, Y, Z2, cmap='Oranges', alpha=0.5)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# 绘制重叠区域
ax.plot_surface(X, Y, overlap, color='red', alpha=0.7)

# 标签和标题
# ax.set_title('Generated Data Distribution with Overlap')
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# 使用proxy artist为3D图形添加图例
# legends = [Line2D([0], [0], linestyle="none", marker='o', color='blue', label='Tail Class Distribution'),
#            Line2D([0], [0], linestyle="none", marker='o', color='orange', label='Head Class Distribution'),
#            Line2D([0], [0], linestyle="none", marker='o', color='red', label='Overlap')]
# ax.legend(handles=legends)

# 保存图像
# plt.savefig('./gaussian_distributions_overlap.pdf', format='pdf')
plt.savefig('./gaussian_distributions_overlap.png', format='png')
plt.show()

# plt.close(fig)



