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


# 设置两组参数，确保没有重叠
params1 = [0, 0, 1, 1, 0]  # 中心在原点，较小的范围
params2 = [4, 4, 1.5, 1.5, 0]  # 中心更偏移，范围相同

# 创建网格
x = np.linspace(-5, 10, 100)
y = np.linspace(-5, 10, 100)
X, Y = np.meshgrid(x, y)

# 计算两个分布
Z1, Z2 = gaussian_3d_two(X, Y, params1, params2)

# 画图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制两个高斯分布，使用不同的颜色
surf1 = ax.plot_surface(X, Y, Z1, cmap='Blues', alpha=0.5)
surf2 = ax.plot_surface(X, Y, Z2, cmap='Oranges', alpha=0.5)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# 标签和标题
# ax.set_title('Generated Data Distribution without Overlap')
# ax.set_xlabel()
# ax.set_ylabel()
# ax.set_zlabel()

# 使用proxy artist为3D图形添加图例
# legends = [Line2D([0], [0], linestyle="none", marker='o', color='blue', label='Tail Class Distribution'),
#            Line2D([0], [0], linestyle="none", marker='o', color='orange', label='Head Class Distribution')]
# ax.legend(handles=legends)

# 保存图像

plt.savefig('./gaussian_distributions_no_overlap.png', format='png')
plt.show()

# plt.savefig('/mnt/data/gaussian_distributions_separated.png', format='png')
# plt.close(fig)  # 关闭图像，避免重复显示
#
# # 返回文件路径以供下载
# '/mnt/data/gaussian_distributions_separated.png'
