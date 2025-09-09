### 基础张量操作

**张量创建与初始化** PyTorch 中的张量是所有操作的基础数据结构。创建张量有多种方式，每种方式都适用于不同的场景。

```python
import torch

# 从列表创建
x = torch.tensor([1, 2, 3])
print(x)
# 输出: tensor([1, 2, 3])

# 创建全零张量
x = torch.zeros(2, 3)
print(x)
# 输出:
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

# 创建全一张量
x = torch.ones(2, 2)
print(x)
# 输出:
# tensor([[1., 1.],
#         [1., 1.]])

# 创建服从正态分布的随机张量
x = torch.randn(2, 2)
print(x)
# 输出:
# tensor([[ 0.4485, -0.1982],
#         [-0.6019,  0.7259]])  # 每次运行结果不同

# 创建服从均匀分布的随机张量
x = torch.rand(2, 2)
print(x)
# 输出:
# tensor([[0.4048, 0.4350],
#         [0.3475, 0.4705]])  # 每次运行结果不同

# 创建等差数列张量
x = torch.arange(0, 10, 2)
print(x)
# 输出: tensor([0, 2, 4, 6, 8])

# 创建线性间隔张量
x = torch.linspace(0, 1, 5)
print(x)
# 输出: tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

# 创建单位矩阵
x = torch.eye(3)
print(x)
# 输出:
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

# 创建未初始化张量
x = torch.empty(2, 3)
print(x)
# 输出:
# tensor([[2.7719e-10, 3.0903e-10, 2.7663e-10],
#         [3.3361e-10, 2.7594e-10, 3.3644e-10]])  # 内容是随机的垃圾值

```

---
### **张量属性与形状操作**

理解和操作张量的形状是深度学习中的核心技能。PyTorch 提供了丰富的形状操作函数来满足各种需求。

```python
import torch

# 假设我们有一个形状为 (2, 3, 4) 的张量
x = torch.randn(2, 3, 4)
# tensor([[[-1.7887,  0.8078, -0.5205,  1.4544],
#          [ 1.1881, -1.3913, -0.1193, -2.3993],
#          [ 0.4388, -0.1799, -1.4559, -0.4960]],

#         [[ 0.2046,  1.1247,  1.4024,  0.2643],
#          [ 1.0282, -1.6724,  1.5416,  0.2444],
#          [-0.8565,  0.6308,  0.6029, -0.0265]]])

# 形状操作
print("原始张量形状:", x.shape)
# 输出: 原始张量形状: torch.Size([2, 3, 4])

# 使用 view 改变形状
y = x.view(6, 4)
print("view 后的张量形状:", y.shape)
# 输出: view 后的张量形状: torch.Size([6, 4])
# 注意: y 和 x 共享内存，改变 y 的值会影响 x
# x: tensor([[[-1.7887,  0.8078, -0.5205,  1.4544],
#          [ 1.1881, -1.3913, -0.1193, -2.3993],
#          [ 0.4388, -0.1799, -1.4559, -0.4960]],

#         [[ 0.2046,  1.1247,  1.4024,  0.2643],
#          [ 1.0282, -1.6724,  1.5416,  0.2444],
#          [-0.8565,  0.6308,  0.6029, -0.0265]]])
# y: tensor([[-1.7887,  0.8078, -0.5205,  1.4544],
#         [ 1.1881, -1.3913, -0.1193, -2.3993],
#         [ 0.4388, -0.1799, -1.4559, -0.4960],
#         [ 0.2046,  1.1247,  1.4024,  0.2643],
#         [ 1.0282, -1.6724,  1.5416,  0.2444],
#         [-0.8565,  0.6308,  0.6029, -0.0265]])

# 使用 reshape 改变形状
z = x.reshape(1, 24)
print("reshape 后的张量形状:", z.shape)
# 输出: reshape 后的张量形状: torch.Size([1, 24])
# z: tensor([[-1.7887,  0.8078, -0.5205,  1.4544,  1.1881, -1.3913, -0.1193, -2.3993,
#           0.4388, -0.1799, -1.4559, -0.4960,  0.2046,  1.1247,  1.4024,  0.2643,
#           1.0282, -1.6724,  1.5416,  0.2444, -0.8565,  0.6308,  0.6029, -0.0265]])

# 移除大小为 1 的维度
# 先创建一个带大小为 1 的维度张量
a = torch.randn(1, 3, 1, 4)
print("\n原始张量形状 (a):", a.shape)
# 输出: 原始张量形状 (a): torch.Size([1, 3, 1, 4])
# a: tensor([[[[ 0.0052, -1.1267,  0.3456,  1.1733]],

#          [[-0.3795, -0.5393, -0.6779, -0.9488]],

#          [[ 0.6640, -0.4624, -0.3095,  0.5408]]]])

b = a.squeeze()
# 它的作用是移除张量中所有大小为 1 的维度
print("squeeze 后的张量形状 (b):", b.shape)
# 输出: squeeze 后的张量形状 (b): torch.Size([3, 4])

# 仅移除第 0 维 
z = torch.squeeze(x, dim=0)
print("只移除第 0 维后的形状:", z.shape) 
# 输出: 只移除第 0 维后的形状: torch.Size([3, 1, 4])

# 添加大小为 1 的维度
c = torch.randn(3, 4)
print("\n原始张量形状 (c):", c.shape)
# 输出: 原始张量形状 (c): torch.Size([3, 4])
d = c.unsqueeze(0)  # 在第 0 维添加一个维度
print("unsqueeze 后的张量形状 (d):", d.shape)
# 输出: unsqueeze 后的张量形状 (d): torch.Size([1, 3, 4])

# 交换维度
e = torch.randn(2, 3, 4)
print("\n原始张量形状 (e):", e.shape)
# 输出: 原始张量形状 (e): torch.Size([2, 3, 4])
f = e.transpose(0, 1)  # 交换第 0 和第 1 维
print("transpose 后的张量形状 (f):", f.shape)
# 输出: transpose 后的张量形状 (f): torch.Size([3, 2, 4])
print('e:', e)
print('f:', f)
# e: tensor([[[ 1.7325, -0.5320, -0.4610, -0.4076],
#          [-1.5115, -0.8013,  2.7673,  1.1098],
#          [ 0.1500,  0.9319,  0.8033, -0.8008]],

#         [[-1.2703,  1.6161, -0.2069,  0.3050],
#          [ 0.5379, -0.6140,  0.2943,  0.9209],
#          [-0.6310,  0.2884, -0.1021, -3.6232]]])
# f: tensor([[[ 1.7325, -0.5320, -0.4610, -0.4076],
#          [-1.2703,  1.6161, -0.2069,  0.3050]],

#         [[-1.5115, -0.8013,  2.7673,  1.1098],
#          [ 0.5379, -0.6140,  0.2943,  0.9209]],

#         [[ 0.1500,  0.9319,  0.8033, -0.8008],
#          [-0.6310,  0.2884, -0.1021, -3.6232]]])
```

---

### **张量运算**

张量运算包括基本的数学运算、线性代数运算以及元素级操作，这些是构建复杂模型的基础。

### 基本运算
```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 加法
print("加法 (a + b):", a + b)
print("torch.add(a, b):", torch.add(a, b))
# tensor([5, 7, 9])

# 减法
print("减法 (a - b):", a - b)
print("torch.sub(a, b):", torch.sub(a, b))
# 减法 (a - b): tensor([-3, -3, -3])
# torch.sub(a, b): tensor([-3, -3, -3])

# 元素级乘法 (element-wise multiplication)
print("元素级乘法 (a * b):", a * b)
print("torch.mul(a, b):", torch.mul(a, b))
# 元素级乘法 (a * b): tensor([ 4, 10, 18])
# torch.mul(a, b): tensor([ 4, 10, 18])

# 除法
print("除法 (a / b):", a / b)
print("torch.div(a, b):", torch.div(a, b))
# 除法 (a / b): tensor([0.2500, 0.4000, 0.5000])
# torch.div(a, b): tensor([0.2500, 0.4000, 0.5000])

# 幂运算
print("幂运算 (a ** 2):", a ** 2)
print("torch.pow(a, 2):", torch.pow(a, 2))
# 幂运算 (a ** 2): tensor([1, 4, 9])
# torch.pow(a, 2): tensor([1, 4, 9])
```

### 线性代数
```python
# 创建两个矩阵
a = torch.randn(2, 3)
b = torch.randn(3, 4)

# 矩阵乘法 (Matrix Multiplication)
c = torch.mm(a, b)
print("矩阵乘法 (torch.mm):", c.shape)
# 矩阵乘法 (torch.mm): torch.Size([2, 4])

# 批量矩阵乘法 (Batch Matrix Multiplication)
# 创建批量张量 (batch, rows, cols)
a_batch = torch.randn(10, 2, 3)
b_batch = torch.randn(10, 3, 4)
c_batch = torch.bmm(a_batch, b_batch)
print("批量矩阵乘法 (torch.bmm):", c_batch.shape)
# 只适用于3维
# 批量矩阵乘法 (torch.bmm): torch.Size([10, 2, 4])

# 通用矩阵乘法 (Generic Matrix Multiplication)
# 当维数大于等于2时，功能与mm相同
c_matmul = torch.matmul(a, b)
print("通用矩阵乘法 (torch.matmul):", c_matmul.shape)
# 支持更高维度的乘法，例如 (10, 2, 3) x (10, 3, 4)
c_matmul_batch = torch.matmul(a_batch, b_batch)
print("通用矩阵乘法 (批量):", c_matmul_batch.shape)
# 通用矩阵乘法 (torch.matmul): torch.Size([2, 4])
# 通用矩阵乘法 (批量): torch.Size([10, 2, 4])

# 向量点积 (Dot Product)
a_vec = torch.tensor([1, 2, 3])
b_vec = torch.tensor([4, 5, 6])
dot_product = torch.dot(a_vec, b_vec)
print("向量点积 (torch.dot):", dot_product)
# 向量点积 (torch.dot): tensor(32)

# 向量叉积 (Cross Product)
a_cross = torch.tensor([1, 2, 3.])
b_cross = torch.tensor([4, 5, 6.])
cross_product = torch.cross(a_cross, b_cross)
print("向量叉积 (torch.cross):", cross_product)
# c = (a2b3 - a3b2, a3b1 - a1b3, a1b2 - a2b1)
```

### **统计运算**
```python
# 创建一个张量
a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print("原始张量:", a)

# 求和 (Sum)
total_sum = torch.sum(a)
print("总和 (torch.sum):", total_sum)
# 总和 (torch.sum): tensor(21.)

# 可以在特定维度上求和
sum_dim0 = torch.sum(a, dim=0)
print("沿第0维求和:", sum_dim0)
# 沿第0维求和: tensor([5., 7., 9.])

# 均值 (Mean)
mean_val = torch.mean(a)
print("平均值 (torch.mean):", mean_val)
# 平均值 (torch.mean): tensor(3.5000)

# 标准差 (Standard Deviation)
# 标准差是方差的**平方根**。
std_val = torch.std(a)
print("标准差 (torch.std):", std_val)
# 标准差 (torch.std): tensor(1.8708)

# 方差 (Variance)
# 方差的定义是：每个数据点与平均值之差的**平方**的平均值。
var_val = torch.var(a)
print("方差 (torch.var):", var_val)

# 最大值和最小值 (Max and Min)
max_val = torch.max(a)
min_val = torch.min(a)
print("最大值 (torch.max):", max_val)
print("最小值 (torch.min):", min_val)

# 最大值和最小值索引 (Argmax and Argmin)
# 默认返回张量中所有元素的最大/小值及其索引
max_index = torch.argmax(a)
min_index = torch.argmin(a)
print("最大值索引 (torch.argmax):", max_index)
print("最小值索引 (torch.argmin):", min_index)
```

---

### **神经网络构建**

**基础网络层**

PyTorch 的 `nn` 模块提供了构建神经网络所需的所有基础层，
```python
import torch
import torch.nn as nn

# 创建一个线性层，输入特征10个，输出特征5个
linear_layer = nn.Linear(10, 5)
# Linear(in_features=10, out_features=5, bias=True)

# 创建一个模拟输入数据
# 通常线性层的输入是二维的 (batch_size, input_features)
input_data = torch.randn(32, 10) # 批量大小为32

# 将输入数据通过线性层
output_data = linear_layer(input_data)

print("线性层输入形状:", input_data.shape)
print("线性层输出形状:", output_data.shape)
# 线性层输入形状: torch.Size([32, 10]) 
# 线性层输出形状: torch.Size([32, 5])
```

---
#### 双线性层有什么用？
**`nn.Bilinear(10, 20, 5)`**: 双线性层，用于学习两个输入张量之间的交互，并输出一个结果。例如，在推荐系统中，可以用来建模用户和物品特征之间的关系。
```python
# 定义一个双线性层
# 假设输入1有10个特征，输入2有20个特征，我们想要得到5个输出特征
bilinear_layer = nn.Bilinear(in1_features=10, in2_features=20, out_features=5)

# 模拟两个输入张量
# 批量大小为 32
input1 = torch.randn(32, 10)  # 形状: (batch_size, in1_features)
input2 = torch.randn(32, 20)  # 形状: (batch_size, in2_features)

# 通过双线性层进行计算
output = bilinear_layer(input1, input2)

print("第一个输入形状:", input1.shape)
print("第二个输入形状:", input2.shape)
print("双线性层输出形状:", output.shape)
# 第一个输入形状: torch.Size([32, 10])
# 第二个输入形状: torch.Size([32, 20])
# 双线性层输出形状: torch.Size([32, 5])
```


双线性层在需要**建模两个不同类型实体之间关系**的场景中非常有用，尤其是在推荐系统和自然语言处理中。

**1. 推荐系统：**

假设你有一个用户向量和一个物品向量。你可以用双线性层来学习用户和物品特征之间的复杂交互，从而预测用户对物品的评分。

- x_1 = 用户特征向量
- x_2 = 物品特征向量
- y = 预测评分
    

**2. 自然语言处理（NLP）：**

在处理句子中的词语关系时，双线性层可以用来捕捉两个词向量之间的语义关系，例如在关系抽取（Relation Extraction）任务中，用来预测两个实体之间的关系类型。

- x_1 = 第一个实体的词向量
- x_2 = 第二个实体的词向量
- y = 预测的关系类型

---

### 卷积层（Convolutional Layer）

卷积的核心思想是，用一个小的、可学习的**滤波器**（或者叫卷积核）在输入数据上进行**滑动**和**计算**，从而提取出数据中的**局部特征**。

你可以把它想象成一个“特征探测器”：

- **输入数据**：比如一张图片。
    
- **滤波器**：一个很小且有特定数值的矩阵，就像一个“探测器”。例如，一个滤波器可以设计成专门检测图片中的垂直边缘，另一个可以检测水平边缘。
    
- **滑动**：滤波器在图片上从左到右、从上到下滑动，就像一个放大镜在移动。
    
- **计算**：在每一个位置，滤波器都会和它覆盖的区域进行**元素级乘法**，然后把所有乘积加起来，得到一个新值。这个新值代表了该区域的特征。
    

每一次计算都会产生一个新的像素值，所有这些新值组合起来就形成了**输出特征图**（Feature Map）。这个特征图是原始数据经过特征提取后的结果。

##### 转置卷积（Transposed Convolution）

转置卷积，也常被称为**反卷积（Deconvolution）或分数步长卷积（Fractionally-strided Convolution）**，它的核心作用与普通卷积相反：**将低维的特征图（Feature Map）上采样（upsample）到高维**。

你可以将转置卷积想象成一个“反向的”卷积过程。普通卷积通过滑动滤波器来缩小特征图的尺寸（下采样），而转置卷积通过一种特殊的方式，将输入特征图的每个像素“扩展”成一个更大的区域，从而放大特征图的尺寸。

这个过程非常重要，尤其是在以下场景中：

1. **图像生成**：在生成对抗网络（GAN）中，转置卷积用于将一个低维的随机噪声向量逐步上采样，最终生成一张完整的图像。
    
2. **语义分割**：在语义分割任务中，模型需要将每个像素分类。通常，编码器（Encoder）部分会通过卷积下采样来提取特征，而解码器（Decoder）部分则需要用转置卷积将这些特征图上采样回原始图像的尺寸，以便对每个像素进行分类。

```python
# 1D卷积层
# 输入通道16，输出通道32，卷积核大小3
conv1d_layer = nn.Conv1d(16, 32, 3)
input_1d = torch.randn(1, 16, 100) # (batch, channels, length)
output_1d = conv1d_layer(input_1d)
print("1D卷积层输入形状:", input_1d.shape)
print("1D卷积层输出形状:", output_1d.shape)
# 1D卷积层输入形状: torch.Size([1, 16, 100])
# 1D卷积层输出形状: torch.Size([1, 32, 98])
# 输出长度 = （输入长度-卷及大小）/ 步长 + 1
# 在 nn.Conv1d 中，你可以通过 stride 参数来定义步长。

# 2D卷积层
# 输入通道3，输出通道64，卷积核大小3
conv2d_layer = nn.Conv2d(3, 64, 3)
input_2d = torch.randn(1, 3, 256, 256) # (batch, channels, height, width)
output_2d = conv2d_layer(input_2d)
print("2D卷积层输入形状:", input_2d.shape)
print("2D卷积层输出形状:", output_2d.shape)
# 2D卷积层输入形状: torch.Size([1, 3, 256, 256])
# 2D卷积层输出形状: torch.Size([1, 64, 254, 254])
# 254 = (256 - 3) / 1 + 1 = 254

# 3D卷积层
# 输入通道1，输出通道8，卷积核大小3
conv3d_layer = nn.Conv3d(1, 8, 3)
input_3d = torch.randn(1, 1, 16, 64, 64) # (batch, channels, depth, height, width)
output_3d = conv3d_layer(input_3d)
print("3D卷积层输入形状:", input_3d.shape)
print("3D卷积层输出形状:", output_3d.shape)
# 3D卷积层输入形状: torch.Size([1, 1, 16, 64, 64])
# 3D卷积层输出形状: torch.Size([1, 8, 14, 62, 62])

# 转置卷积层
# 用来上采样，通常在生成对抗网络或语义分割中使用
conv_transpose_layer = nn.ConvTranspose2d(64, 32, 4, 2, 1)
input_transpose = torch.randn(1, 64, 32, 32)
output_transpose = conv_transpose_layer(input_transpose)
print("转置卷积层输入形状:", input_transpose.shape)
print("转置卷积层输出形状:", output_transpose.shape)
# 转置卷积层输入形状: torch.Size([1, 64, 32, 32])
# 转置卷积层输出形状: torch.Size([1, 32, 64, 64])
```

---

### 池化层（Pooling Layer）
```python
# 最大池化 (Max Pooling)
# 卷积核大小2x2
maxpool_layer = nn.MaxPool2d(2)
input_pool = torch.randn(1, 64, 128, 128)
output_maxpool = maxpool_layer(input_pool)
print("最大池化输入形状:", input_pool.shape)
print("最大池化输出形状:", output_maxpool.shape)

# 平均池化 (Average Pooling)
# 卷积核大小2x2
avgpool_layer = nn.AvgPool2d(2)
output_avgpool = avgpool_layer(input_pool)
print("平均池化输出形状:", output_avgpool.shape)

# 自适应最大池化 (Adaptive Max Pooling)
# 将输入自动池化到 1x1 的大小
adaptive_maxpool = nn.AdaptiveMaxPool2d((1, 1))
output_adaptive_max = adaptive_maxpool(input_pool)
print("自适应最大池化输出形状:", output_adaptive_max.shape)

# 自适应平均池化 (Adaptive Average Pooling)
# 将输入自动池化到 7x7 的大小
adaptive_avgpool = nn.AdaptiveAvgPool2d((7, 7))
output_adaptive_avg = adaptive_avgpool(input_pool)
print("自适应平均池化输出形状:", output_adaptive_avg.shape)
```


---

### 归一化层（Normalization Layer）
```python
# 1D批归一化 (BatchNorm1d)
# 用于处理一维输入，通常是全连接层或1D卷积的输出
# 参数是特征维度，这里是100
bn1d_layer = nn.BatchNorm1d(100)
input_bn1d = torch.randn(32, 100) # (batch, features)
output_bn1d = bn1d_layer(input_bn1d)
print("1D批归一化输入形状:", input_bn1d.shape)
print("1D批归一化输出形状:", output_bn1d.shape)

# 2D批归一化 (BatchNorm2d)
# 用于处理图像数据，通常在2D卷积层之后
# 参数是通道数，这里是64
bn2d_layer = nn.BatchNorm2d(64)
input_bn2d = torch.randn(1, 64, 32, 32) # (batch, channels, height, width)
output_bn2d = bn2d_layer(input_bn2d)
print("2D批归一化输入形状:", input_bn2d.shape)
print("2D批归一化输出形状:", output_bn2d.shape)

# 层归一化 (LayerNorm)
# 参数是需要归一化的维度，这里是最后一个维度(10)
layernorm_layer = nn.LayerNorm([10, 10])
input_layernorm = torch.randn(20, 10, 10)
output_layernorm = layernorm_layer(input_layernorm)
print("层归一化输入形状:", input_layernorm.shape)
print("层归一化输出形状:", output_layernorm.shape)

# 组归一化 (GroupNorm)
# 将通道分成组进行归一化
# 参数是分组数(10)和通道数(20)
groupnorm_layer = nn.GroupNorm(10, 20)
input_groupnorm = torch.randn(1, 20, 16, 16)
output_groupnorm = groupnorm_layer(input_groupnorm)
print("组归一化输入形状:", input_groupnorm.shape)
print("组归一化输出形状:", output_groupnorm.shape)
```