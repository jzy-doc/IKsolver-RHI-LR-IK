import numpy as np

# 读取txt文件，假设每行数据用逗号分隔
file_path = f'/home/desc/jzy2/rhi-lr-ik/dataset/piper/all_area/generated_data_train.txt'  # 替换为你的文件路径

# 读取数据
data = np.loadtxt(file_path, delimiter=',')

# 计算每一列的最大值和最小值
column_max = np.max(data, axis=0)
column_min = np.min(data, axis=0)

# 打印每一列的最大值和最小值
for i in range(data.shape[1]):
    print(f"列 {i+1} 的最大值: {column_max[i]}, 最小值: {column_min[i]}")