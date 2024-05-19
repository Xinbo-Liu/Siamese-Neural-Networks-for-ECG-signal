"""
本文件仅致于根据已划分.csv文件，将相应集合（训练集、测试集、验证集）的.h5文件从总集合文件夹 “SPH” 中提取到其对应集合子文件夹（SPH_train、SPH_test、SPH_val），与后续数据处理无关。
"""

import pandas as pd
import shutil
import os

# 路径设置
source_folder = 'D:/PyCharm 2022.1/PyCharm Documents/Siamese Neural Networks for ECG-signal/Data/SPH'
# csv_file_path = 'D:/PyCharm 2022.1/PyCharm Documents/Siamese Neural Networks for ECG-signal/Data/sph_cinc_train.csv'
# destination_folder = 'D:/PyCharm 2022.1/PyCharm Documents/Siamese Neural Networks for ECG-signal/Data/SPH_train'
# csv_file_path = 'D:/PyCharm 2022.1/PyCharm Documents/Siamese Neural Networks for ECG-signal/Data/sph_cinc_test.csv'
# destination_folder = 'D:/PyCharm 2022.1/PyCharm Documents/Siamese Neural Networks for ECG-signal/Data/SPH_test'

csv_file_path = '/Data/sph_cinc_val.csv'
destination_folder = 'D:/PyCharm 2022.1/PyCharm Documents/Siamese Neural Networks for ECG-signal/Data/SPH_val'

# 创建目标文件夹（如果不存在）
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 读取CSV文件并获取ID列的数据
df = pd.read_csv(csv_file_path, usecols=['ID'])

# 遍历并复制文件
for filename in df['ID']:
    source_file = os.path.join(source_folder, str(filename) + '.h5')  # 将filename转换为字符串
    destination_file = os.path.join(destination_folder, str(filename) + '.h5')

    if os.path.exists(source_file):
        shutil.copy2(source_file, destination_file)

# 代码执行完毕，文件应已复制到目标文件夹
print("文件复制完成。")
