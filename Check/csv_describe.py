import pandas as pd

# 读取CSV文件
# df = pd.read_csv("D:\PyCharm 2022.1\PyCharm Documents\Siamese Neural Networks for ECG-signal\Data\sph_cinc_test.csv")
# df = pd.read_csv("D:\PyCharm 2022.1\PyCharm Documents\Siamese Neural Networks for ECG-signal\Data\sph_cinc_train.csv")
df = pd.read_csv("D:\PyCharm 2022.1\PyCharm Documents\Siamese Neural Networks for ECG-signal\Data\sph_cinc_val.csv")

# 读取名为 "age" 的列的详细统计信息
column_stats = df["age"].describe()

print(column_stats)
