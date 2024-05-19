
import pandas as pd

# 路径设置
csv_file_path = '/Data/sph_cinc_test.csv'

# 尝试读取CSV文件的前几行
try:
    sample_df = pd.read_csv(csv_file_path, nrows=5)
    print(sample_df.columns)
except Exception as e:
    print("Error reading CSV file:", e)

