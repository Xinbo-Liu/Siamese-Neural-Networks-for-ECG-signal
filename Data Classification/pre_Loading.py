import os
import h5py
import pandas as pd
import numpy as np

"""
本文件仅致于根据已划分.csv文件，将相应集合（训练集、测试集、验证集）的.h5文件从总集合文件夹 “SPH” 中提取到其对应集合子文件夹（SPH_train、SPH_test、SPH_val），与后续数据处理无关。
"""

# 设置数据文件夹路径
BASE_PATH = "/Data"
TRAIN_PATH = os.path.join(BASE_PATH, "SPH_train")


# 函数用于加载.h5格式的心电信号数据并将每个信号处理为5000长度
def load_and_preprocess_ecg_signals(h5_directory, signal_length=5000):
    ecg_signals = []
    ecg_ids = []

    # 遍历.h5文件并加载数据
    for filename in os.listdir(h5_directory):
        file_path = os.path.join(h5_directory, filename)
        with h5py.File(file_path, 'r') as f:
            # 心电信号数据存储在h5文件的'signal'键下
            signal = f['signal'][:]
            # 裁剪或填充信号
            if signal.shape[1] > signal_length:
                # 裁剪多余的部分
                signal = signal[:, :signal_length]
            elif signal.shape[1] < signal_length:
                # 如果信号太短，则填充零
                signal = np.pad(signal, ((0, 0), (0, signal_length - signal.shape[1])), 'constant')

            # Z-score标准化
            # mean = np.mean(signal, axis=1, keepdims=True)
            # std = np.std(signal, axis=1, keepdims=True)
            # signal = (signal - mean) / std

            # 存储处理后的信号和ID
            ecg_signals.append(signal)
            ecg_ids.append(filename.split('.')[0])  # 文件名即ID

    # 将列表转换为NumPy数组
    return np.array(ecg_signals), ecg_ids


# 加载并预处理训练数据
train_signals, train_ids = load_and_preprocess_ecg_signals(TRAIN_PATH)

# print(train_signals, train_ids)
