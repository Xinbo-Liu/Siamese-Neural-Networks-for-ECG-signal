import os
import numpy as np
import pandas as pd
import h5py
import random

import tensorflow as tf

from keras.models import load_model
from keras.layers import Layer
from keras import backend as Kb

from Config.hyper_parameter import initial_margin

'''函数用于加载.h5格式的心电信号数据并将每个信号处理为 5000 长度'''


def load_and_preprocess_ecg_signals(h5_directory, csv_file, signal_length=5000):
    ecg_signals = []
    ecg_file_labels = []  # 文件名作为文件标签
    ecg_category_labels = []  # 从CSV文件读取的类别标签

    # 从CSV文件读取心电信号的DX编码
    df = pd.read_csv(csv_file)
    # 配对CSV文件中包含的心电信号文件名（转换为整型）和对应的DX编码
    filename_to_dx = dict(zip(df['ID'].astype(str), df['Dx']))  # 确保类型一致

    # 遍历.h5文件并加载数据
    filenames = sorted(os.listdir(h5_directory))[:]
    for filename in filenames:
        file_label = filename.replace('.h5', '')
        file_path = os.path.join(h5_directory, filename)

        # 心电信号数据存储在h5文件的 'signal' 键下
        with h5py.File(file_path, 'r') as f:
            signal = f['signal'][:]

            # 裁剪或填充信号，如果信号太短，则填充零
            if signal.shape[1] > signal_length:
                signal = signal[:, :signal_length]
            elif signal.shape[1] < signal_length:
                signal = np.pad(signal, ((0, 0), (0, signal_length - signal.shape[1])), 'constant')

            # Z-score标准化
            # mean = np.mean(signal, axis=1, keepdims=True)
            # std = np.std(signal, axis=1, keepdims=True)
            # signal = (signal - mean) / std

            ecg_signals.append(signal)
            ecg_file_labels.append(file_label)

            # 获取类别标签
            # 使用dict.get方法来安全地获取字典中的类别标签，如果找不到匹配项，则默认返回"未知"
            category_label = filename_to_dx.get(file_label, "未知")
            ecg_category_labels.append(category_label)

            # print("处理文件: ", filename)
            # print("信号形状: ", signal.shape)
            # print("文件标签: ", file_label)
            # print("类别标签: ", category_label)

    # 函数返回心电信号、心电信号文件标签、心电信号类别标签
    return np.array(ecg_signals), np.array(ecg_file_labels), np.array(ecg_category_labels)


'''generate_n_way_k_shot函数是用于小样本学习（Few-Shot Learning）场景中，在实现N-way K-shot学习方法时。其目的是从给定的心电图信号数据中
生成一个训练和测试用的支持集（support set）和查询集（query set）。 

下面是函数的具体作用：

1. 选择类别：
categories变量通过从label_to_index字典的值中随机选择N个不重复的类别来初始化。这里的N表示你想在一次训练中考虑的不同类别的数量。

2. 初始化支持集和查询集：
support_set和query_set是用来存储训练和测试数据的集合。

3. 为每个类别生成样本：
对于选定的每个类别，函数首先找出所有属于该类别的心电图信号的索引。
检查每个类别是否有足够的样本（至少K+1个），因为在K-shot学习中，你需要从每个类别中选择K个样本进行训练，以及至少一个样本用于测试或查询。

4. 选择样本并添加到支持集和查询集：
参数Q控制每个类别在查询集中的样本数量，默认为1。
对于每个类别，随机选择K+Q个样本。其中K个样本被添加到支持集中，剩余Q个样本被添加到查询集中。
support_set包含了用于训练模型的样本，而query_set包含了用于测试或评估模型的样本。

5. 返回生成的支持集和查询集：
函数最终返回这两个集合，以供后续的模型训练和评估过程使用。
'''


def generate_n_way_k_shot(ecg_signals, ecg_labels, N, K, label_to_index, index_to_label, Q=1):
    """
    生成N-way K-shot任务的支持集和查询集。

    参数:
    - ecg_signals (np.array): 心电图信号数组。
    - ecg_labels (np.array): 对应的类别标签数组。
    - N (int): 要随机选择的类别数量（N-way）。
    - K (int): 每个类别的样本数量（K-shot）。
    - label_to_index (dict): 从类别名称到索引的映射。
    - index_to_label (dict): 从索引到类别名称的映射。
    - Q (int): 查询集中每个类别的样本数，默认为1。

    返回:
    - tuple: 包含支持集和查询集样本对，以及类别详细信息的三元组。
    """
    # 从可能的类别标签中随机选择N个
    categories = np.random.choice(list(label_to_index.values()), size=N, replace=False)
    support_set = []  # 初始化支持集
    query_set = []  # 初始化查询集
    category_details = []  # 用于存储每个选定类别的详细信息

    # 遍历每一个随机选中的类别
    for category in categories:
        # 获取属于当前类别的所有样本的索引
        indices = np.where(ecg_labels == index_to_label[category])[0]

        # 确保每个类别都有足够的样本供选择
        if len(indices) >= (K + Q):
            # 从这些样本中随机选择K + Q个样本
            chosen_indices = np.random.choice(indices, size=(K + Q), replace=False)
            # 将前K个样本加入支持集
            support_set.extend([(ecg_signals[i], category) for i in chosen_indices[:K]])
            # 将剩余的样本加入查询集
            query_set.extend([(ecg_signals[i], category) for i in chosen_indices[K:]])

            # 收集并存储当前类别的详细信息
            category_details.append({
                "类别_DX": index_to_label[category],
                "总样本数": len(indices),
                "选中的支持集样本": [ecg_signals[i] for i in chosen_indices[:K]],
                "选中的查询集样本": [ecg_signals[i] for i in chosen_indices[K:]]
            })

    return support_set, query_set, category_details


'''
create_pairs_from_support_query_sets函数的目的是为孪生网络训练创建样本对。在孪生网络中，训练过程需要成对的样本来比较它们之间的相似性或差异性。
该函数的作用是根据提供的支持集（support_set）和查询集（query_set）生成样本对。
详细解释如下：

1. 初始化pairs和labels列表：
pairs用于存储成对的样本。
labels用于存储每对样本是否属于同一类别的标签（是或否）。

2. 遍历查询集中的每个样本：
对于查询集（query_set）中的每个样本（称为query），提取出它的数据（query_data）和标签（query_label）。

3. 为每个查询样本创建正负样本对：
遍历支持集（support_set）中的每个样本（称为support），提取出它的数据（support_data）和标签（support_label）。
将查询样本的数据与支持集中每个样本的数据配对，并将这些配对添加到pairs列表中。
如果查询样本的标签与支持样本的标签相同（即它们属于同一类别），则在labels列表中添加1（表示匹配）。如果它们的标签不同，则添加0（表示不匹配）。

4. 返回样本对和标签列表：
最终，函数返回包含所有生成的样本对的pairs数组和相应的labels数组。
'''


def create_balanced_pairs_from_support_query_sets(support_set, query_set):
    """
    生成类别均衡的正负样本对。

    Args:
    - support_set: 支持集样本，格式为[(data, label), ...]
    - query_set: 查询集样本，格式为[(data, label), ...]

    Returns:
    - dict: 包含正负样本对数组的字典。
    """

    def generate_pairs(samples):
        from collections import defaultdict

        class_data = defaultdict(list)
        positive_pairs = []
        negative_pairs = []

        # 将样本按类别分组
        for data, label in samples:
            class_data[label].append(data)

        # 遍历每个类别生成正样本对
        # 对于每个类别的所有数据，进行双重循环生成所有可能的配对
        for data_list in class_data.values():
            for i in range(len(data_list)):
                for j in range(i + 1, len(data_list)):
                    positive_pairs.append((data_list[i], data_list[j]))
                    positive_pairs.append((data_list[j], data_list[i]))  # 添加双向对

        # 遍历不同类别生成负样本对
        # 外层循环遍历类别，内层循环遍历不同类别间的数据组合
        labels = list(class_data.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                for data_i in class_data[labels[i]]:
                    for data_j in class_data[labels[j]]:
                        negative_pairs.append((data_i, data_j))
                        negative_pairs.append((data_j, data_i))  # 添加双向对

        # 限制正负样本对数量以保持平衡
        min_pairs = min(len(positive_pairs), len(negative_pairs))
        random.shuffle(positive_pairs)
        random.shuffle(negative_pairs)
        positive_pairs = positive_pairs[:min_pairs]
        negative_pairs = negative_pairs[:min_pairs]

        return positive_pairs, negative_pairs

    # 分别为支持集和查询集生成正负样本对
    support_positive_pairs, support_negative_pairs = generate_pairs(support_set)
    query_positive_pairs, query_negative_pairs = generate_pairs(query_set)

    # 返回构建的样本对和对应标签的字典
    return {
        'support_positive_pairs': np.array(support_positive_pairs),
        'support_negative_pairs': np.array(support_negative_pairs),
        'query_positive_pairs': np.array(query_positive_pairs),
        'query_negative_pairs': np.array(query_negative_pairs)
    }


class SpatialPool(Layer):
    """
    自定义的Keras层，用于执行空间池化操作。

    该层首先对输入特征执行全局平均池化和全局最大池化，然后将这两种池化结果沿最后一个轴合并。这种结合平均和最大池化的方法
    用于提高模型在处理图像或时间序列数据时的性能。

    参数:
    inputs (tensor): 输入的特征张量，通常是神经网络中间层的输出。

    返回:
    tensor: 池化后的输出张量，其中包含了输入特征的平均池化和最大池化结果。
    """

    def __init__(self, **kwargs):
        """
        初始化层，可以接受额外的关键字参数传递给Keras层基类。
        """
        super(SpatialPool, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        计算输入张量的全局平均池化和全局最大池化，然后将这两个池化结果合并。

        参数:
        inputs (tensor): 输入的特征张量。

        返回:
        tensor: 池化后的输出张量。
        """
        # 计算输入特征的全局平均池化
        avg_pool = Kb.mean(inputs, axis=-1, keepdims=True)
        # 计算输入特征的全局最大池化
        max_pool = Kb.max(inputs, axis=-1, keepdims=True)
        # 合并平均池化和最大池化的结果
        return Kb.concatenate([avg_pool, max_pool], axis=-1)


# 计算输入向量之间的欧几里得距离
class EuclideanDistance(Layer):
    def _restore_from_tensors(self, restored_tensors):
        pass

    def _serialize_to_tensors(self):
        pass

    def __init__(self, **kwargs):
        super(EuclideanDistance, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x, y = inputs
        # 计算两个向量之间的欧几里得距离
        distance = Kb.sqrt(Kb.maximum(Kb.sum(Kb.square(x - y), axis=1, keepdims=True), Kb.epsilon()))
        return distance  # 确保输出为(batch_size, 1)

    def compute_output_shape(self, input_shape):
        # 定义输出形状
        return input_shape[0][0], 1


# 对比损失函数

class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin=initial_margin, **kwargs):
        super().__init__(**kwargs)

        # self.margin = margin  # 使用固定值，非训练参数
        # 设置margin为可训练的变量
        self.margin = tf.Variable(float(margin), dtype=tf.float32, trainable=True, name="margin")

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(self.margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        # 重写 get_config 来确保在保存和加载模型时包含自定义变量
        config = super(ContrastiveLoss, self).get_config()
        config.update({'margin': self.margin.numpy()})
        return config


def load_custom_model(model_path, margin_value):
    """
    加载包含自定义层或自定义函数的Keras模型。
    Margin: 对比损失的间距参数
    """
    custom_objects = {
        'ContrastiveLoss': ContrastiveLoss(margin=margin_value),

        # 确保自定义层被包括在内，后续可补充添加
        'SpatialPool': SpatialPool,
        'EuclideanDistance': EuclideanDistance
    }
    return load_model(model_path, custom_objects=custom_objects)


def save_model(the_model, path):
    """
    安全保存模型到指定路径。
    """
    try:
        the_model.save(path)
        return True
    except Exception as e:
        print(f"保存模型失败，错误信息：{e}")
        return False


def make_json_serializable(item):
    """
    递归地遍历给定的数据结构，检查每个元素。如果元素是 NumPy 的数据类型，它会转换为等效的 Python 类型。
    这确保了最终生成的数据结构完全由原生 Python 类型组成，可以安全地序列化为 JSON 格式
    """
    if isinstance(item, np.integer):
        return int(item)
    elif isinstance(item, np.floating):
        return float(item)
    elif isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, list):
        return [make_json_serializable(sub_item) for sub_item in item]
    elif isinstance(item, dict):
        return {key: make_json_serializable(value) for key, value in item.items()}
    else:
        return item
