import os
import re

import numpy as np
import pandas as pd
import concurrent.futures

from keras.layers import Input, Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Dense, Dropout, \
    BatchNormalization, Activation, Add, MultiHeadAttention, LayerNormalization, add, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, multiply

from keras.models import Model
from keras.optimizers import Adam

# 调用 向量距离、空间池化层、对比损失函数
from matplotlib import pyplot as plt

from Config.function import (EuclideanDistance, SpatialPool, ContrastiveLoss)
# 调用 N-way K-shot：根据支持集和查询集进行正负样本配对
from Config.function import create_balanced_pairs_from_support_query_sets
# 调用 N-way K-shot：生成支持集（support set）和查询集（query set）
from Config.function import generate_n_way_k_shot
# 调用 数据加载函数、模型自定义层、模型保存函数
from Config.function import load_and_preprocess_ecg_signals, load_custom_model, save_model

# 调用 超参数
from Config.hyper_parameter import input_shape, initial_margin, feature_extraction_layers_config, dense_layers_config, \
    learning_rate

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关闭oneDNN（MKL-DNN）优化深度学习计算
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # 开启oneDNN（MKL-DNN）优化深度学习计算

# 设置数据和文件路径
BASE_PATH = "D:\\PyCharm 2022.1\\PyCharm Documents\\Siamese Neural Networks for ECG-signal"
DATA_PATH = os.path.join(BASE_PATH, "Data")
TRAIN_H5_DIR = os.path.join(DATA_PATH, "SPH_train")
TRAIN_CSV = os.path.join(DATA_PATH, "sph_cinc_train.csv")
Saved_DIR = os.path.join(BASE_PATH, "Saved")  # 总结果保存路径
Model_save_dir = os.path.join(Saved_DIR, "Saved_Models")  # 已训练模型保存路径

# 加载训练数据并进行预处理
train_signals, train_file_labels, train_category_labels = load_and_preprocess_ecg_signals(TRAIN_H5_DIR, TRAIN_CSV)

# 将DX编码映射到连续的类别索引
'''
1. 读取CSV文件：
train_df = pd.read_csv(TRAIN_CSV)
这行代码使用Pandas的read_csv函数读取存储DX编码的CSV文件。train_df变成了一个DataFrame对象，包含CSV文件中的所有数据。

2. 提取唯一的DX编码：
unique_labels = train_df['Dx'].unique()
train_df['Dx']提取DataFrame中名为'Dx'的列，它包含了各个心电图样本的DX编码。.unique()方法则从这些DX编码中提取出所有唯一的值，即去除重复的编码，得到一个包含所有不同心脏病DX编码的数组。

3. label_to_index：
label_to_index = {label: index for index, label in enumerate(unique_labels)}
创建字典label_to_index，将每个唯一的DX编码（label）映射到一个唯一的索引（index）。enumerate(unique_labels)会遍历每个唯一的DX编码，并为其分配一个唯一的数字索引（从0开始）。对于将非数字的类别标签（例如字符串形式的DX编码）转换为可以用于机器学习模型的数值非常有用。

4. index_to_label：
index_to_label = {index: label for label, index in label_to_index.items()}
创建字典index_to_label，它是label_to_index的逆映射。它可以将之前分配的每个索引映射回它的原始DX编码。用于模型预测后将预测的类别索引转换回可读的DX编码进行后续判断。

5. index_to_name：从索引到DX的中文名称的映射
'''

train_df = pd.read_csv(TRAIN_CSV)
unique_labels = train_df['Dx'].unique()
label_to_index = {label: index for index, label in enumerate(unique_labels)}
index_to_label = {index: label for label, index in label_to_index.items()}


# 假设预览 ecg_signals = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20,
# 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]])
# ecg_labels = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C','D'])
# label_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
# index_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout_rate=0.1):
    """Transformer 编码器块，包含多头自注意力和前馈网络。

    参数:
      inputs (Tensor): 输入特征，形状为 (batch_size, steps, filters)。
      head_size (int): 每个头的维度大小。
      num_heads (int): 多头注意力中的头数。
      ff_dim (int): 前馈网络中间层的维度。
      dropout_rate (float): Dropout比率，用于正则化以防止过拟合。

    返回:
      Tensor: Transformer层的输出。
    """
    # 多头自注意力
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout_rate)(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(add([inputs, attention_output]))

    # 前馈网络
    ffn_output = Dense(ff_dim, activation='relu')(attention_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = LayerNormalization(epsilon=1e-6)(add([attention_output, ffn_output]))
    return ffn_output


def cbam_block(input_feature, ratio=16, kernel_size=3):
    """ CBAM（Convolutional Block Attention Module），卷积块注意力模块，针对一维时间序列数据
    Args:
    input_feature: 输入特征图 (batch_size, steps, filters)
    ratio: 神经元缩减比例
    kernel_size: 空间注意力机制卷积核大小

    Returns:
    x: 经过注意力机制调整后的特征图
    """
    channel_avg_pool = GlobalAveragePooling1D()(input_feature)
    channel_max_pool = GlobalMaxPooling1D()(input_feature)
    channel_avg_out = Dense(input_feature.shape[-1] // ratio, activation='relu')(channel_avg_pool)
    channel_max_out = Dense(input_feature.shape[-1] // ratio, activation='relu')(channel_max_pool)
    channel_avg_out = Dense(input_feature.shape[-1], activation='sigmoid')(channel_avg_out)
    channel_max_out = Dense(input_feature.shape[-1], activation='sigmoid')(channel_max_out)
    channel_out = Add()([channel_avg_out, channel_max_out])

    # 空间注意力机制
    spatial_pool = SpatialPool()(input_feature)
    spatial_out = Conv1D(1, kernel_size=kernel_size, padding='same', activation='sigmoid')(spatial_pool)

    # 应用通道和空间注意力机制
    x = multiply([input_feature, channel_out])
    x = multiply([x, spatial_out])
    return x


'''
residual_block 函数，定义残差块，自适应是否使用1x1卷积调整通道数
残差块的目标是让网络学习输入与输出之间的残差（差异），通过将快捷连接的输出（原始输入特征）和主路径的输出（学到的残差）相加，。
网络可以直接优化这个残差，而不是从头学习整个输出，这使得训练更加高效。

参数的含义如下：
1. x: 函数的输入，即残差块的输入特征图（Feature Map）。
2. filters: 卷积层中的过滤器（或称为核）数量。这决定了卷积层的输出特征图的深度。
3. kernel_size: 卷积核的大小，决定了卷积操作覆盖的区域大小。
4. strides: 卷积时的步长，即卷积核移动的距离。
'''


# 构建残差块
def residual_block(x, filters, kernel_size, strides, use_cbam=False, cbam_kernel_size=3, cbam_ratio=16):
    # 主路径
    shortcut = x
    x = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)  # 批归一化。正规化卷积层的输出，使其在网络的不同层之间保持数值稳定，助于加快训练过程。
    x = Activation('relu')(x)

    x = Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    if use_cbam:
        x = cbam_block(x, ratio=cbam_ratio, kernel_size=cbam_kernel_size)  # 应用CBAM

    # 如果通道数改变，则在快捷连接中使用1x1卷积
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # 相加并激活
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


# 制定 CNN 网络架构
def build_custom_cnn(Input_shape, Feature_extraction_layers_config, Dense_layers_config):
    input_layer = Input(shape=Input_shape)
    x = input_layer

    for layer_config in Feature_extraction_layers_config:
        if layer_config['type'] == 'conv':  # 卷积层处理
            x = Conv1D(filters=layer_config['filters'], kernel_size=layer_config['kernel_size'],
                       strides=layer_config['strides'], padding='same')(x)
            if layer_config.get('batch_norm', False):  # 默认为不使用批归一化
                x = BatchNormalization()(x)
            if layer_config.get('activation', False):  # 指定激活函数，默认无激活函数
                x = Activation(layer_config['activation'])(x)
            if layer_config.get('dropout', 0):  # 应用 Dropout，默认返回键值 0（例：如返回0.3，则每次在训练期间，30%的节点随机失活）
                x = Dropout(layer_config['dropout'])(x)

        elif layer_config['type'] == 'pool':
            # 平均池化（Average Pooling）
            if layer_config.get('pool_type', 'max') == 'max':
                x = MaxPooling1D(pool_size=layer_config.get('pool_size', 2),
                                 strides=layer_config.get('strides', 2),
                                 padding=layer_config.get('padding', 'same'))(x)
            # 最大池化（Max Pooling）
            elif layer_config.get('pool_type', 'max') == 'average':
                x = AveragePooling1D(pool_size=layer_config.get('pool_size', 2),
                                     strides=layer_config.get('strides', 2),
                                     padding=layer_config.get('padding', 'same'))(x)

        elif layer_config['type'] == 'residual':
            # 应用残差块
            use_cbam = layer_config.get('use_cbam', False)  # 判断是否激活 CBAM
            x = residual_block(x, filters=layer_config['filters'],
                               kernel_size=layer_config['kernel_size'],
                               strides=layer_config['strides'], use_cbam=use_cbam)

        # 判断是否启用Transformer层
        elif layer_config['type'] == 'transformer' and layer_config.get('transformer_apply', False):
            x = transformer_encoder(x, head_size=layer_config['head_size'],
                                    num_heads=layer_config['num_heads'],
                                    ff_dim=layer_config['ff_dim'],
                                    dropout_rate=layer_config['dropout_rate'])

        # ...（其他层的配置）

    x = Flatten()(x)

    for dense_layer in Dense_layers_config:  # 单独处理密集层配置
        x = Dense(dense_layer['units'], activation='relu')(x)
        if dense_layer.get('dropout', 0) > 0:
            x = Dropout(dense_layer['dropout'])(x)

    custom_model = Model(inputs=input_layer, outputs=x)
    return custom_model


# 构建孪生网络的部分
def build_siamese_network(Input_shape, Feature_extraction_layers_config, Dense_layers_config):
    # 基础CNN模型
    base_cnn = build_custom_cnn(Input_shape, Feature_extraction_layers_config, Dense_layers_config)

    # 孪生网络的两个输入
    input_a = Input(shape=Input_shape)
    input_b = Input(shape=Input_shape)

    # 通过同一个网络处理两个输入
    processed_a = base_cnn(input_a)
    processed_b = base_cnn(input_b)

    # 使用自定义的EuclideanDistance层来计算两个特征向量之间的距离
    distance = EuclideanDistance()([processed_a, processed_b])

    # 创建模型，输出为特征向量距离
    siamese_network = Model(inputs=[input_a, input_b], outputs=distance)
    loss_function = ContrastiveLoss(margin=initial_margin)

    # 应用对比损失层
    siamese_network.compile(optimizer='adam', loss=ContrastiveLoss(margin=initial_margin), metrics=['accuracy'])

    return siamese_network, loss_function


'''模型训练及后续保存'''

"""
    训练孪生网络模型的参数：

    siamese_network: 孪生网络架构。
    train_signals: 训练数据中的心电信号。
    train_category_labels: 训练数据中的类别标签。
    label_to_index: 将标签映射到索引的字典。
    index_to_label: 将索引映射回标签的字典。
    save_dir: 模型保存目录。
    num_epochs: 训练的 epoch 数，默认为 10。
    num_iterations_per_epoch: 预期在每个 epoch 中生成集合的次数，默认为 1000。
    N: N-way分类，即每次选择的类别数，默认为 5。
    Kn: K-shot训练，即每个类别的样本数，默认为 5。
    Q: 每个类别的查询样本数，默认为 2。
    learning_rate: 学习率，默认为 0.001。
    continue_training: 是否继续之前的训练，默认为 False。
    continue_model_file: 需继续训练的已有模型文件路径。
    use_transformer: 判断是否使用自注意力机制，默认为 False。
    feature_extraction_layers_config：特征提取层配置参数。
    dense_layers_config：密集层配置参数。
    transformer_config：自注意机制配置参数。
"""

# 异步数据生成函数 async_data_generation
'''
    # 模拟数据生成过程
    # 该函数用于生成每个epoch需要的支持集和查询集
    # 参数:
    # epoch - 当前的训练轮次
    # Train_signals - 训练用的信号数据
    # Train_category_labels - 训练数据的类别标签
    # N - N-way分类的类别数
    # Kn - 每类中选择的样本数（K-shot）
    # Label_to_index - 类别标签到索引的映射
    # Index_to_label - 索引到类别标签的映射
    # Q - 查询集中每类的样本数
'''


def async_data_generation(epoch, Train_signals, Train_category_labels, N, Kn, Label_to_index, Index_to_label, Q):
    # 模拟数据生成过程，可以根据实际情况替换
    support_set, query_set, _ = generate_n_way_k_shot(Train_signals, Train_category_labels, N, Kn, Label_to_index,
                                                      Index_to_label, Q)
    pairs_data = create_balanced_pairs_from_support_query_sets(support_set, query_set)
    return pairs_data


def prepare_model_directory(base_save_dir):
    """
    准备模型保存目录，如果不存在则创建一个新的，编号最高的目录。

    Args:
    - base_save_dir: 模型保存的基本目录

    Returns:
    - save_dir: 创建或选择的最新模型目录的完整路径
    """
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)

    # 搜索现有的模型文件夹，找到最高的编号
    highest_num = -1
    pattern = re.compile(r'^model_(\d+)$')
    for dir_name in os.listdir(base_save_dir):
        if os.path.isdir(os.path.join(base_save_dir, dir_name)):
            match = pattern.match(dir_name)
            if match:
                num = int(match.group(1))
                if num > highest_num:
                    highest_num = num

    # 创建新的模型目录，编号为当前最高编号+1
    new_model_dir_name = f"model_{highest_num + 1}"
    new_model_dir_path = os.path.join(base_save_dir, new_model_dir_name)
    os.makedirs(new_model_dir_path, exist_ok=True)

    return new_model_dir_path


def train_siamese_network(Train_signals, Train_category_labels, Label_to_index, Index_to_label, config):
    # 定义全局变量
    loss_function = None

    # 从配置字典中提取所有设置
    base_save_dir = config.get('save_dir', '.')
    num_epochs = config.get('num_epochs', 10)
    num_iterations_per_epoch = config.get('num_iterations_per_epoch', 1000)
    N = config.get('N', 5)
    Kn = config.get('Kn', 5)
    Q = config.get('Q', 2)
    Learning_rate = config.get('learning_rate', 0.001)
    Margin = config.get('margin', 1.0)
    continue_training = config.get('continue_training', False)
    continue_model_file = config.get('continue_model_file', 'model_0_best.h5')

    # 预设置初始的最佳模型路径，避免赋值前引用
    save_dir = prepare_model_directory(base_save_dir)
    Best_model_path = os.path.join(save_dir, "")

    # 网络架构配置参数修改详见 hyper_parameter.py, 训练函数默认为空
    Feature_extraction_layers_config = config.get('feature_extraction_layers_config', None)
    Dense_layers_config = config.get('dense_layers_config', None)

    # 初始化线程池执行器，max_workers指定了线程池中的线程数量，若设置为1，则意味着在任何时刻最多只有一个任务在单独的线程中运行
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    # 判断并创建模型保存路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    '''获取目前已经保存的模型数量，用于编号'''
    # 直接在列表推导中进行文件名的匹配和转换，使用生成器表达式配合max函数获取最大值
    model_count = max((int(re.search(r'model_(\d+)_final', f).group(1))
                       for f in os.listdir(save_dir)
                       if re.match(r'model_\d+_final', f) and os.path.isfile(os.path.join(save_dir, f))),
                      default=-1) + 1

    # 初始化或加载模型
    if continue_training:
        continue_model_path = os.path.join(save_dir, continue_model_file)
        if os.path.exists(continue_model_path):
            print("加载模型继续训练...")
            siamese_network = load_custom_model(continue_model_path, Margin)
        else:
            print("未找到模型文件，将开始新的训练...")
            siamese_network, loss_function = build_siamese_network(input_shape, Feature_extraction_layers_config,
                                                                   Dense_layers_config)

            siamese_network.compile(optimizer=Adam(Learning_rate), loss=ContrastiveLoss(Margin), metrics=['accuracy'])
    else:
        print("开始新的训练...")
        siamese_network, loss_function = build_siamese_network(input_shape, Feature_extraction_layers_config,
                                                               Dense_layers_config)

        siamese_network.compile(optimizer=Adam(Learning_rate), loss=ContrastiveLoss(Margin), metrics=['accuracy'])

    best_support_loss = float('inf')  # 初始化支持集最佳损失
    best_query_loss = float('inf')  # 初始化查询集最佳损失
    best_epoch_support = 0  # 初始化支持集得到最佳损失的 epoch 位次
    best_epoch_query = 0  # 初始化查询集得到最佳损失的 epoch 位次
    final_epochs = 0  # 初始化实际训练最终轮次

    best_margin = None  # 添加变量来存储最佳的 margin
    patience_counter = 0  # 对于提早停止，损失无改善计数器
    reduce_lr_patience = 10  # 降低学习率的epoch数耐心

    iteration_losses_support = []  # 初始化支持集单个 epoch 中每次迭代的损失集合
    iteration_losses_query = []  # 初始化查询集单个 epoch 中每次迭代的损失集合
    epoch_average_loss_support = []  # 初始化支持集所有 epoch 的平均损失集合
    epoch_average_loss_query = []  # 初始化查询集所有 epoch 的平均损失集合
    latest_query_loss = float('inf')
    learn_state = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}...")

        # 判断是否继续已有模型训练
        continue_model_filename = os.path.join(save_dir, "final_model.h5")  # 自定义继续训练的模型
        if continue_training and os.path.exists(continue_model_filename):
            print("加载模型继续训练...")
            siamese_network = load_custom_model(continue_model_filename, initial_margin)

        else:
            print("开始新的训练...")

        # 异步加载数据，# 从支持集和查询集生成样本对
        future = executor.submit(async_data_generation, epoch, Train_signals, Train_category_labels, N, Kn,
                                 Label_to_index, Index_to_label, Q)
        pairs_data = future.result()

        # 构建复循环，进行迭代训练
        for iteration in range(num_iterations_per_epoch):
            print(f"Epoch {epoch + 1}/{num_epochs} --- Iteration {iteration + 1}/{num_iterations_per_epoch}")

            # 生成支持集和查询集（根据参数设置 N-way Kn-shot 的任务，其中每个类别有 Q 个查询样本）
            support_set, query_set, _ = generate_n_way_k_shot(Train_signals, Train_category_labels, N, Kn,
                                                              Label_to_index, Index_to_label, Q)
            print(f"支持集样本数： {len(support_set)}")
            print(f"查询集样本数： {len(query_set)}")

            # 训练支持集样本对
            support_pairs = np.concatenate((pairs_data['support_positive_pairs'], pairs_data['support_negative_pairs']))
            support_labels = np.array(
                [1] * len(pairs_data['support_positive_pairs']) + [0] * len(pairs_data['support_negative_pairs']))

            # 训练模型：使用支持集
            support_loss = siamese_network.train_on_batch([support_pairs[:, 0], support_pairs[:, 1]], support_labels)

            # 加载查询集样本对及其标签
            query_pairs = np.concatenate((pairs_data['query_positive_pairs'], pairs_data['query_negative_pairs']))
            query_labels = np.array(
                [1] * len(pairs_data['query_positive_pairs']) + [0] * len(pairs_data['query_negative_pairs']))

            # 验证模型效果：使用查询集
            query_loss, query_accuracy = siamese_network.evaluate([query_pairs[:, 0], query_pairs[:, 1]], query_labels)

            iteration_losses_support.append(support_loss)  # 添加当前迭代的损失到列表中
            iteration_losses_query.append(query_loss)  # 添加当前迭代的损失到列表中

        # 计算支持集单个 epoch 所有损失的平均值
        epoch_support_loss = np.mean(iteration_losses_support)
        print(f"Epoch {epoch + 1} 支持集: {epoch_support_loss:.4f}")
        epoch_average_loss_support.append(epoch_support_loss)

        # 计算查询集单个 epoch 所有损失的平均值
        epoch_query_loss = np.mean(iteration_losses_query)
        print(f"Epoch {epoch + 1} 查询集: {epoch_query_loss:.4f}")
        epoch_average_loss_query.append(epoch_query_loss)

        # 保存支持集最佳模型
        if epoch_support_loss < best_support_loss:
            best_support_loss = epoch_support_loss
            best_epoch_support = epoch + 1
            best_margin = loss_function.margin  # 获取当前的最佳 margin
            # model_filename_support = f"model_{model_count}_support_best.h5"
            model_filename_support = f"model_support_best.h5"
            Best_model_path = os.path.join(save_dir, model_filename_support)
            # 尝试保存模型
            if save_model(siamese_network, Best_model_path):
                print(f"已保存支持集性能最佳模型至 {Best_model_path}，当前损失为{best_support_loss}。")
            else:
                print(f"未能保存支持集性能最佳模型至 {Best_model_path}。")

        # 保存查询集最佳模型
        if epoch_query_loss < best_query_loss:
            best_query_loss = epoch_query_loss
            best_epoch_query = epoch + 1
            best_margin = loss_function.margin  # 获取当前的最佳 margin
            # model_filename_query = f"model_{model_count}_query_best.h5"
            model_filename_query = f"model_query_best.h5"
            Best_model_path = os.path.join(save_dir, model_filename_query)
            patience_counter = 0  # 重置耐心计数器
            # 尝试保存模型
            if save_model(siamese_network, Best_model_path):
                print(f"已保存查询集性能最佳模型至 {Best_model_path}，当前损失为{best_query_loss}。")
            else:
                print(f"未能保存查询集性能最佳模型至 {Best_model_path}。")
        else:
            if epoch_query_loss < latest_query_loss:
                patience_counter += 1  # 损失没有改善则增加计数器

        latest_query_loss = epoch_query_loss

        # 动态调整学习率
        if patience_counter > reduce_lr_patience and learn_state == 0:
            old_lr = siamese_network.optimizer.learning_rate.numpy()  # 获取当前学习率
            new_lr = old_lr * 0.5
            siamese_network.optimizer.learning_rate.assign(new_lr)  # 更新学习率
            learn_state = 1
            print(f"学习率自动更新： {old_lr} to {new_lr}")

        if patience_counter < reduce_lr_patience and learn_state == 1:
            old_lr = siamese_network.optimizer.learning_rate.numpy()  # 获取当前学习率
            new_lr = Learning_rate * 2.0
            siamese_network.optimizer.learning_rate.assign(new_lr)  # 更新学习率
            learn_state = 0
            print(f"学习率自动更新： {old_lr} to {new_lr}")

        # 提早停止
        if patience_counter >= 40:
            print("Early stopping triggered.")
            final_epochs = epoch
            break

    # 保存最后一次 epoch 模型
    # final_model_filename = f"model_{model_count}_final.h5"
    final_model_filename = f"model_final.h5"
    final_model_path = os.path.join(save_dir, final_model_filename)

    if save_model(siamese_network, final_model_path):
        print(f"已保存最后一次epoch模型至 {final_model_path}。")
    else:
        print(f"未能保存最后一次epoch模型至 {final_model_path}。")

    # 打印支持集每个 epoch 的平均损失与最佳 margin
    average_loss_support = np.mean(epoch_average_loss_support)
    print("--------------------------")
    print(f"已完成支持集 {final_epochs} 轮次训练。平均损失为: {average_loss_support:.4f}。最佳训练轮次 epoch：{best_epoch_support}")
    print(f"最佳 margin: {best_margin.numpy():.4f}")

    # 打印查询集每个 epoch 的平均损失与最佳 margin
    average_loss_query = np.mean(epoch_average_loss_query)
    print("--------------------------")
    print(f"已完成查询集 {final_epochs} 轮次训练。平均损失为: {average_loss_query:.4f}。最佳训练轮次 epoch：{best_epoch_query}")
    print(f"最佳 margin: {best_margin.numpy():.4f}")

    # 绘制支持集训练损失变化图
    #
    # print(len(epoch_average_loss_support))
    # print(type(epoch_average_loss_support))
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_average_loss_support, label=f'Learning Rate = {learning_rate}, Margin = {Margin}')
    plt.xlabel('Epoch')
    plt.ylabel('平均损失')
    plt.title('支持集训练损失变化一览')
    plt.legend()
    plt.grid(True)

    # 图像保存路径
    # plot_filename_support = os.path.join(save_dir, f"model_{model_count}_支持集训练损失变化一览.png")
    plot_filename_support = os.path.join(save_dir, f"model_支持集训练损失变化一览.png")
    plt.savefig(plot_filename_support, dpi=600)  # 保存图像到文件
    print(f"支持集训练损失变化图像已保存至：{plot_filename_support}")
    plt.show()

    # 绘制查询集训练损失变化图
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_average_loss_query, label=f'Learning Rate = {learning_rate}, Margin = {Margin}')
    plt.xlabel('Epoch')
    plt.ylabel('平均损失')
    plt.title('查询集训练损失变化一览')
    plt.legend()
    plt.grid(True)

    # 图像保存路径
    # plot_filename_query = os.path.join(save_dir, f"model_{model_count}_查询集训练损失变化一览.png")
    plot_filename_query = os.path.join(save_dir, f"model_查询集训练损失变化一览.png")
    plt.savefig(plot_filename_query, dpi=600)  # 保存图像到文件
    print(f"支持集训练损失变化图像已保存至：{plot_filename_query}")
    plt.show()

    # 将结果写入到文件
    # train_file_path_support = os.path.join(save_dir, f"model_support_{model_count}_轮次细节.csv")
    train_file_path_support = os.path.join(save_dir, f"model_support_轮次细节.csv")
    with open(train_file_path_support, 'w') as file:
        file.write(f"支持集应训练轮次：{num_epochs}，实际完成轮次： {final_epochs}；每轮次epoch数目：{num_iterations_per_epoch}\n")
        file.write(f"{N}-way {Kn}-shot learning; {Q}-sample query set\n")
        file.write(f"平均损失为: {average_loss_support:.4f}。最佳训练轮次 epoch：{best_epoch_support}\n")
        file.write(f"最佳 margin: {best_margin.numpy():.4f}\n")
        file.write(f"Learning_rate: {Learning_rate:.4f}\n")
    print(f"支持集训练轮次细节已保存至：{train_file_path_support}")

    # 将结果写入到文件
    # train_file_path_query = os.path.join(save_dir, f"model_query_{model_count}_轮次细节.csv")
    train_file_path_query = os.path.join(save_dir, f"model_query_轮次细节.csv")
    with open(train_file_path_query, 'w') as file:
        file.write(f"查询集集应测试轮次：{num_epochs}，实际完成轮次： {final_epochs}；每轮次epoch数目：{num_iterations_per_epoch}\n")
        file.write(f"{N}-way {Kn}-shot learning; {Q}-sample query set\n")
        file.write(f"平均损失为: {average_loss_query:.4f}。最佳训练轮次 epoch：{best_epoch_query}\n")
        file.write(f"最佳 margin: {best_margin.numpy():.4f}\n")
        file.write(f"Learning_rate: {Learning_rate:.4f}\n")
    print(f"训练轮次细节已保存至：{train_file_path_query}")

    # 返回模型和平均损失
    return siamese_network, average_loss_support, average_loss_query, Best_model_path


# 模型训练实例

# 检查目录是否存在，如果不存在则创建
if not os.path.exists(Model_save_dir):
    os.makedirs(Model_save_dir)

train_config = {
    'save_dir': Model_save_dir,
    'num_epochs': 300,
    'num_iterations_per_epoch': 20,
    'N': 5,
    'Kn': 5,
    'Q': 2,
    'learning_rate': learning_rate,
    'margin': initial_margin,
    'continue_training': False,
    # 'continue_training': True,
    'continue_model_file': 'model_0_best.h5',

    # 网络架构配置参数修改详见 hyper_parameter.py, 训练函数默认为空
    'feature_extraction_layers_config': feature_extraction_layers_config,
    'dense_layers_config': dense_layers_config
}

model, avg_loss_support, avg_loss_query, best_model_path = train_siamese_network(train_signals, train_category_labels,
                                                                                 label_to_index, index_to_label,
                                                                                 train_config)

# print(best_model_path)

'''孪生网络输入输出结构测试

# 控制变量
# enable_testing：决定是否执行测试。
# use_trained_model：决定是否使用已训练的模型进行测试。
'''

# enable_testing = False
# use_trained_model = False
enable_testing = True
use_trained_model = True

if enable_testing:
    # 预先决定是否加载已训练模型或构建新模型，可指定，默认继续本次训练最佳模型测试
    # the_model_filename = "model_1_best.h5"
    # the_test_model_path = os.path.join(Model_save_dir, the_model_filename)

    the_test_model_path = best_model_path
    test_filename = os.path.basename(the_test_model_path)

    if use_trained_model:
        if os.path.exists(best_model_path):
            print("--------------------------")
            print(f"加载已训练的孪生网络 {test_filename} 用于测试...")
            Test_network = load_custom_model(best_model_path, initial_margin)
        else:
            print("--------------------------")
            print("没有找到有效的训练模型...一个新模型已经创建以供测试...")
            Test_network, _ = build_siamese_network(input_shape,
                                                    Feature_extraction_layers_config=feature_extraction_layers_config,
                                                    Dense_layers_config=dense_layers_config)
            Test_network.compile(loss=ContrastiveLoss, optimizer=Adam(learning_rate))
    else:
        print("--------------------------")
        print("构建新的孪生网络用于测试...")
        Test_network, _ = build_siamese_network(input_shape,
                                                Feature_extraction_layers_config=feature_extraction_layers_config,
                                                Dense_layers_config=dense_layers_config)
        Test_network.compile(loss=ContrastiveLoss, optimizer=Adam(learning_rate))

    # 假设批大小为10，每个批中有12个通道，每个通道5000个数据点
    batch_size = 10
    test_input_a = np.random.random((batch_size, 12, 5000))
    test_input_b = np.random.random((batch_size, 12, 5000))
    # test_input_b = test_input_a

    # 获取模型输出
    test_output = Test_network.predict([test_input_a, test_input_b])

    # 打印输出形状并预览
    print("Output shape:", test_output.shape)
    print("网络输出预览:\n", test_output[:-1])  # 显示前5个输出样本的结果，距离应是非负值。

    # 保存测试网络模型
    test_save_dir = os.path.join(Saved_DIR, "Saved_architecture")
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)
    saved_model_path = os.path.join(test_save_dir, "enhanced_siamese_network.h5")
    Test_network.save(saved_model_path)
    print(f"网络模型保存路径： {saved_model_path}.")
