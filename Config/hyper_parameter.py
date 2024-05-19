# 心电图12导联输入，输入信号的形状
input_shape = (12, 5000)

# 对比损失的间距参数
initial_margin = 3

# 神经网络学习率
learning_rate = 0.001

# 特征提取层配置
'''考虑从更多的过滤器开始，以更好地捕捉心电信号的复杂性。
使用较小的卷积核（如3x3），帮助模型捕捉更细粒度的时间特征。
在某些层中使用平均池化代替最大池化，以保留更多的背景信息，'''

feature_extraction_layers_config = [
    # 第一层卷积：较小的卷积核和更多的过滤器可以捕捉细微的信号变化
    {'type': 'conv', 'filters': 32, 'kernel_size': 3, 'strides': 1, 'batch_norm': True, 'activation': 'relu',
     'dropout': 0.1},

    # 第一层池化：使用平均池化以保留更多原始信号的信息
    {'type': 'pool', 'pool_size': 2, 'pool_type': 'average'},

    # 第二层卷积：增加过滤器数量，加深特征提取
    {'type': 'conv', 'filters': 64, 'kernel_size': 3, 'strides': 1, 'batch_norm': True, 'activation': 'relu',
     'dropout': 0.1},

    # 第二层池化：继续使用平均池化
    {'type': 'pool', 'pool_size': 2, 'pool_type': 'average'},

    # 引入Transformer层以增加模型对时序数据的理解
    {'type': 'transformer', 'transformer_apply': True, 'head_size': 64, 'num_heads': 4, 'ff_dim': 256,
     'dropout_rate': 0.1},

    # 第一个残差块：使用残差连接来增强信号流，包含CBAM以关注重要的特征
    {'type': 'residual', 'filters': 128, 'kernel_size': 3, 'strides': 1, 'use_cbam': True, 'cbam_kernel_size': 3,
     'cbam_ratio': 16},

    # 进一步的卷积和池化层，逐渐增加复杂度
    {'type': 'conv', 'filters': 128, 'kernel_size': 3, 'strides': 1, 'batch_norm': True, 'activation': 'relu',
     'dropout': 0.1},
    {'type': 'pool', 'pool_size': 2, 'pool_type': 'max'},
    {'type': 'residual', 'filters': 256, 'kernel_size': 3, 'strides': 1, 'use_cbam': True, 'cbam_kernel_size': 3,
     'cbam_ratio': 16},
    {'type': 'conv', 'filters': 256, 'kernel_size': 3, 'strides': 1, 'batch_norm': True, 'activation': 'relu',
     'dropout': 0.1},
    {'type': 'pool', 'pool_size': 2, 'pool_type': 'max'},
    {'type': 'residual', 'filters': 512, 'kernel_size': 3, 'strides': 1, 'use_cbam': True, 'cbam_kernel_size': 3,
     'cbam_ratio': 16}
]

# '''
# 基于 VGG 层配置网络
# '''
# '''
# 自注意机制配置层参数：
#     'head_size': 每个注意力头的维度
#     'num_heads': 注意力头数
#     'ff_dim': 前馈网络的宽度
#     'dropout_rate': Dropout比率
# '''
# feature_extraction_layers_config = [
#     {'type': 'conv', 'filters': 64, 'kernel_size': 3, 'strides': 1, 'activation': 'relu', 'batch_norm': True,
#      'dropout': 0.1},
#     {'type': 'pool', 'pool_size': 2, 'pool_type': 'max', 'strides': 2},
#     {'type': 'conv', 'filters': 128, 'kernel_size': 3, 'strides': 1, 'activation': 'relu', 'batch_norm': True,
#      'dropout': 0.1},
#     {'type': 'pool', 'pool_size': 2, 'pool_type': 'max', 'strides': 2},
#     {'type': 'conv', 'filters': 256, 'kernel_size': 3, 'strides': 1, 'activation': 'relu', 'batch_norm': True,
#      'dropout': 0.1},
#     {'type': 'conv', 'filters': 256, 'kernel_size': 3, 'strides': 1, 'activation': 'relu', 'batch_norm': True,
#      'dropout': 0.1},
#     {'type': 'pool', 'pool_size': 2, 'pool_type': 'max', 'strides': 2},
#     {'type': 'conv', 'filters': 512, 'kernel_size': 3, 'strides': 1, 'activation': 'relu', 'batch_norm': True,
#      'dropout': 0.1},
#     {'type': 'conv', 'filters': 512, 'kernel_size': 3, 'strides': 1, 'activation': 'relu', 'batch_norm': True,
#      'dropout': 0.1},
#     {'type': 'pool', 'pool_size': 2, 'pool_type': 'max', 'strides': 2},
#     # 更多层可按需添加
# ]

# feature_extraction_layers_config = [
#     # 第一层卷积：目标是捕捉ECG信号中的初步特征，如基线噪声和波形起始点。
#     {'type': 'conv', 'filters': 64, 'kernel_size': 5, 'strides': 1, 'activation': 'relu', 'batch_norm': True,
#      'dropout': 0.1},
#
#     # 第一层池化：通过降低特征维度来增加后续层的感受野，并减少计算量。
#     {'type': 'pool', 'pool_size': 2, 'pool_type': 'max', 'strides': 2},
#
#     # 第二层卷积：增加过滤器数量，进一步提取ECG信号中的复杂特征，如QRS波群。
#     {'type': 'conv', 'filters': 128, 'kernel_size': 5, 'strides': 1, 'activation': 'relu', 'batch_norm': True,
#      'dropout': 0.1},
#
#     # 引入第一个残差块：增强网络的学习能力，避免深层网络中的梯度消失问题。
#     {'type': 'residual', 'filters': 128, 'kernel_size': 5, 'strides': 1, 'use_cbam': True,
#      'cbam_ratio': 16, 'cbam_kernel_size': 3},
#
#     # 第三层卷积：进一步增强特征提取能力。
#     {'type': 'conv', 'filters': 256, 'kernel_size': 3, 'strides': 1, 'activation': 'relu', 'batch_norm': True,
#      'dropout': 0.1},
#
#     # 引入第二个残差块：使用CBAM关注QRS波等关键特征。
#     {'type': 'residual', 'filters': 256, 'kernel_size': 3, 'strides': 1, 'use_cbam': True,
#      'cbam_ratio': 16, 'cbam_kernel_size': 3},
#
#     # 第四层池化：进一步降低特征维度，准备进行全局特征学习。
#     {'type': 'pool', 'pool_size': 2, 'pool_type': 'max', 'strides': 2},
#
#     # 引入Transformer层：用于捕获长距离依赖，理解心电信号中的时间动态特性。
#     {'type': 'transformer', 'transformer_apply': True, 'head_size': 64, 'num_heads': 4, 'ff_dim': 256,
#      'dropout_rate': 0.1}
# ]

# 密集层配置：用于最终特征的汇总和分类。
dense_layers_config = [
    {'units': 256, 'dropout': 0.2},  # 高维特征处理
    {'units': 128, 'dropout': 0.15},  # 进一步特征压缩
    {'units': 64, 'dropout': 0.1}
]
