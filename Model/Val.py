import os

import numpy as np
import pandas as pd
import seaborn as sns

from keras.callbacks import LambdaCallback
from keras.optimizers import Adam

from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from Config.function import (load_and_preprocess_ecg_signals, generate_n_way_k_shot,
                             create_balanced_pairs_from_support_query_sets,
                             load_custom_model, ContrastiveLoss)
from Config.hyper_parameter import initial_margin


def prepare_datasets(csv_data_path, h5_data_folder, N, K, Q):
    """
    准备支持集和查询集，并生成正负样本对。
    Args:
        csv_data_path (str): 存放信号标签的.csv文件路径。
        h5_data_folder (str): 存放.h5格式ECG信号的文件夹路径。
        N (int): N-way分类的类别数。
        K (int): 每类的样本数K-shot。
        Q (int): 查询集中每类的样本数。
    Returns:
        dict: 包含支持集和查询集的正负样本对数组的字典，包含DX编号。
    """
    # 加载和预处理数据
    signals, _, labels = load_and_preprocess_ecg_signals(h5_data_folder, csv_data_path)
    train_df = pd.read_csv(csv_data_path)
    unique_labels = train_df['Dx'].unique()
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}

    # 生成支持集和查询集
    support_set, query_set, category_dx_details = generate_n_way_k_shot(signals, labels, N, K, label_to_index,
                                                                        index_to_label, Q)

    # 创建样本对
    pairs_data = create_balanced_pairs_from_support_query_sets(support_set, query_set)

    return pairs_data, category_dx_details


def load_dx_translation_table(csv_path):
    """
    加载并解析CSV文件，然后根据提取的DX编号查找相应的中文疾病名称
    """
    # 加载CSV文件
    dx_table = pd.read_csv(csv_path, dtype={'DX': str})  # 强制DX列为字符串类型
    dx_table['DX'] = dx_table['DX'].str.strip()  # 移除可能的空格
    # 创建字典以DX编码为键，中文名称为值
    dx_to_chinese = pd.Series(dx_table['Chinese_Name'].values, index=dx_table['DX']).to_dict()
    return dx_to_chinese


def fine_tune_model(model, support_data, Margin, Learning_rate, epochs):
    """
    使用支持集对模型进行预学习微调。
    Args:
        model (tf.keras.Model): 要微调的孪生网络模型。
        support_data: 样本对支持字典。
        Margin: 对比损失的间距参数.
        Learning_rate (float): 微调过程中的学习率。
        epochs (int): 微调的周期数，默认为5。

    """

    support_pairs = np.concatenate((support_data['support_positive_pairs'], support_data['support_negative_pairs']),
                                   axis=0)
    support_labels = np.array(
        [1] * len(support_data['support_positive_pairs']) + [0] * len(support_data['support_negative_pairs']))

    model = load_custom_model(model_path, initial_margin)
    model.compile(optimizer=Adam(Learning_rate), loss=ContrastiveLoss(Margin), metrics=['accuracy'])

    # 创建回调函数来打印训练进度
    print_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(
            f'Epoch {epoch + 1}: Loss={logs["loss"]:.4f}, Accuracy={logs["accuracy"]:.4f}')
    )

    # 训练模型
    support_data = [support_pairs[:, 0], support_pairs[:, 1]]
    model.fit(support_data, support_labels, epochs=epochs, verbose=1, callbacks=[print_callback])


def plot_metrics(metrics_dict, plot_type, save_path=None):
    """
    绘制给定指标的图表。

    Args:
        metrics_dict (dict): 包含各个阈值下的性能指标，包括混淆矩阵等。
        plot_type (str): 指定绘制图表的类型（'confusion', 'pr', 'sensitivity_specificity'）。
        save_path (str): 图表保存路径。如果提供，图表将被保存而不是显示。
    """
    # 定义全局变量
    Category_save_path = None

    # 检查并创建图表可视化保存目录
    if save_path:
        # 创建图表类型对应的子目录
        Category_save_path = os.path.join(save_path, plot_type)
        if not os.path.exists(Category_save_path):
            os.makedirs(Category_save_path)

    if plot_type == 'confusion':
        # 绘制混淆矩阵
        for threshold, metrics in metrics_dict.items():
            cm = np.array(metrics['confusion_matrix'])
            plt.figure()
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False)
            plt.title(f'混淆矩阵 - 阈值: {threshold:.2f}')
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            if save_path:
                plt.savefig(os.path.join(Category_save_path, f'Confusion_Matrix_{threshold:.2f}.png'), dpi=600)
            else:
                plt.show()
            plt.close()

    elif plot_type == 'pr':
        # 绘制精确率-召回率曲线
        precision_list, recall_list, thresholds = [], [], []

        # 数据加载
        for threshold, metrics in metrics_dict.items():
            precision_list.append(metrics['precision'])
            recall_list.append(metrics['recall'])
            thresholds.append(threshold)

        # 将阈值及相应的精确率和召回率一起排序
        sorted_data = sorted(zip(thresholds, precision_list, recall_list))
        thresholds, precision_list, recall_list = zip(*sorted_data)  # 解压排序后的数据
        fig, ax1 = plt.subplots()

        # 设置第一个y轴（精确率）
        color = 'tab:red'
        ax1.set_xlabel('阈值')
        ax1.set_ylabel('精确率', color=color)
        ax1.plot(thresholds, precision_list, color=color, label='精确率', marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.invert_yaxis()  # 翻转Y轴

        # 创建一个共享x轴的第二个y轴（召回率）
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('召回率', color=color)
        ax2.plot(thresholds, recall_list, color=color, label='召回率', marker='x')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('精确率与召回率随阈值变化的双y轴图')
        fig.tight_layout()

        if save_path:
            plt.savefig(os.path.join(Category_save_path, 'Precision_Recall_Curve_Dual_Y.png'), dpi=600)
        else:
            plt.show()

        plt.close()

    elif plot_type == 'sensitivity_specificity':
        # 绘制敏感性与特异性的图表
        thresholds, sensitivity, specificity = [], [], []

        for threshold, metrics in metrics_dict.items():
            thresholds.append(threshold)
            sensitivity.append(metrics['recall'])  # 敏感性通常等同于召回率
            specificity.append(metrics['specificity'])

        fig, ax1 = plt.subplots()

        # 敏感性曲线
        color = 'tab:red'
        ax1.set_xlabel('阈值')
        ax1.set_ylabel('敏感性', color=color)
        ax1.plot(thresholds, sensitivity, color=color, label='敏感性', marker='o')
        ax1.tick_params(axis='y', labelcolor=color)

        # 特异性曲线
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('特异性', color=color)
        ax2.plot(thresholds, specificity, color=color, label='特异性', marker='x')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.invert_yaxis()  # 翻转Y轴
        plt.title('敏感性与特异性随阈值变化的双y轴图')

        fig.tight_layout()

        if save_path:
            plt.savefig(os.path.join(Category_save_path, 'Sensitivity_Specificity_Curve.png'), dpi=600)
        else:
            plt.show()

        plt.close()

    else:
        raise ValueError("未知的plot_type. 选择 'confusion', 'pr', 或 'sensitivity_specificity'.")


def evaluate_model(model, query_pairs, query_labels, threshold=None, save_plots=False, save_dir=None):
    """
    使用查询集评估模型。
    Args:
        model (tf.keras.Model): 已微调的孪生网络模型。
        query_pairs (np.array): 查询集样本对。
        query_labels (np.array): 查询集样本对的标签。
        threshold (list or float or None): 用于将相似度得分转换为二分类输出的阈值。
        save_plots (bool): 是否保存图表。
        save_dir (str): 保存图表的目录。
    Returns:
        dict: 包含模型性能指标的字典。
    """

    # 预测查询集样本对的相似度

    query_data = [query_pairs[:, 0], query_pairs[:, 1]]
    predictions = model.predict(query_data).flatten()  # 得到样本对的距离

    detailed_metrics = {}  # 初始化指标字典，存储每个阈值下的多个性能指标，便于后续的访问和显示。

    # 根据提供的阈值类型决定如何处理阈值
    if threshold is None:
        thresholds = np.arange(0.0, 1.0, 0.05)  # 默认阈值范围
    elif isinstance(threshold, (list, np.ndarray)):
        thresholds = threshold  # 直接使用提供的列表或numpy数组
    elif isinstance(threshold, (int, float)):
        thresholds = [threshold]  # 单个阈值转换为列表
    else:
        raise ValueError("Threshold must be a float, a list, a numpy array of floats, or None.")

    # 记录二分类输出
    all_predictions_binary = []

    # 遍历阈值并计算性能指标
    for _ in thresholds:
        predictions_binary = np.where(predictions > _, 0, 1)  # 将预测的相似度转换成二分类输出，超出阈值输出为0，否则输出为1。
        all_predictions_binary.append(predictions_binary)

        print("----------------------------------------------------")
        print(f"真实样本对分类标签：{query_labels}，相同类别输出1，不同类别输出0")
        print(f"当阈值为:{_},预测样本对分类标签：{predictions_binary}")
        print("----------------------------------------------------")

        # 计算性能指标并格式化输出
        accuracy = accuracy_score(query_labels, predictions_binary)
        precision = precision_score(query_labels, predictions_binary, zero_division=1)
        recall = recall_score(query_labels, predictions_binary, zero_division=1)
        f1 = f1_score(query_labels, predictions_binary, zero_division=1)
        cm = confusion_matrix(query_labels, predictions_binary)

        # 特异性计算 (Specificity)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)

        detailed_metrics[_] = {
            "accuracy": f"{accuracy:.4f}",
            "precision": f"{precision:.4f}",
            "recall": f"{recall:.4f}",
            "specificity": f"{specificity:.4f}",
            "f1_score": f"{f1:.4f}",
            "confusion_matrix": cm.tolist(),  # 将numpy数组转换为列表以便更好地序列化或打印
        }

    # 绘制图表
    if save_plots:
        # 自动更新主目录编号
        existing_folders = [d for d in os.listdir(save_dir) if d.startswith('Indicator_visualization_')
                            and d[24:].isdigit()]
        highest_num = max([int(folder.split('Indicator_visualization_')[-1]) for folder in existing_folders] or [-1])
        new_folder = f"Indicator_visualization_{highest_num + 1}"
        visualization_path = os.path.join(save_dir, new_folder)
        os.makedirs(visualization_path)

        # plot_metrics(detailed_metrics, 'confusion', save_path=visualization_path)
        plot_metrics(detailed_metrics, 'pr', save_path=visualization_path)
        plot_metrics(detailed_metrics, 'sensitivity_specificity', save_path=visualization_path)

    return detailed_metrics


def save_results_to_file(result, save_folder, file_name="validation_results.txt"):
    """
    将所有周期的评估结果和类别详细信息保存到一个文本文件中。
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

# # 自动创建编号的子目录
# existing_folders = [d for d in os.listdir(save_folder) if d.startswith('validation_results_') and d[19:].isdigit()]
# highest_num = max([int(folder.split('validation_results_')[-1]) for folder in existing_folders] or [-1])
# new_folder = f"validation_results_{highest_num + 1}"

    new_folder = f"validation_results"
    subfolder_path = os.path.join(save_folder, new_folder)
    os.makedirs(subfolder_path)

    file_path = os.path.join(subfolder_path, file_name)

    with open(file_path, 'w', encoding='utf-8') as file:
        for cycle_data in result:
            file.write(f"Cycle {cycle_data['cycle']} Results:\n")
            file.write("Category Details:\n")
            for detail, detail_dx in zip(cycle_data['category_details'], cycle_data['category_dx_details']):
                dx = detail['DX']
                chinese_name = detail['Chinese_Name']
                # samples = detail_dx['总样本数']
                chosen_support = detail_dx['选中的支持集样本']
                chosen_query = detail_dx['选中的查询集样本']
                file.write(f"DX Code: {dx['类别_DX']}\t中文疾病名称: {chinese_name}\t"
                           f"支持集: {len(chosen_support)}对\t查询集: {len(chosen_query)}对\n")
            file.write("Metrics:\n")
            for threshold, metrics in cycle_data['metrics'].items():
                file.write(f"Threshold {threshold:.4f}: {metrics}\n")
            file.write("\n")  # 在每个周期之间添加一个空行以提高可读性

    print(f"All evaluation data saved to {file_path}")


def evaluate_and_record(Config):
    """
    微调模型并评估其性能，将结果记录到文件中。接受配置字典作为参数。
    """
    # 加载模型
    model = load_custom_model(Config['model_path'], Config['initial_margin'])

    # 初始化指标列表
    all_cycles_metrics = []

    # 循环评估
    for _ in range(Config['cycle_index']):
        print(f"开始周期 {_ + 1} 的数据准备和模型评估...")

        # 准备数据集
        dataset, category_dx_details = prepare_datasets(Config['csv_data_path'], Config['h5_data_folder'], Config['N'],
                                                        Config['K'],
                                                        Config['Q'])

        # 根据DX编号获取中文名称
        category_details = [{'DX': dx, 'Chinese_Name': dx_translation.get(str(dx['类别_DX']), "未知疾病")}
                            for dx in category_dx_details]

        # 微调模型
        fine_tune_model(model, dataset, Config['margin'], Config['learning_rate'], Config['train_epochs'])

        # 准备查询集数据
        query_pairs = np.concatenate((dataset['query_positive_pairs'], dataset['query_negative_pairs']), axis=0)
        query_labels = np.array([1] * len(dataset['query_positive_pairs']) + [0] * len(dataset['query_negative_pairs']))

        # 评估模型
        metrics = evaluate_model(model, query_pairs, query_labels,
                                 Config['th'], Config['save_plot'], Config['save_folder'])

        print(f"周期 {_ + 1} 评估完成...")
        print("----------------------------------------------------")

        # 聚合每个周期的结果和相关信息
        cycle_data = {
            "cycle": _ + 1,
            "metrics": metrics,
            "category_dx_details": category_dx_details,
            "category_details": category_details
        }
        all_cycles_metrics.append(cycle_data)

    # 保存所有评估结果到文件
    save_results_to_file(all_cycles_metrics, Config['save_folder'], Config['file_name'])

    print("已成功记录所有评估")


# 初始化项目路径
BASE_PATH = "D:\\PyCharm 2022.1\\PyCharm Documents\\Siamese Neural Networks for ECG-signal"
DATA_PATH = os.path.join(BASE_PATH, "Data")
Saved_DIR = os.path.join(BASE_PATH, "Saved")
MODEL_SAVE_DIR = os.path.join(Saved_DIR, "Saved_Models")
Result_SAVE_DIR = os.path.join(Saved_DIR, "Saved_Val_result")

# 路径设置
dx_table_path = os.path.join(DATA_PATH, "心电信号DX对照表.csv")
dx_translation = load_dx_translation_table(dx_table_path)


def setup_save_folder(base_dir):
    """
    检查给定的目录下的文件夹，并按照 'Val_n' 的模式创建新文件夹，其中 n 自动递增。
    返回新创建的文件夹的路径。

    Args:
        base_dir (str): 基础目录，用于检查和创建新的文件夹。

    Returns:
        str: 新创建的文件夹的路径。
    """
    # 确保基础目录存在
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 检索现有文件夹
    existing_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    val_folders = [folder for folder in existing_folders if folder.startswith('Val_')]

    # 确定新文件夹的编号
    if val_folders:
        highest_num = max([int(folder.split('_')[-1]) for folder in val_folders])
        new_num = highest_num + 1
    else:
        new_num = 0  # 如果没有现有文件夹，从0开始

    # 创建新文件夹
    new_folder_name = f"Val_{new_num}"
    new_folder_path = os.path.join(base_dir, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)  # 使用exist_ok避免在并发情况下的错误

    return new_folder_path


# 每次验证新建独立文件夹路径
new_save_dir = setup_save_folder(Result_SAVE_DIR)

# VAL_H5_DIR = os.path.join(DATA_PATH, "SPH_val")
# VAL_CSV = os.path.join(DATA_PATH, "sph_cinc_val.csv")
VAL_H5_DIR = os.path.join(DATA_PATH, "SPH_test")
VAL_CSV = os.path.join(DATA_PATH, "sph_cinc_test.csv")
# VAL_H5_DIR = os.path.join(DATA_PATH, "SPH_train")
# VAL_CSV = os.path.join(DATA_PATH, "sph_cinc_train.csv")
model_path = os.path.join(MODEL_SAVE_DIR, "model_5\\model_query_best.h5")
# model_path = os.path.join(MODEL_SAVE_DIR, "model_4\\model_support_best.h5")

# 配置字典
config = {
    'model_path': model_path,
    'initial_margin': initial_margin,
    'csv_data_path': VAL_CSV,
    'h5_data_folder': VAL_H5_DIR,
    'save_folder': new_save_dir,
    'file_name': "summary_metrics.json",

    'N': 5,
    'K': 5,
    'Q': 2,
    'cycle_index': 20,  # 循环评估次数
    'train_epochs': 0,  # 验证集支持集训练轮次

    # 'th': None,  # 使用默认阈值范围[0.0, 1.0, 0.05]
    # 'th': 0.3,  # 指定分类阈值
    'th': np.arange(0.0, 3.0, 0.05),  # 指定区间阈值

    'margin': initial_margin,
    'learning_rate': 0.001,

    'save_plot': True  # 是否可视化评价指标
}

evaluate_and_record(config)

