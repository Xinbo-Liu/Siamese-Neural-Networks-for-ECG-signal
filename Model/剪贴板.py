import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from Config.function import load_and_preprocess_ecg_signals, generate_n_way_k_shot, \
    create_balanced_pairs_from_support_query_sets, load_custom_model, ContrastiveLoss
from Config.hyper_parameter import initial_margin


def prepare_datasets(csv_data_path, h5_data_folder, N, K, Q):
    """
    准备支持集和查询集。
    """
    signals, _, labels = load_and_preprocess_ecg_signals(h5_data_folder, csv_data_path)
    train_df = pd.read_csv(csv_data_path)
    unique_labels = train_df['Dx'].unique()
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}

    support_set, query_set = generate_n_way_k_shot(signals, labels, N, K, label_to_index, index_to_label, Q)
    pairs, pair_labels = create_balanced_pairs_from_support_query_sets(support_set, query_set)

    return pairs, pair_labels


def fine_tune_model(model, support_pairs, support_labels, Margin, Learning_rate=0.001, epochs=5):
    """
    使用支持集对模型进行预学习微调。
    Args:
        model (tf.keras.Model): 要微调的孪生网络模型。
        support_pairs (np.array): 支持集样本对。
        support_labels (np.array): 支持集样本对的标签。
        Margin: 对比损失的间距参数.
        Learning_rate (float): 微调过程中的学习率。
        epochs (int): 微调的周期数。
    """

    model = load_custom_model(model_path, initial_margin)
    model.compile(optimizer=Adam(Learning_rate), loss=ContrastiveLoss(Margin), metrics=['accuracy'])

    # 训练模型
    support_data = [support_pairs[:, 0], support_pairs[:, 1]]
    model.fit(support_data, support_labels, epochs=epochs, verbose=1)


def evaluate_model(model, query_pairs, query_labels):
    """
    使用查询集评估模型。
    Args:
        model (tf.keras.Model): 已微调的孪生网络模型。
        query_pairs (np.array): 查询集样本对。
        query_labels (np.array): 查询集样本对的标签。
    Returns:
        dict: 包含模型性能指标的字典。
    """
    query_data = [query_pairs[:, 0], query_pairs[:, 1]]
    predictions = model.predict(query_data).flatten()
    predictions_binary = np.where(predictions > 0.5, 1, 0)  # 使用0.5作为默认阈值

    # 计算性能指标
    accuracy = accuracy_score(query_labels, predictions_binary)
    precision = precision_score(query_labels, predictions_binary)
    recall = recall_score(query_labels, predictions_binary)
    f1 = f1_score(query_labels, predictions_binary)
    roc_auc = roc_auc_score(query_labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


def evaluate_and_record(model, support_pairs, support_labels, query_pairs, query_labels, save_folder, file_name,
                        learning_rate=1e-4, epochs=5):
    """
    微调模型并评估其性能，将结果记录到文件中。
    """
    # 微调模型
    fine_tune_model(model, support_pairs, support_labels, learning_rate, epochs)

    # 评估模型
    metrics = evaluate_model(model, query_pairs, query_labels)

    # 保存结果到文件
    save_results_to_file(metrics, save_folder, file_name)

    print("Evaluation and recording complete. Metrics:", metrics)


def save_results_to_file(results, save_folder, file_name):
    """
    将结果保存到文件。
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    file_path = os.path.join(save_folder, file_name)

    with open(file_path, 'w') as file:
        for threshold, result in results.items():
            file.write(f"Threshold {threshold:.2f}: {result}\n")
        if 'roc_auc' in results:
            file.write(f"ROC AUC: {results['roc_auc']}\n")

    print(f"Validation results saved to {file_path}")


# 使用示例
model_path = 'path_to_your_model.h5'
h5_data_folder = 'path_to_h5_folder'
csv_data_path = 'path_to_csv'
save_folder = 'path_to_save_folder'
N, K, Q = 5, 5, 1

model = load_custom_model(model_path, initial_margin)
pairs, labels = prepare_datasets(csv_data_path, h5_data_folder, N, K, Q)
evaluate_and_record(model, pairs, labels, save_folder, "validation_results.txt")
