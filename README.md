# 项目名称：Siamese Neural Networks for ECG-signal

## CUDA 和 cuDNN
本项目需要特定版本的 CUDA 和 cuDNN：

- CUDA: 11.2.0
- cuDNN: 8.1.0.77

请确保这些版本已安装并正确配置在您的系统中。

## 依赖项

### Python 版本
python-3.9.6-amd64

### Python 包
所有必需的 Python 包都列在 `requirements.txt` 文件中。您可以使用以下命令安装它们：

```sh
pip install -r requirements.txt
```

## 项目结构和内容

### .idea
- 项目文件夹，包含IDE相关配置和项目设置。

### Check
- 用于文档与库版本兼容性检查的脚本和配置文件（与项目本身无关）。

### Config
- 项目配置文件夹，存放超参数配置、函数库等相关设置。

### Data Classification
- 数据划分与预览加载模块，用于数据的初始处理和预览（与后续数据集划分无关）。

### Data
- 数据文件夹，存储心电图信号数据、csv格式的数据集和DX对照表等相关数据文件。

### Model
- 神经网络模型文件夹，包含网络模型的构建、训练、测试和验证相关代码。

### Saved
- 用于保存训练结果的文件夹，可由程序自动创建。


