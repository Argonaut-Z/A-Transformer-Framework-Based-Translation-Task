# 项目说明

## A Transformer Framework Based Translation Task

- 一个**基于Transformer Encoder-Decoder网络架构**的文本分类模型

- 论文[Attention Is All You Need](https://arxiv.org/abs/1706.03762) 基于PyTorch的实现

## 1. 环境准备

在conda环境下创建虚拟环境并安装如下必要库：

```shell
conda create -n transformer_translate python=3.10 -y
conda activate transformer_translate

cd translate/
pip install torch==2.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install torchtext==0.17.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install -r requirementes.txt
```

requirements.txt 内容如下：

```txt
torch==2.2.0
torchtext==0.17.0
numpy==1.26.4
```

也可以直接使用命令：

```shell
pip install torch==2.2.0 torchtext==0.17.0 numpy==1.26.4
```

安装`de_core_news_sm-3.0.0.tar.gz`和`en_core_web_sm-3.0.0.tar.gz`

在本地或者云平台上对于下载好的两个文件，在终端执行如下命令进行安装：

- 云平台：魔塔社区

```shell
pip install /mnt/workspace/translate/data/de_core_news_sm-3.0.0.tar.gz
pip install /mnt/workspace/translate/data/en_core_web_sm-3.0.0.tar.gz
```

- 本地

```shell
cd /translate/data
pip install de_core_news_sm-3.0.0.tar.gz
pip install en_core_web_sm-3.0.0.tar.gz
```

## 2. 数据

数据已下载完毕，保存在该项目根目录下的data文件夹中

本项目是一个基于 Transformer 的文本翻译任务，主要使用了双语平行语料作为训练和测试数据。以下是数据的详细信息：

------

#### **数据来源**

1. **数据集文件**： 数据存放在项目目录的 `data/` 文件夹中，包括以下文件：
   - **`train.de`**：用于训练的德语语料。
   - **`train.en`**：用于训练的英语语料，与 `train.de` 一一对应。
   - **`val.de`**：用于验证的德语语料。
   - **`val.en`**：用于验证的英语语料，与 `val.de` 一一对应。
   - **`test_2016_flickr.de`**：用于测试的德语语料。
   - **`test_2016_flickr.en`**：用于测试的英语语料，与 `test_2016_flickr.de` 一一对应。
2. **预训练分词模型**：
   - **`de_core_news_sm-3.0.0.tar.gz`**：小型德语分词模型（Spacy 格式）。
   - **`en_core_web_sm-3.0.0.tar.gz`**：小型英语分词模型（Spacy 格式）。
   - 这些分词模型用于对原始语料进行分词和预处理。

------

#### **数据描述**

1. **语料类型**：

   - 该数据集是德语到英语的平行语料，用于构建文本翻译模型。
   - 数据集包含三个部分：训练集、验证集和测试集。

2. **文件格式**：

   - 所有数据均为纯文本格式，每一行代表一句话。

   - 每个文件中的句子按照顺序一一对应。例如，`train.de` 第 1 行与 `train.en` 第 1 行是平行的翻译句对。

   - `train.de`

     ```txt
     Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.
     Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.
     Ein kleines Mädchen klettert in ein Spielhaus aus Holz.
     Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.
     ...
     ```

   - `train.en`

     ```txt
     Two young, White males are outside near many bushes.
     Several men in hard hats are operating a giant pulley system.
     A little girl climbing into a wooden playhouse.
     A man in a blue shirt is standing on a ladder cleaning a window.
     ...
     ```

3. **数据分布**：

   - 训练集：
     - 文件：`train.de` 和 `train.en`
     - 用于训练模型的句子对。
   - 验证集：
     - 文件：`val.de` 和 `val.en`
     - 用于模型训练过程中监控性能的句子对。
   - 测试集：
     - 文件：`test_2016_flickr.de` 和 `test_2016_flickr.en`
     - 用于评估模型性能的句子对。

4. **预处理需求**：

   - 原始数据需要经过分词、编码、添加特殊标记（如 `<bos>` 和 `<eos>`）等步骤，转化为模型可用的输入格式。

------

#### **数据的作用**

1. **训练集**：
   - 提供德语-英语的平行句对，用于学习语言映射关系。
2. **验证集**：
   - 通过定期在验证集上计算损失和翻译性能，监控模型的过拟合和泛化能力。
3. **测试集**：
   - 最终测试模型的翻译效果，评估 BLEU 分数等指标。
4. **分词模型**：
   - 使用 `de_core_news_sm` 和 `en_core_web_sm` 提供的分词工具，提高数据预处理的效率和质量。

------

#### **总结**

- 该数据集由训练集、验证集和测试集组成，涵盖德语到英语的平行翻译任务。
- 数据文件采用纯文本格式，每行是一对平行句子。
- Spacy 的分词模型是预处理的重要工具，能够帮助规范化输入并提高模型性能。 

## 3. 项目目录结构

以下是`translate`项目下目录结构的详细说明，包括每个文件和文件夹的作用：

------

```css
translate/
├── cache/                       # 缓存文件夹
│   ├── log_train_2024-12-19.txt # 训练日志文件，记录训练过程中的重要信息
│   ├── log_train_2024-12-24.txt # 另一个训练日志文件，记录不同时间的训练情况
│   ├── model.pkl                # 保存的模型参数（PyTorch 格式），可用于推理或继续训练
│
├── config/                      # 配置文件夹
│   ├── __pycache__/             # Python 缓存文件夹（自动生成，存储已编译的`.pyc`文件）
│   ├── __init__.py              # 标识 `config` 为 Python 模块
│   ├── config.py                # 项目配置文件，包含超参数和路径设置
│
├── data/                        # 数据文件夹
│   ├── de_core_news_sm-3.0.0.tar.gz  # 德语分词模型（Spacy 格式）
│   ├── en_core_web_sm-3.0.0.tar.gz   # 英语分词模型（Spacy 格式）
│   ├── test_2016_flickr.de      # 测试集德语文件
│   ├── test_2016_flickr.en      # 测试集英语文件
│   ├── train.de                 # 训练集德语文件
│   ├── train.en                 # 训练集英语文件
│   ├── val.de                   # 验证集德语文件
│   ├── val.en                   # 验证集英语文件
│
├── model/                       # 模型定义文件夹
│   ├── __pycache__/             # Python 缓存文件夹
│   ├── __init__.py              # 标识 `model` 为 Python 模块
│   ├── CustomScheduleLearningRate.py  # 自定义学习率调度器
│   ├── Embedding.py             # 嵌入层模块，包含词嵌入、位置嵌入等
│   ├── MyTransformer.py         # Transformer 模型的核心组件
│   ├── TranslationModel.py      # 翻译模型的完整实现
│
├── test/                        # 测试相关代码文件夹
│   ├── __init__.py              # 标识 `test` 为 Python 模块
│   ├── test_Config.py           # 测试配置模块
│   ├── test_Embedding.py        # 测试嵌入层模块
│   ├── test_LoadEnglishGermanDataset.py # 测试数据加载功能
│   ├── test_MyTransformer.py    # 测试 Transformer 模型组件
│   ├── test_TranslationModel.py # 测试翻译模型
│
├── utils/                       # 工具模块文件夹
│   ├── __pycache__/             # Python 缓存文件夹
│   ├── __init__.py              # 标识 `utils` 为 Python 模块
│   ├── data_helpers.py          # 数据预处理和加载工具
│   ├── log_helper.py            # 日志工具模块
│
├── 文本翻译项目说明.md             # 中文项目说明文档
├── README.md                    # 项目主文档，介绍项目背景和使用方法
├── requirements.txt             # 项目依赖文件，包含所需 Python 包及版本
├── train.py                     # 训练脚本，用于执行模型训练逻辑
├── translate.ipynb              # Jupyter Notebook 文件，记录整个翻译任务的流程
├── translate.py                 # 翻译脚本，用于加载模型并进行推理
```

## 4. 详细内容介绍

以下按照 **项目代码**、**数据部分**、**结果与缓存**、**依赖文件**、**项目说明文档** 和 **自动生成的文件** 六大模块进行分类说明。

------

### 4.1 项目代码

**功能**：存储与模型训练、测试和推理相关的所有代码。

**目录和文件**

```css
translate/
├── train.py                     # 训练脚本，用于模型训练
├── translate.py                 # 翻译脚本，用于推理
├── translate.ipynb              # Jupyter Notebook，记录项目流程
├── model/                       # 模型定义文件夹
│   ├── __init__.py              # 标识 `model` 为 Python 模块
│   ├── CustomScheduleLearningRate.py  # 自定义学习率调度器
│   ├── Embedding.py             # 嵌入层模块，包括词嵌入和位置嵌入
│   ├── MyTransformer.py         # Transformer 模型核心组件
│   ├── TranslationModel.py      # 翻译模型的完整实现
├── utils/                       # 工具模块文件夹
│   ├── __init__.py              # 标识 `utils` 为 Python 模块
│   ├── data_helpers.py          # 数据加载和预处理工具
│   ├── log_helper.py            # 日志管理工具
├── test/                        # 测试代码文件夹
│   ├── __init__.py              # 标识 `test` 为 Python 模块
│   ├── test_Config.py           # 测试配置模块
│   ├── test_Embedding.py        # 测试嵌入层模块
│   ├── test_LoadEnglishGermanDataset.py # 测试数据加载功能
│   ├── test_MyTransformer.py    # 测试 Transformer 模型组件
│   ├── test_TranslationModel.py # 测试翻译模型
```

**主要内容**

1. **训练相关**：
   - **`train.py`**：主训练脚本，整合数据加载、模型初始化和训练逻辑。
   - **`translate.ipynb`**：Notebook 文件，直观记录从数据加载到模型训练的完整过程。
2. **推理相关**：
   - **`translate.py`**：加载训练好的模型并实现推理。
3. **模型定义**：
   - **`Embedding.py`**：定义嵌入层（包括词嵌入和位置嵌入）。
   - **`MyTransformer.py`**：实现 Transformer 的核心组件（如多头注意力）。
   - **`TranslationModel.py`**：构建基于 Transformer 的翻译模型。
4. **工具模块**：
   - **`data_helpers.py`**：数据加载和预处理（如分词、构造词表和数据封装）。
   - **`log_helper.py`**：日志管理。
5. **测试代码**：
   - 每个子模块对应一个测试文件（如 `test_Embedding.py` 测试嵌入层）。

------

### 4.2 数据部分

**功能**：存储训练、验证和测试所需的德语-英语平行语料，以及预训练的分词模型。

**目录和文件**

```css
translate/
├── data/                        # 数据文件夹
│   ├── de_core_news_sm-3.0.0.tar.gz  # 德语分词模型
│   ├── en_core_web_sm-3.0.0.tar.gz   # 英语分词模型
│   ├── train.de                 # 训练集德语文件
│   ├── train.en                 # 训练集英语文件
│   ├── val.de                   # 验证集德语文件
│   ├── val.en                   # 验证集英语文件
│   ├── test_2016_flickr.de      # 测试集德语文件
│   ├── test_2016_flickr.en      # 测试集英语文件
```

**主要内容**

1. 训练数据：

   - **`train.de`** 和 **`train.en`**：训练集的德语和英语句子对。

2. 验证数据：

   - **`val.de`** 和 **`val.en`**：验证集的德语和英语句子对。

3. 测试数据：

   - **`test_2016_flickr.de`** 和 **`test_2016_flickr.en`**：测试集的德语和英语句子对。

4. 分词模型：

   `de_core_news_sm-3.0.0.tar.gz` 和 `en_core_web_sm-3.0.0.tar.gz`：

   - Spacy 格式的预训练分词模型，用于分词。

------

### 4.3 结果与缓存

**功能**：存储训练过程中生成的中间文件，包括日志和模型参数。

**目录和文件**

```css
translate/
├── cache/                       # 缓存文件夹
│   ├── log_train_2024-12-19.txt # 训练日志文件
│   ├── log_train_2024-12-24.txt # 另一个训练日志文件
│   ├── model.pkl                # 训练好的模型参数（PyTorch 格式）
```

**主要内容**

1. 训练日志：
   - 记录训练过程中的损失值、准确率等。
2. 模型参数：
   - **`model.pkl`**：保存的模型参数，可用于推理或继续训练。

------

### 4.4 依赖文件

**功能**：记录项目所需的 Python 包及其版本。

**目录和文件**

```css
translate/
├── requirements.txt             # 项目依赖文件
```

**主要内容**

- `requirements.txt`：

  - 包含所有依赖的 Python 包，如：

    ```txt
    torch==2.0.0
    spacy==3.5.0
    numpy==1.23.0
    ```

  - 可通过以下命令快速安装：

    ```shell
    pip install -r requirements.txt
    ```

------

### 4.5 项目说明文档

**功能**：存储项目背景、数据集描述、模型结构和运行步骤等说明。

**目录和文件**

```css
translate/
├── 文本翻译项目说明.md             # 中文说明文档
├── README.md                    # 项目主文档
```

**主要内容**

1. **`文本翻译项目说明.md`**：
   - 详细记录项目背景、目标、实现步骤。
   - 数据集描述、模型结构和训练流程的中文说明。
2. **`README.md`**：
   - 项目简介，包含运行指导和结果展示。

------

### 4.6 自动生成的文件

**功能**：Python 自动生成的缓存文件。

**目录和文件**

```css
translate/
├── config/
│   ├── __pycache__/             # Python 缓存文件
├── model/
│   ├── __pycache__/             # Python 缓存文件
├── utils/
│   ├── __pycache__/             # Python 缓存文件
├── test/
│   ├── __pycache__/             # Python 缓存文件
```

**主要内容**

- 缓存文件夹：

  ```css
  __pycache__/
  ```

  - Python 运行时生成的已编译 `.pyc` 文件，用于加快模块加载速度。

## 5. 使用方法

### 5.1 修改配置文件（可选）

- STEP 1.直接下载或克隆本项目
- STEP 2.可自定义修改配置文件`config.py`中的配置参数，也可以保持默认

### 5.2 命令行添加系统路径

```shell
export PYTHONPATH=/mnt/workspace/translate:$PYTHONPATH
```

### 5.3 训练

直接执行如下命令即可进行模型训练：

```shell
python train.py
```

训练过程和结果：

```shell
[2024-12-24 17:22:58] - INFO: ############载入数据集############
[2024-12-24 17:23:05] - INFO: ############划分数据集############
[2024-12-24 17:23:05] - INFO: ### 正在将数据集 ['/mnt/workspace/translate/data/train.de', '/mnt/workspace/translate/data/train.en'] 转换成 Token ID 
[2024-12-24 17:23:07] - INFO: ### 正在将数据集 ['/mnt/workspace/translate/data/val.de', '/mnt/workspace/translate/data/val.en'] 转换成 Token ID 
[2024-12-24 17:23:07] - INFO: ### 正在将数据集 ['/mnt/workspace/translate/data/test_2016_flickr.de', '/mnt/workspace/translate/data/test_2016_flickr.en'] 转换成 Token ID 
[2024-12-24 17:23:07] - INFO: ############初始化模型############
[2024-12-24 17:23:08] - INFO: #### 成功载入已有模型，进行追加训练...
[2024-12-24 17:23:11] - INFO: Epoch: 0, Batch[0/227], Train loss :2.137, Train acc: 0.5632808256382401
[2024-12-24 17:23:14] - INFO: Epoch: 0, Batch[1/227], Train loss :2.208, Train acc: 0.5523408732246187
[2024-12-24 17:23:17] - INFO: Epoch: 0, Batch[2/227], Train loss :2.140, Train acc: 0.5592493297587131
[2024-12-24 17:23:19] - INFO: Epoch: 0, Batch[3/227], Train loss :2.112, Train acc: 0.5766871165644172
[2024-12-24 17:23:21] - INFO: Epoch: 0, Batch[4/227], Train loss :2.134, Train acc: 0.5474330357142857
[2024-12-24 17:23:24] - INFO: Epoch: 0, Batch[5/227], Train loss :2.070, Train acc: 0.5859375
[2024-12-24 17:23:26] - INFO: Epoch: 0, Batch[6/227], Train loss :2.142, Train acc: 0.58603066439523
[2024-12-24 17:23:28] - INFO: Epoch: 0, Batch[7/227], Train loss :2.160, Train acc: 0.5597138139790864
[2024-12-24 17:23:30] - INFO: Epoch: 0, Batch[8/227], Train loss :2.134, Train acc: 0.5734265734265734
[2024-12-24 17:23:32] - INFO: Epoch: 0, Batch[9/227], Train loss :2.108, Train acc: 0.581858407079646
[2024-12-24 17:23:34] - INFO: Epoch: 0, Batch[10/227], Train loss :2.224, Train acc: 0.5743645213628988
[2024-12-24 17:23:36] - INFO: Epoch: 0, Batch[11/227], Train loss :2.075, Train acc: 0.5791245791245792
[2024-12-24 17:23:39] - INFO: Epoch: 0, Batch[12/227], Train loss :2.154, Train acc: 0.5774798927613941
[2024-12-24 17:23:41] - INFO: Epoch: 0, Batch[13/227], Train loss :1.995, Train acc: 0.5886563876651982
[2024-12-24 17:23:43] - INFO: Epoch: 0, Batch[14/227], Train loss :2.063, Train acc: 0.569826135726304
[2024-12-24 17:23:45] - INFO: Epoch: 0, Batch[15/227], Train loss :2.151, Train acc: 0.5582172701949861
[2024-12-24 17:23:47] - INFO: Epoch: 0, Batch[16/227], Train loss :2.046, Train acc: 0.5810234541577826
[2024-12-24 17:23:50] - INFO: Epoch: 0, Batch[17/227], Train loss :2.128, Train acc: 0.5723059743160246
[2024-12-24 17:23:52] - INFO: Epoch: 0, Batch[18/227], Train loss :2.067, Train acc: 0.5632183908045977
[2024-12-24 17:23:54] - INFO: Epoch: 0, Batch[19/227], Train loss :2.149, Train acc: 0.5660377358490566
[2024-12-24 17:23:56] - INFO: Epoch: 0, Batch[20/227], Train loss :2.173, Train acc: 0.5462091864969563
[2024-12-24 17:23:58] - INFO: Epoch: 0, Batch[21/227], Train loss :1.984, Train acc: 0.5941866964784795
[2024-12-24 17:24:00] - INFO: Epoch: 0, Batch[22/227], Train loss :2.133, Train acc: 0.5895096921322691
[2024-12-24 17:24:02] - INFO: Epoch: 0, Batch[23/227], Train loss :2.110, Train acc: 0.566027397260274
[2024-12-24 17:24:04] - INFO: Epoch: 0, Batch[24/227], Train loss :1.882, Train acc: 0.5959093421779988
[2024-12-24 17:24:07] - INFO: Epoch: 0, Batch[25/227], Train loss :2.108, Train acc: 0.5832414553472988
[2024-12-24 17:24:10] - INFO: Epoch: 0, Batch[26/227], Train loss :2.241, Train acc: 0.5616666666666666
[2024-12-24 17:24:11] - INFO: Epoch: 0, Batch[27/227], Train loss :2.152, Train acc: 0.555028962611901
[2024-12-24 17:24:14] - INFO: Epoch: 0, Batch[28/227], Train loss :2.102, Train acc: 0.5783492822966507
[2024-12-24 17:24:16] - INFO: Epoch: 0, Batch[29/227], Train loss :2.071, Train acc: 0.5895522388059702
[2024-12-24 17:24:18] - INFO: Epoch: 0, Batch[30/227], Train loss :2.060, Train acc: 0.5657392253136934
[2024-12-24 17:24:21] - INFO: Epoch: 0, Batch[31/227], Train loss :2.088, Train acc: 0.5849673202614379
[2024-12-24 17:24:24] - INFO: Epoch: 0, Batch[32/227], Train loss :1.927, Train acc: 0.5976364659538548
[2024-12-24 17:24:26] - INFO: Epoch: 0, Batch[33/227], Train loss :2.028, Train acc: 0.5731566820276498
[2024-12-24 17:24:28] - INFO: Epoch: 0, Batch[34/227], Train loss :2.019, Train acc: 0.5788058095750404
[2024-12-24 17:24:30] - INFO: Epoch: 0, Batch[35/227], Train loss :2.089, Train acc: 0.5852334419109664
[2024-12-24 17:24:32] - INFO: Epoch: 0, Batch[36/227], Train loss :2.096, Train acc: 0.5786802030456852
[2024-12-24 17:24:34] - INFO: Epoch: 0, Batch[37/227], Train loss :2.068, Train acc: 0.5717356260075228
[2024-12-24 17:24:37] - INFO: Epoch: 0, Batch[38/227], Train loss :1.980, Train acc: 0.5858528237307473
[2024-12-24 17:24:39] - INFO: Epoch: 0, Batch[39/227], Train loss :1.943, Train acc: 0.591726618705036
[2024-12-24 17:24:41] - INFO: Epoch: 0, Batch[40/227], Train loss :2.091, Train acc: 0.5954620918649696
[2024-12-24 17:24:43] - INFO: Epoch: 0, Batch[41/227], Train loss :2.047, Train acc: 0.5817204301075268
[2024-12-24 17:24:45] - INFO: Epoch: 0, Batch[42/227], Train loss :2.073, Train acc: 0.5858021240916713
[2024-12-24 17:24:47] - INFO: Epoch: 0, Batch[43/227], Train loss :2.117, Train acc: 0.5547445255474452
[2024-12-24 17:24:49] - INFO: Epoch: 0, Batch[44/227], Train loss :2.121, Train acc: 0.5733695652173914
[2024-12-24 17:24:51] - INFO: Epoch: 0, Batch[45/227], Train loss :1.999, Train acc: 0.5964611872146118
[2024-12-24 17:24:54] - INFO: Epoch: 0, Batch[46/227], Train loss :2.035, Train acc: 0.5785753126699293
[2024-12-24 17:24:56] - INFO: Epoch: 0, Batch[47/227], Train loss :2.005, Train acc: 0.5953693495038589
[2024-12-24 17:24:58] - INFO: Epoch: 0, Batch[48/227], Train loss :2.175, Train acc: 0.5484581497797357
[2024-12-24 17:25:01] - INFO: Epoch: 0, Batch[49/227], Train loss :2.000, Train acc: 0.5896436525612472
[2024-12-24 17:25:03] - INFO: Epoch: 0, Batch[50/227], Train loss :2.100, Train acc: 0.5646670335718217
[2024-12-24 17:25:04] - INFO: Epoch: 0, Batch[51/227], Train loss :1.936, Train acc: 0.6052023121387283
[2024-12-24 17:25:07] - INFO: Epoch: 0, Batch[52/227], Train loss :2.004, Train acc: 0.5830653804930332
[2024-12-24 17:25:10] - INFO: Epoch: 0, Batch[53/227], Train loss :1.976, Train acc: 0.609715242881072
[2024-12-24 17:25:13] - INFO: Epoch: 0, Batch[54/227], Train loss :1.950, Train acc: 0.5898520084566596
[2024-12-24 17:25:15] - INFO: Epoch: 0, Batch[55/227], Train loss :2.187, Train acc: 0.5657051282051282
[2024-12-24 17:25:17] - INFO: Epoch: 0, Batch[56/227], Train loss :2.071, Train acc: 0.5678251121076233
[2024-12-24 17:25:19] - INFO: Epoch: 0, Batch[57/227], Train loss :2.116, Train acc: 0.5749185667752443
[2024-12-24 17:25:22] - INFO: Epoch: 0, Batch[58/227], Train loss :2.236, Train acc: 0.5481960150780829
[2024-12-24 17:25:24] - INFO: Epoch: 0, Batch[59/227], Train loss :1.989, Train acc: 0.5911845730027548
[2024-12-24 17:25:27] - INFO: Epoch: 0, Batch[60/227], Train loss :2.098, Train acc: 0.5597620335316387
[2024-12-24 17:25:29] - INFO: Epoch: 0, Batch[61/227], Train loss :2.071, Train acc: 0.5824175824175825
[2024-12-24 17:25:31] - INFO: Epoch: 0, Batch[62/227], Train loss :1.951, Train acc: 0.5970737197523917
[2024-12-24 17:25:32] - INFO: Epoch: 0, Batch[63/227], Train loss :1.892, Train acc: 0.6148923792902851
[2024-12-24 17:25:34] - INFO: Epoch: 0, Batch[64/227], Train loss :2.155, Train acc: 0.5522620904836193
[2024-12-24 17:25:36] - INFO: Epoch: 0, Batch[65/227], Train loss :1.981, Train acc: 0.5815642458100558
[2024-12-24 17:25:39] - INFO: Epoch: 0, Batch[66/227], Train loss :2.121, Train acc: 0.5714285714285714
[2024-12-24 17:25:41] - INFO: Epoch: 0, Batch[67/227], Train loss :1.938, Train acc: 0.6021680216802168
[2024-12-24 17:25:43] - INFO: Epoch: 0, Batch[68/227], Train loss :1.985, Train acc: 0.5898845519516218
[2024-12-24 17:25:46] - INFO: Epoch: 0, Batch[69/227], Train loss :2.095, Train acc: 0.575973669775096
[2024-12-24 17:25:48] - INFO: Epoch: 0, Batch[70/227], Train loss :1.935, Train acc: 0.6110800223838836
[2024-12-24 17:25:51] - INFO: Epoch: 0, Batch[71/227], Train loss :2.223, Train acc: 0.5498607242339832
[2024-12-24 17:25:53] - INFO: Epoch: 0, Batch[72/227], Train loss :2.045, Train acc: 0.5867490174059518
[2024-12-24 17:25:56] - INFO: Epoch: 0, Batch[73/227], Train loss :2.058, Train acc: 0.581148912437256
[2024-12-24 17:25:58] - INFO: Epoch: 0, Batch[74/227], Train loss :1.987, Train acc: 0.5905680600214362
[2024-12-24 17:26:00] - INFO: Epoch: 0, Batch[75/227], Train loss :1.971, Train acc: 0.6144114411441144
[2024-12-24 17:26:03] - INFO: Epoch: 0, Batch[76/227], Train loss :1.984, Train acc: 0.6009667024704619
[2024-12-24 17:26:05] - INFO: Epoch: 0, Batch[77/227], Train loss :1.980, Train acc: 0.5988764044943821
[2024-12-24 17:26:07] - INFO: Epoch: 0, Batch[78/227], Train loss :2.024, Train acc: 0.5955431754874652
[2024-12-24 17:26:10] - INFO: Epoch: 0, Batch[79/227], Train loss :2.098, Train acc: 0.5709745762711864
[2024-12-24 17:26:11] - INFO: Epoch: 0, Batch[80/227], Train loss :2.096, Train acc: 0.5746606334841629
[2024-12-24 17:26:14] - INFO: Epoch: 0, Batch[81/227], Train loss :2.078, Train acc: 0.5715823466092572
[2024-12-24 17:26:16] - INFO: Epoch: 0, Batch[82/227], Train loss :1.996, Train acc: 0.6090440755580996
[2024-12-24 17:26:18] - INFO: Epoch: 0, Batch[83/227], Train loss :1.977, Train acc: 0.6075027995520716
[2024-12-24 17:26:21] - INFO: Epoch: 0, Batch[84/227], Train loss :1.945, Train acc: 0.6121212121212121
[2024-12-24 17:26:23] - INFO: Epoch: 0, Batch[85/227], Train loss :2.022, Train acc: 0.5965008201202843
[2024-12-24 17:26:24] - INFO: Epoch: 0, Batch[86/227], Train loss :2.001, Train acc: 0.5889328063241107
[2024-12-24 17:26:26] - INFO: Epoch: 0, Batch[87/227], Train loss :1.842, Train acc: 0.617461229178633
[2024-12-24 17:26:28] - INFO: Epoch: 0, Batch[88/227], Train loss :1.930, Train acc: 0.6116343490304709
[2024-12-24 17:26:30] - INFO: Epoch: 0, Batch[89/227], Train loss :1.953, Train acc: 0.6093489148580968
[2024-12-24 17:26:32] - INFO: Epoch: 0, Batch[90/227], Train loss :1.923, Train acc: 0.6103312745648513
[2024-12-24 17:26:34] - INFO: Epoch: 0, Batch[91/227], Train loss :2.093, Train acc: 0.5795711060948081
[2024-12-24 17:26:37] - INFO: Epoch: 0, Batch[92/227], Train loss :2.123, Train acc: 0.5734806629834254
[2024-12-24 17:26:39] - INFO: Epoch: 0, Batch[93/227], Train loss :2.038, Train acc: 0.5826468973091707
[2024-12-24 17:26:41] - INFO: Epoch: 0, Batch[94/227], Train loss :1.984, Train acc: 0.5909090909090909
[2024-12-24 17:26:43] - INFO: Epoch: 0, Batch[95/227], Train loss :2.047, Train acc: 0.5789757412398921
[2024-12-24 17:26:45] - INFO: Epoch: 0, Batch[96/227], Train loss :2.129, Train acc: 0.5799776286353467
[2024-12-24 17:26:47] - INFO: Epoch: 0, Batch[97/227], Train loss :2.017, Train acc: 0.5762245459548707
[2024-12-24 17:26:50] - INFO: Epoch: 0, Batch[98/227], Train loss :1.942, Train acc: 0.6131593874078276
[2024-12-24 17:26:53] - INFO: Epoch: 0, Batch[99/227], Train loss :2.103, Train acc: 0.5694896851248643
[2024-12-24 17:26:56] - INFO: Epoch: 0, Batch[100/227], Train loss :2.170, Train acc: 0.5694006309148265
[2024-12-24 17:26:58] - INFO: Epoch: 0, Batch[101/227], Train loss :2.032, Train acc: 0.5812964930924548
[2024-12-24 17:27:00] - INFO: Epoch: 0, Batch[102/227], Train loss :2.020, Train acc: 0.5846069868995634
[2024-12-24 17:27:03] - INFO: Epoch: 0, Batch[103/227], Train loss :2.000, Train acc: 0.5915343915343916
[2024-12-24 17:27:05] - INFO: Epoch: 0, Batch[104/227], Train loss :2.119, Train acc: 0.567476002258611
[2024-12-24 17:27:07] - INFO: Epoch: 0, Batch[105/227], Train loss :2.000, Train acc: 0.5968018275271274
[2024-12-24 17:27:10] - INFO: Epoch: 0, Batch[106/227], Train loss :2.037, Train acc: 0.5781837721655024
[2024-12-24 17:27:12] - INFO: Epoch: 0, Batch[107/227], Train loss :2.101, Train acc: 0.5815991237677984
[2024-12-24 17:27:14] - INFO: Epoch: 0, Batch[108/227], Train loss :1.959, Train acc: 0.59278059785674
[2024-12-24 17:27:16] - INFO: Epoch: 0, Batch[109/227], Train loss :1.835, Train acc: 0.6235431235431236
[2024-12-24 17:27:19] - INFO: Epoch: 0, Batch[110/227], Train loss :1.992, Train acc: 0.5846905537459284
[2024-12-24 17:27:21] - INFO: Epoch: 0, Batch[111/227], Train loss :1.889, Train acc: 0.6107683679192373
[2024-12-24 17:27:23] - INFO: Epoch: 0, Batch[112/227], Train loss :1.981, Train acc: 0.6044198895027625
[2024-12-24 17:27:25] - INFO: Epoch: 0, Batch[113/227], Train loss :2.058, Train acc: 0.5901287553648069
[2024-12-24 17:27:27] - INFO: Epoch: 0, Batch[114/227], Train loss :1.940, Train acc: 0.5941727367325702
[2024-12-24 17:27:29] - INFO: Epoch: 0, Batch[115/227], Train loss :2.008, Train acc: 0.6045104510451045
[2024-12-24 17:27:33] - INFO: Epoch: 0, Batch[116/227], Train loss :2.117, Train acc: 0.5699339207048458
[2024-12-24 17:27:35] - INFO: Epoch: 0, Batch[117/227], Train loss :1.989, Train acc: 0.6095551894563427
[2024-12-24 17:27:37] - INFO: Epoch: 0, Batch[118/227], Train loss :1.894, Train acc: 0.6292798110979929
[2024-12-24 17:27:39] - INFO: Epoch: 0, Batch[119/227], Train loss :1.886, Train acc: 0.6005714285714285
[2024-12-24 17:27:41] - INFO: Epoch: 0, Batch[120/227], Train loss :2.074, Train acc: 0.5877984084880636
[2024-12-24 17:27:43] - INFO: Epoch: 0, Batch[121/227], Train loss :1.943, Train acc: 0.6041072447233314
[2024-12-24 17:27:44] - INFO: Epoch: 0, Batch[122/227], Train loss :2.027, Train acc: 0.5915721231766613
[2024-12-24 17:27:47] - INFO: Epoch: 0, Batch[123/227], Train loss :2.056, Train acc: 0.6004645760743321
[2024-12-24 17:27:49] - INFO: Epoch: 0, Batch[124/227], Train loss :1.950, Train acc: 0.6016666666666667
[2024-12-24 17:27:52] - INFO: Epoch: 0, Batch[125/227], Train loss :1.860, Train acc: 0.6038781163434903
[2024-12-24 17:27:54] - INFO: Epoch: 0, Batch[126/227], Train loss :1.941, Train acc: 0.6157354618015963
[2024-12-24 17:27:56] - INFO: Epoch: 0, Batch[127/227], Train loss :1.915, Train acc: 0.6011111111111112
[2024-12-24 17:27:58] - INFO: Epoch: 0, Batch[128/227], Train loss :1.996, Train acc: 0.6036892118501956
[2024-12-24 17:28:00] - INFO: Epoch: 0, Batch[129/227], Train loss :1.930, Train acc: 0.5966913861950941
[2024-12-24 17:28:02] - INFO: Epoch: 0, Batch[130/227], Train loss :1.974, Train acc: 0.6094091903719913
[2024-12-24 17:28:04] - INFO: Epoch: 0, Batch[131/227], Train loss :2.006, Train acc: 0.5901455767077267
[2024-12-24 17:28:07] - INFO: Epoch: 0, Batch[132/227], Train loss :1.939, Train acc: 0.5968184311574328
[2024-12-24 17:28:09] - INFO: Epoch: 0, Batch[133/227], Train loss :1.965, Train acc: 0.5984892504357932
[2024-12-24 17:28:11] - INFO: Epoch: 0, Batch[134/227], Train loss :1.993, Train acc: 0.6066591422121896
[2024-12-24 17:28:13] - INFO: Epoch: 0, Batch[135/227], Train loss :2.094, Train acc: 0.5801011804384486
[2024-12-24 17:28:15] - INFO: Epoch: 0, Batch[136/227], Train loss :1.779, Train acc: 0.6315192743764172
[2024-12-24 17:28:17] - INFO: Epoch: 0, Batch[137/227], Train loss :2.071, Train acc: 0.5820977484898407
[2024-12-24 17:28:20] - INFO: Epoch: 0, Batch[138/227], Train loss :1.980, Train acc: 0.5959540732640787
[2024-12-24 17:28:23] - INFO: Epoch: 0, Batch[139/227], Train loss :1.977, Train acc: 0.598705501618123
[2024-12-24 17:28:25] - INFO: Epoch: 0, Batch[140/227], Train loss :1.997, Train acc: 0.5929203539823009
[2024-12-24 17:28:27] - INFO: Epoch: 0, Batch[141/227], Train loss :1.945, Train acc: 0.6081382385730212
[2024-12-24 17:28:29] - INFO: Epoch: 0, Batch[142/227], Train loss :2.016, Train acc: 0.5882984433709071
[2024-12-24 17:28:31] - INFO: Epoch: 0, Batch[143/227], Train loss :1.888, Train acc: 0.6154292343387471
[2024-12-24 17:28:33] - INFO: Epoch: 0, Batch[144/227], Train loss :2.017, Train acc: 0.5866666666666667
[2024-12-24 17:28:35] - INFO: Epoch: 0, Batch[145/227], Train loss :2.041, Train acc: 0.5926906779661016
[2024-12-24 17:28:37] - INFO: Epoch: 0, Batch[146/227], Train loss :1.952, Train acc: 0.6139380530973452
[2024-12-24 17:28:40] - INFO: Epoch: 0, Batch[147/227], Train loss :2.020, Train acc: 0.605190502484815
[2024-12-24 17:28:41] - INFO: Epoch: 0, Batch[148/227], Train loss :2.071, Train acc: 0.5891737891737892
[2024-12-24 17:28:43] - INFO: Epoch: 0, Batch[149/227], Train loss :2.019, Train acc: 0.5892448512585813
[2024-12-24 17:28:45] - INFO: Epoch: 0, Batch[150/227], Train loss :1.954, Train acc: 0.6070038910505836
[2024-12-24 17:28:47] - INFO: Epoch: 0, Batch[151/227], Train loss :2.095, Train acc: 0.5825446898002103
[2024-12-24 17:28:50] - INFO: Epoch: 0, Batch[152/227], Train loss :1.932, Train acc: 0.6107613050944476
[2024-12-24 17:28:53] - INFO: Epoch: 0, Batch[153/227], Train loss :2.077, Train acc: 0.5679333680374805
[2024-12-24 17:28:56] - INFO: Epoch: 0, Batch[154/227], Train loss :2.078, Train acc: 0.5717476270240089
[2024-12-24 17:28:58] - INFO: Epoch: 0, Batch[155/227], Train loss :1.860, Train acc: 0.6055096418732783
[2024-12-24 17:29:00] - INFO: Epoch: 0, Batch[156/227], Train loss :2.086, Train acc: 0.5893451720310766
[2024-12-24 17:29:02] - INFO: Epoch: 0, Batch[157/227], Train loss :1.870, Train acc: 0.6233236151603498
[2024-12-24 17:29:04] - INFO: Epoch: 0, Batch[158/227], Train loss :1.975, Train acc: 0.6047164514317799
[2024-12-24 17:29:06] - INFO: Epoch: 0, Batch[159/227], Train loss :2.013, Train acc: 0.6035502958579881
[2024-12-24 17:29:09] - INFO: Epoch: 0, Batch[160/227], Train loss :2.092, Train acc: 0.5668421052631579
[2024-12-24 17:29:11] - INFO: Epoch: 0, Batch[161/227], Train loss :1.945, Train acc: 0.5965909090909091
[2024-12-24 17:29:13] - INFO: Epoch: 0, Batch[162/227], Train loss :1.835, Train acc: 0.61328125
[2024-12-24 17:29:15] - INFO: Epoch: 0, Batch[163/227], Train loss :1.841, Train acc: 0.618941504178273
[2024-12-24 17:29:18] - INFO: Epoch: 0, Batch[164/227], Train loss :2.038, Train acc: 0.5796033994334278
[2024-12-24 17:29:21] - INFO: Epoch: 0, Batch[165/227], Train loss :1.949, Train acc: 0.6161504424778761
[2024-12-24 17:29:23] - INFO: Epoch: 0, Batch[166/227], Train loss :1.940, Train acc: 0.5952655889145496
[2024-12-24 17:29:24] - INFO: Epoch: 0, Batch[167/227], Train loss :1.979, Train acc: 0.5957086391869001
[2024-12-24 17:29:26] - INFO: Epoch: 0, Batch[168/227], Train loss :1.912, Train acc: 0.6189402480270575
[2024-12-24 17:29:28] - INFO: Epoch: 0, Batch[169/227], Train loss :1.996, Train acc: 0.6003372681281619
[2024-12-24 17:29:30] - INFO: Epoch: 0, Batch[170/227], Train loss :1.979, Train acc: 0.5939635535307517
[2024-12-24 17:29:33] - INFO: Epoch: 0, Batch[171/227], Train loss :1.947, Train acc: 0.5900109769484083
[2024-12-24 17:29:35] - INFO: Epoch: 0, Batch[172/227], Train loss :1.890, Train acc: 0.5983651226158038
[2024-12-24 17:29:38] - INFO: Epoch: 0, Batch[173/227], Train loss :2.023, Train acc: 0.5901360544217688
[2024-12-24 17:29:40] - INFO: Epoch: 0, Batch[174/227], Train loss :1.966, Train acc: 0.6054384017758047
[2024-12-24 17:29:41] - INFO: Epoch: 0, Batch[175/227], Train loss :1.897, Train acc: 0.5958286358511837
[2024-12-24 17:29:44] - INFO: Epoch: 0, Batch[176/227], Train loss :2.001, Train acc: 0.588016967126193
[2024-12-24 17:29:47] - INFO: Epoch: 0, Batch[177/227], Train loss :2.013, Train acc: 0.5920314253647587
[2024-12-24 17:29:50] - INFO: Epoch: 0, Batch[178/227], Train loss :2.082, Train acc: 0.5785354946897708
[2024-12-24 17:29:52] - INFO: Epoch: 0, Batch[179/227], Train loss :1.989, Train acc: 0.588008800880088
[2024-12-24 17:29:54] - INFO: Epoch: 0, Batch[180/227], Train loss :1.969, Train acc: 0.5962059620596206
[2024-12-24 17:29:56] - INFO: Epoch: 0, Batch[181/227], Train loss :1.937, Train acc: 0.6032786885245902
[2024-12-24 17:29:58] - INFO: Epoch: 0, Batch[182/227], Train loss :1.875, Train acc: 0.6101021566401816
[2024-12-24 17:30:00] - INFO: Epoch: 0, Batch[183/227], Train loss :1.854, Train acc: 0.6183547845551203
[2024-12-24 17:30:02] - INFO: Epoch: 0, Batch[184/227], Train loss :1.970, Train acc: 0.6057585825027686
[2024-12-24 17:30:04] - INFO: Epoch: 0, Batch[185/227], Train loss :1.902, Train acc: 0.6112975391498882
[2024-12-24 17:30:06] - INFO: Epoch: 0, Batch[186/227], Train loss :1.897, Train acc: 0.5987654320987654
[2024-12-24 17:30:09] - INFO: Epoch: 0, Batch[187/227], Train loss :1.990, Train acc: 0.5875402792696026
[2024-12-24 17:30:11] - INFO: Epoch: 0, Batch[188/227], Train loss :2.117, Train acc: 0.5788888888888889
[2024-12-24 17:30:13] - INFO: Epoch: 0, Batch[189/227], Train loss :2.043, Train acc: 0.5899772209567198
[2024-12-24 17:30:15] - INFO: Epoch: 0, Batch[190/227], Train loss :1.885, Train acc: 0.6190748143917761
[2024-12-24 17:30:17] - INFO: Epoch: 0, Batch[191/227], Train loss :2.066, Train acc: 0.587431693989071
[2024-12-24 17:30:19] - INFO: Epoch: 0, Batch[192/227], Train loss :2.013, Train acc: 0.5927576601671309
[2024-12-24 17:30:22] - INFO: Epoch: 0, Batch[193/227], Train loss :2.047, Train acc: 0.592880978865406
[2024-12-24 17:30:24] - INFO: Epoch: 0, Batch[194/227], Train loss :2.051, Train acc: 0.5896047644829453
[2024-12-24 17:30:26] - INFO: Epoch: 0, Batch[195/227], Train loss :1.891, Train acc: 0.6150537634408603
[2024-12-24 17:30:28] - INFO: Epoch: 0, Batch[196/227], Train loss :1.922, Train acc: 0.614207650273224
[2024-12-24 17:30:30] - INFO: Epoch: 0, Batch[197/227], Train loss :2.081, Train acc: 0.5836575875486382
[2024-12-24 17:30:33] - INFO: Epoch: 0, Batch[198/227], Train loss :2.040, Train acc: 0.5800653594771242
[2024-12-24 17:30:35] - INFO: Epoch: 0, Batch[199/227], Train loss :1.946, Train acc: 0.5982758620689655
[2024-12-24 17:30:37] - INFO: Epoch: 0, Batch[200/227], Train loss :1.913, Train acc: 0.6091825307950728
[2024-12-24 17:30:39] - INFO: Epoch: 0, Batch[201/227], Train loss :1.946, Train acc: 0.58968850698174
[2024-12-24 17:30:41] - INFO: Epoch: 0, Batch[202/227], Train loss :2.005, Train acc: 0.5883324160704458
[2024-12-24 17:30:43] - INFO: Epoch: 0, Batch[203/227], Train loss :1.991, Train acc: 0.5864197530864198
[2024-12-24 17:30:45] - INFO: Epoch: 0, Batch[204/227], Train loss :1.955, Train acc: 0.6056022408963585
[2024-12-24 17:30:47] - INFO: Epoch: 0, Batch[205/227], Train loss :2.098, Train acc: 0.5911062906724512
[2024-12-24 17:30:49] - INFO: Epoch: 0, Batch[206/227], Train loss :2.000, Train acc: 0.5926544240400667
[2024-12-24 17:30:52] - INFO: Epoch: 0, Batch[207/227], Train loss :1.915, Train acc: 0.6144236229415105
[2024-12-24 17:30:54] - INFO: Epoch: 0, Batch[208/227], Train loss :1.926, Train acc: 0.6138728323699422
[2024-12-24 17:30:56] - INFO: Epoch: 0, Batch[209/227], Train loss :2.074, Train acc: 0.5775956284153005
[2024-12-24 17:30:59] - INFO: Epoch: 0, Batch[210/227], Train loss :2.139, Train acc: 0.5696132596685083
[2024-12-24 17:31:01] - INFO: Epoch: 0, Batch[211/227], Train loss :2.066, Train acc: 0.5777525539160046
[2024-12-24 17:31:03] - INFO: Epoch: 0, Batch[212/227], Train loss :2.005, Train acc: 0.5980941704035875
[2024-12-24 17:31:05] - INFO: Epoch: 0, Batch[213/227], Train loss :1.955, Train acc: 0.5843733043950081
[2024-12-24 17:31:08] - INFO: Epoch: 0, Batch[214/227], Train loss :2.080, Train acc: 0.5837651122625216
[2024-12-24 17:31:10] - INFO: Epoch: 0, Batch[215/227], Train loss :2.019, Train acc: 0.5825510767531751
[2024-12-24 17:31:13] - INFO: Epoch: 0, Batch[216/227], Train loss :1.885, Train acc: 0.6197183098591549
[2024-12-24 17:31:14] - INFO: Epoch: 0, Batch[217/227], Train loss :1.989, Train acc: 0.5872844827586207
[2024-12-24 17:31:17] - INFO: Epoch: 0, Batch[218/227], Train loss :2.033, Train acc: 0.5889070146818923
[2024-12-24 17:31:19] - INFO: Epoch: 0, Batch[219/227], Train loss :1.955, Train acc: 0.5980941704035875
[2024-12-24 17:31:22] - INFO: Epoch: 0, Batch[220/227], Train loss :2.036, Train acc: 0.5974458634092171
[2024-12-24 17:31:24] - INFO: Epoch: 0, Batch[221/227], Train loss :1.905, Train acc: 0.6056034482758621
[2024-12-24 17:31:26] - INFO: Epoch: 0, Batch[222/227], Train loss :2.076, Train acc: 0.5858021240916713
[2024-12-24 17:31:28] - INFO: Epoch: 0, Batch[223/227], Train loss :1.901, Train acc: 0.6038374717832957
[2024-12-24 17:31:30] - INFO: Epoch: 0, Batch[224/227], Train loss :2.055, Train acc: 0.5825991189427313
[2024-12-24 17:31:32] - INFO: Epoch: 0, Batch[225/227], Train loss :1.980, Train acc: 0.5889967637540453
[2024-12-24 17:31:33] - INFO: Epoch: 0, Batch[226/227], Train loss :2.115, Train acc: 0.5948103792415169
[2024-12-24 17:31:33] - INFO: Epoch: 0, Train loss: 2.017, Epoch time = 504.398s
[2024-12-24 17:31:35] - INFO: Epoch: 1, Batch[0/227], Train loss :2.018, Train acc: 0.5821771611526148
[2024-12-24 17:31:38] - INFO: Epoch: 1, Batch[1/227], Train loss :1.993, Train acc: 0.6039713182570325
[2024-12-24 17:31:40] - INFO: Epoch: 1, Batch[2/227], Train loss :1.824, Train acc: 0.6228473019517795
[2024-12-24 17:31:42] - INFO: Epoch: 1, Batch[3/227], Train loss :1.870, Train acc: 0.6236263736263736
[2024-12-24 17:31:44] - INFO: Epoch: 1, Batch[4/227], Train loss :1.953, Train acc: 0.5797901711761457
[2024-12-24 17:31:46] - INFO: Epoch: 1, Batch[5/227], Train loss :1.919, Train acc: 0.6066376496191512
[2024-12-24 17:31:49] - INFO: Epoch: 1, Batch[6/227], Train loss :1.885, Train acc: 0.6109947643979058
[2024-12-24 17:31:52] - INFO: Epoch: 1, Batch[7/227], Train loss :1.909, Train acc: 0.6094182825484764
[2024-12-24 17:31:54] - INFO: Epoch: 1, Batch[8/227], Train loss :1.964, Train acc: 0.597682119205298
[2024-12-24 17:31:56] - INFO: Epoch: 1, Batch[9/227], Train loss :1.948, Train acc: 0.5956639566395664
[2024-12-24 17:31:58] - INFO: Epoch: 1, Batch[10/227], Train loss :1.981, Train acc: 0.5855188141391106
[2024-12-24 17:32:01] - INFO: Epoch: 1, Batch[11/227], Train loss :1.982, Train acc: 0.6031487513572205
[2024-12-24 17:32:03] - INFO: Epoch: 1, Batch[12/227], Train loss :1.919, Train acc: 0.6007972665148064
[2024-12-24 17:32:05] - INFO: Epoch: 1, Batch[13/227], Train loss :1.924, Train acc: 0.6002120890774125
[2024-12-24 17:32:07] - INFO: Epoch: 1, Batch[14/227], Train loss :1.981, Train acc: 0.5934782608695652
[2024-12-24 17:32:09] - INFO: Epoch: 1, Batch[15/227], Train loss :1.886, Train acc: 0.613469156762875
[2024-12-24 17:32:11] - INFO: Epoch: 1, Batch[16/227], Train loss :1.942, Train acc: 0.5992152466367713
[2024-12-24 17:32:13] - INFO: Epoch: 1, Batch[17/227], Train loss :1.912, Train acc: 0.5854922279792746
[2024-12-24 17:32:15] - INFO: Epoch: 1, Batch[18/227], Train loss :1.900, Train acc: 0.6061281337047354
[2024-12-24 17:32:17] - INFO: Epoch: 1, Batch[19/227], Train loss :2.036, Train acc: 0.5956164383561644
[2024-12-24 17:32:19] - INFO: Epoch: 1, Batch[20/227], Train loss :1.955, Train acc: 0.6075801749271137
[2024-12-24 17:32:22] - INFO: Epoch: 1, Batch[21/227], Train loss :1.835, Train acc: 0.615257048092869
[2024-12-24 17:32:24] - INFO: Epoch: 1, Batch[22/227], Train loss :1.788, Train acc: 0.6265857694429123
[2024-12-24 17:32:26] - INFO: Epoch: 1, Batch[23/227], Train loss :1.936, Train acc: 0.5844594594594594
[2024-12-24 17:32:28] - INFO: Epoch: 1, Batch[24/227], Train loss :1.923, Train acc: 0.6054840514829323
[2024-12-24 17:32:30] - INFO: Epoch: 1, Batch[25/227], Train loss :1.928, Train acc: 0.5973154362416108
[2024-12-24 17:32:33] - INFO: Epoch: 1, Batch[26/227], Train loss :1.906, Train acc: 0.5884907709011944
[2024-12-24 17:32:34] - INFO: Epoch: 1, Batch[27/227], Train loss :1.960, Train acc: 0.5847860538827259
[2024-12-24 17:32:37] - INFO: Epoch: 1, Batch[28/227], Train loss :1.964, Train acc: 0.6138560687432868
[2024-12-24 17:32:39] - INFO: Epoch: 1, Batch[29/227], Train loss :1.925, Train acc: 0.5983935742971888
[2024-12-24 17:32:41] - INFO: Epoch: 1, Batch[30/227], Train loss :2.041, Train acc: 0.6030751708428246
[2024-12-24 17:32:43] - INFO: Epoch: 1, Batch[31/227], Train loss :1.930, Train acc: 0.59978009895547
[2024-12-24 17:32:46] - INFO: Epoch: 1, Batch[32/227], Train loss :1.941, Train acc: 0.5963855421686747
[2024-12-24 17:32:48] - INFO: Epoch: 1, Batch[33/227], Train loss :1.989, Train acc: 0.5938841201716738
[2024-12-24 17:32:51] - INFO: Epoch: 1, Batch[34/227], Train loss :1.856, Train acc: 0.6116557734204793
[2024-12-24 17:32:54] - INFO: Epoch: 1, Batch[35/227], Train loss :2.070, Train acc: 0.5826468973091707
[2024-12-24 17:32:56] - INFO: Epoch: 1, Batch[36/227], Train loss :1.974, Train acc: 0.5985130111524164
[2024-12-24 17:32:58] - INFO: Epoch: 1, Batch[37/227], Train loss :1.924, Train acc: 0.6067988668555241
[2024-12-24 17:33:00] - INFO: Epoch: 1, Batch[38/227], Train loss :1.930, Train acc: 0.599778883360973
[2024-12-24 17:33:03] - INFO: Epoch: 1, Batch[39/227], Train loss :1.903, Train acc: 0.6136612021857923
[2024-12-24 17:33:05] - INFO: Epoch: 1, Batch[40/227], Train loss :1.925, Train acc: 0.5980707395498392
[2024-12-24 17:33:07] - INFO: Epoch: 1, Batch[41/227], Train loss :1.933, Train acc: 0.5877192982456141
[2024-12-24 17:33:09] - INFO: Epoch: 1, Batch[42/227], Train loss :1.860, Train acc: 0.6124454148471615
[2024-12-24 17:33:13] - INFO: Epoch: 1, Batch[43/227], Train loss :1.942, Train acc: 0.5977198697068404
[2024-12-24 17:33:14] - INFO: Epoch: 1, Batch[44/227], Train loss :2.006, Train acc: 0.5957086391869001
[2024-12-24 17:33:17] - INFO: Epoch: 1, Batch[45/227], Train loss :1.862, Train acc: 0.6160409556313993
[2024-12-24 17:33:19] - INFO: Epoch: 1, Batch[46/227], Train loss :1.982, Train acc: 0.5919824272377814
[2024-12-24 17:33:22] - INFO: Epoch: 1, Batch[47/227], Train loss :1.919, Train acc: 0.6018930957683741
[2024-12-24 17:33:24] - INFO: Epoch: 1, Batch[48/227], Train loss :1.975, Train acc: 0.6076662908680946
[2024-12-24 17:33:26] - INFO: Epoch: 1, Batch[49/227], Train loss :1.901, Train acc: 0.6090308370044053
[2024-12-24 17:33:28] - INFO: Epoch: 1, Batch[50/227], Train loss :1.948, Train acc: 0.5959488272921108
[2024-12-24 17:33:31] - INFO: Epoch: 1, Batch[51/227], Train loss :1.953, Train acc: 0.5971692977681001
[2024-12-24 17:33:34] - INFO: Epoch: 1, Batch[52/227], Train loss :1.945, Train acc: 0.6102449888641426
[2024-12-24 17:33:36] - INFO: Epoch: 1, Batch[53/227], Train loss :1.960, Train acc: 0.5925122083559414
[2024-12-24 17:33:39] - INFO: Epoch: 1, Batch[54/227], Train loss :2.050, Train acc: 0.5952643171806168
[2024-12-24 17:33:41] - INFO: Epoch: 1, Batch[55/227], Train loss :1.930, Train acc: 0.6
[2024-12-24 17:33:43] - INFO: Epoch: 1, Batch[56/227], Train loss :1.940, Train acc: 0.6047008547008547
[2024-12-24 17:33:45] - INFO: Epoch: 1, Batch[57/227], Train loss :1.917, Train acc: 0.6097832128960534
[2024-12-24 17:33:48] - INFO: Epoch: 1, Batch[58/227], Train loss :1.961, Train acc: 0.6018790369935408
[2024-12-24 17:33:50] - INFO: Epoch: 1, Batch[59/227], Train loss :2.065, Train acc: 0.598694942903752
[2024-12-24 17:33:53] - INFO: Epoch: 1, Batch[60/227], Train loss :1.967, Train acc: 0.5885117493472585
[2024-12-24 17:33:54] - INFO: Epoch: 1, Batch[61/227], Train loss :1.779, Train acc: 0.6209490740740741
[2024-12-24 17:33:57] - INFO: Epoch: 1, Batch[62/227], Train loss :2.029, Train acc: 0.5903614457831325
[2024-12-24 17:33:59] - INFO: Epoch: 1, Batch[63/227], Train loss :2.006, Train acc: 0.5891255605381166
[2024-12-24 17:34:02] - INFO: Epoch: 1, Batch[64/227], Train loss :2.027, Train acc: 0.6142938173567782
[2024-12-24 17:34:04] - INFO: Epoch: 1, Batch[65/227], Train loss :1.914, Train acc: 0.6048292108362779
[2024-12-24 17:34:06] - INFO: Epoch: 1, Batch[66/227], Train loss :1.993, Train acc: 0.5980758952431855
[2024-12-24 17:34:08] - INFO: Epoch: 1, Batch[67/227], Train loss :1.947, Train acc: 0.603954802259887
[2024-12-24 17:34:11] - INFO: Epoch: 1, Batch[68/227], Train loss :1.854, Train acc: 0.6282339707536558
[2024-12-24 17:34:13] - INFO: Epoch: 1, Batch[69/227], Train loss :1.902, Train acc: 0.6046128500823723
[2024-12-24 17:34:16] - INFO: Epoch: 1, Batch[70/227], Train loss :1.898, Train acc: 0.6211111111111111
[2024-12-24 17:34:18] - INFO: Epoch: 1, Batch[71/227], Train loss :1.980, Train acc: 0.61003861003861
[2024-12-24 17:34:21] - INFO: Epoch: 1, Batch[72/227], Train loss :1.932, Train acc: 0.6107042253521127
[2024-12-24 17:34:22] - INFO: Epoch: 1, Batch[73/227], Train loss :2.055, Train acc: 0.5794392523364486
[2024-12-24 17:34:26] - INFO: Epoch: 1, Batch[74/227], Train loss :1.973, Train acc: 0.6131855309218203
[2024-12-24 17:34:27] - INFO: Epoch: 1, Batch[75/227], Train loss :1.922, Train acc: 0.6035734226689
[2024-12-24 17:34:30] - INFO: Epoch: 1, Batch[76/227], Train loss :1.830, Train acc: 0.6200798630918426
[2024-12-24 17:34:32] - INFO: Epoch: 1, Batch[77/227], Train loss :1.933, Train acc: 0.6054461181923523
[2024-12-24 17:34:34] - INFO: Epoch: 1, Batch[78/227], Train loss :1.893, Train acc: 0.6165540540540541
[2024-12-24 17:34:36] - INFO: Epoch: 1, Batch[79/227], Train loss :1.906, Train acc: 0.6036446469248291
[2024-12-24 17:34:38] - INFO: Epoch: 1, Batch[80/227], Train loss :1.925, Train acc: 0.603648424543947
[2024-12-24 17:34:40] - INFO: Epoch: 1, Batch[81/227], Train loss :2.020, Train acc: 0.5818759936406995
[2024-12-24 17:34:42] - INFO: Epoch: 1, Batch[82/227], Train loss :1.903, Train acc: 0.6095396561286744
[2024-12-24 17:34:44] - INFO: Epoch: 1, Batch[83/227], Train loss :1.900, Train acc: 0.6216517857142857
[2024-12-24 17:34:46] - INFO: Epoch: 1, Batch[84/227], Train loss :2.066, Train acc: 0.5764383561643835
[2024-12-24 17:34:49] - INFO: Epoch: 1, Batch[85/227], Train loss :2.061, Train acc: 0.5873972602739727
[2024-12-24 17:34:52] - INFO: Epoch: 1, Batch[86/227], Train loss :1.955, Train acc: 0.6048988285410011
[2024-12-24 17:34:54] - INFO: Epoch: 1, Batch[87/227], Train loss :1.953, Train acc: 0.5982388552559164
[2024-12-24 17:34:57] - INFO: Epoch: 1, Batch[88/227], Train loss :1.912, Train acc: 0.6300056401579244
[2024-12-24 17:34:59] - INFO: Epoch: 1, Batch[89/227], Train loss :1.974, Train acc: 0.5963656387665198
[2024-12-24 17:35:01] - INFO: Epoch: 1, Batch[90/227], Train loss :1.977, Train acc: 0.6001109262340544
[2024-12-24 17:35:03] - INFO: Epoch: 1, Batch[91/227], Train loss :1.824, Train acc: 0.6229050279329609
[2024-12-24 17:35:05] - INFO: Epoch: 1, Batch[92/227], Train loss :1.735, Train acc: 0.6436233611442194
[2024-12-24 17:35:07] - INFO: Epoch: 1, Batch[93/227], Train loss :1.875, Train acc: 0.6121521862578081
[2024-12-24 17:35:10] - INFO: Epoch: 1, Batch[94/227], Train loss :1.958, Train acc: 0.6090351366424986
[2024-12-24 17:35:11] - INFO: Epoch: 1, Batch[95/227], Train loss :1.935, Train acc: 0.6049861495844875
[2024-12-24 17:35:13] - INFO: Epoch: 1, Batch[96/227], Train loss :2.027, Train acc: 0.5992303463441452
[2024-12-24 17:35:15] - INFO: Epoch: 1, Batch[97/227], Train loss :2.017, Train acc: 0.5890333521763709
[2024-12-24 17:35:17] - INFO: Epoch: 1, Batch[98/227], Train loss :1.925, Train acc: 0.6101141924959217
[2024-12-24 17:35:19] - INFO: Epoch: 1, Batch[99/227], Train loss :2.045, Train acc: 0.5877644368210406
[2024-12-24 17:35:22] - INFO: Epoch: 1, Batch[100/227], Train loss :1.932, Train acc: 0.6043360433604336
[2024-12-24 17:35:24] - INFO: Epoch: 1, Batch[101/227], Train loss :2.056, Train acc: 0.5965401785714286
[2024-12-24 17:35:26] - INFO: Epoch: 1, Batch[102/227], Train loss :1.833, Train acc: 0.6196854979615609
[2024-12-24 17:35:27] - INFO: Epoch: 1, Batch[103/227], Train loss :1.897, Train acc: 0.5915649278579356
[2024-12-24 17:35:29] - INFO: Epoch: 1, Batch[104/227], Train loss :1.936, Train acc: 0.6145952109464082
[2024-12-24 17:35:31] - INFO: Epoch: 1, Batch[105/227], Train loss :1.961, Train acc: 0.5949152542372881
[2024-12-24 17:35:34] - INFO: Epoch: 1, Batch[106/227], Train loss :1.992, Train acc: 0.5903954802259888
[2024-12-24 17:35:36] - INFO: Epoch: 1, Batch[107/227], Train loss :2.023, Train acc: 0.5760986066452305
[2024-12-24 17:35:38] - INFO: Epoch: 1, Batch[108/227], Train loss :1.963, Train acc: 0.598434004474273
[2024-12-24 17:35:41] - INFO: Epoch: 1, Batch[109/227], Train loss :1.958, Train acc: 0.5923159018143009
[2024-12-24 17:35:43] - INFO: Epoch: 1, Batch[110/227], Train loss :1.989, Train acc: 0.6187782805429864
[2024-12-24 17:35:45] - INFO: Epoch: 1, Batch[111/227], Train loss :1.835, Train acc: 0.612884834663626
[2024-12-24 17:35:47] - INFO: Epoch: 1, Batch[112/227], Train loss :1.945, Train acc: 0.5978021978021978
[2024-12-24 17:35:50] - INFO: Epoch: 1, Batch[113/227], Train loss :2.044, Train acc: 0.5898109243697479
[2024-12-24 17:35:52] - INFO: Epoch: 1, Batch[114/227], Train loss :1.896, Train acc: 0.6152125279642058
[2024-12-24 17:35:54] - INFO: Epoch: 1, Batch[115/227], Train loss :1.900, Train acc: 0.6125211505922166
[2024-12-24 17:35:56] - INFO: Epoch: 1, Batch[116/227], Train loss :1.976, Train acc: 0.5962596259625963
[2024-12-24 17:35:59] - INFO: Epoch: 1, Batch[117/227], Train loss :1.965, Train acc: 0.5979547900968784
[2024-12-24 17:36:01] - INFO: Epoch: 1, Batch[118/227], Train loss :2.022, Train acc: 0.6036892118501956
[2024-12-24 17:36:03] - INFO: Epoch: 1, Batch[119/227], Train loss :2.003, Train acc: 0.5885837372105547
[2024-12-24 17:36:04] - INFO: Epoch: 1, Batch[120/227], Train loss :2.006, Train acc: 0.5837372105546581
[2024-12-24 17:36:06] - INFO: Epoch: 1, Batch[121/227], Train loss :2.004, Train acc: 0.6054732041049031
[2024-12-24 17:36:08] - INFO: Epoch: 1, Batch[122/227], Train loss :1.988, Train acc: 0.5975405254332029
[2024-12-24 17:36:11] - INFO: Epoch: 1, Batch[123/227], Train loss :1.831, Train acc: 0.6201463140123804
[2024-12-24 17:36:13] - INFO: Epoch: 1, Batch[124/227], Train loss :2.010, Train acc: 0.589041095890411
[2024-12-24 17:36:15] - INFO: Epoch: 1, Batch[125/227], Train loss :2.011, Train acc: 0.5748898678414097
[2024-12-24 17:36:18] - INFO: Epoch: 1, Batch[126/227], Train loss :2.011, Train acc: 0.5997818974918212
[2024-12-24 17:36:22] - INFO: Epoch: 1, Batch[127/227], Train loss :1.982, Train acc: 0.5951302378255946
[2024-12-24 17:36:24] - INFO: Epoch: 1, Batch[128/227], Train loss :2.037, Train acc: 0.6010752688172043
[2024-12-24 17:36:26] - INFO: Epoch: 1, Batch[129/227], Train loss :2.056, Train acc: 0.5911949685534591
[2024-12-24 17:36:28] - INFO: Epoch: 1, Batch[130/227], Train loss :2.089, Train acc: 0.5849691531127313
[2024-12-24 17:36:30] - INFO: Epoch: 1, Batch[131/227], Train loss :1.979, Train acc: 0.6104718066743383
[2024-12-24 17:36:32] - INFO: Epoch: 1, Batch[132/227], Train loss :2.007, Train acc: 0.6046511627906976
[2024-12-24 17:36:34] - INFO: Epoch: 1, Batch[133/227], Train loss :2.065, Train acc: 0.581984897518878
[2024-12-24 17:36:38] - INFO: Epoch: 1, Batch[134/227], Train loss :2.010, Train acc: 0.580226904376013
[2024-12-24 17:36:40] - INFO: Epoch: 1, Batch[135/227], Train loss :1.958, Train acc: 0.5943447525829255
[2024-12-24 17:36:42] - INFO: Epoch: 1, Batch[136/227], Train loss :2.062, Train acc: 0.5851305334846765
[2024-12-24 17:36:44] - INFO: Epoch: 1, Batch[137/227], Train loss :1.779, Train acc: 0.6395348837209303
[2024-12-24 17:36:46] - INFO: Epoch: 1, Batch[138/227], Train loss :2.025, Train acc: 0.5954746136865342
[2024-12-24 17:36:48] - INFO: Epoch: 1, Batch[139/227], Train loss :1.838, Train acc: 0.6242555495397942
[2024-12-24 17:36:51] - INFO: Epoch: 1, Batch[140/227], Train loss :1.951, Train acc: 0.5978142076502733
[2024-12-24 17:36:53] - INFO: Epoch: 1, Batch[141/227], Train loss :1.860, Train acc: 0.6175972927241963
[2024-12-24 17:36:56] - INFO: Epoch: 1, Batch[142/227], Train loss :2.054, Train acc: 0.5884315906562848
[2024-12-24 17:36:58] - INFO: Epoch: 1, Batch[143/227], Train loss :1.870, Train acc: 0.6155115511551155
[2024-12-24 17:37:00] - INFO: Epoch: 1, Batch[144/227], Train loss :2.019, Train acc: 0.5821522309711286
[2024-12-24 17:37:02] - INFO: Epoch: 1, Batch[145/227], Train loss :1.917, Train acc: 0.6012338754907459
[2024-12-24 17:37:05] - INFO: Epoch: 1, Batch[146/227], Train loss :1.978, Train acc: 0.5939597315436241
[2024-12-24 17:37:07] - INFO: Epoch: 1, Batch[147/227], Train loss :1.999, Train acc: 0.6037282020444978
[2024-12-24 17:37:10] - INFO: Epoch: 1, Batch[148/227], Train loss :1.927, Train acc: 0.5954994511525796
[2024-12-24 17:37:11] - INFO: Epoch: 1, Batch[149/227], Train loss :1.861, Train acc: 0.6078323221180364
[2024-12-24 17:37:13] - INFO: Epoch: 1, Batch[150/227], Train loss :1.928, Train acc: 0.609366391184573
[2024-12-24 17:37:15] - INFO: Epoch: 1, Batch[151/227], Train loss :1.859, Train acc: 0.6231802911534154
[2024-12-24 17:37:17] - INFO: Epoch: 1, Batch[152/227], Train loss :1.835, Train acc: 0.6124586549062845
[2024-12-24 17:37:21] - INFO: Epoch: 1, Batch[153/227], Train loss :1.958, Train acc: 0.6056185845488925
[2024-12-24 17:37:23] - INFO: Epoch: 1, Batch[154/227], Train loss :1.979, Train acc: 0.5943238731218697
[2024-12-24 17:37:25] - INFO: Epoch: 1, Batch[155/227], Train loss :1.950, Train acc: 0.5884884346422808
[2024-12-24 17:37:27] - INFO: Epoch: 1, Batch[156/227], Train loss :1.914, Train acc: 0.5953998954521693
[2024-12-24 17:37:29] - INFO: Epoch: 1, Batch[157/227], Train loss :2.010, Train acc: 0.5840807174887892
[2024-12-24 17:37:32] - INFO: Epoch: 1, Batch[158/227], Train loss :1.832, Train acc: 0.6251428571428571
[2024-12-24 17:37:34] - INFO: Epoch: 1, Batch[159/227], Train loss :1.913, Train acc: 0.6099020674646355
[2024-12-24 17:37:36] - INFO: Epoch: 1, Batch[160/227], Train loss :2.007, Train acc: 0.6002202643171806
[2024-12-24 17:37:38] - INFO: Epoch: 1, Batch[161/227], Train loss :1.942, Train acc: 0.5948464912280702
[2024-12-24 17:37:40] - INFO: Epoch: 1, Batch[162/227], Train loss :1.892, Train acc: 0.6138079827400216
[2024-12-24 17:37:42] - INFO: Epoch: 1, Batch[163/227], Train loss :2.064, Train acc: 0.5864406779661017
[2024-12-24 17:37:44] - INFO: Epoch: 1, Batch[164/227], Train loss :1.938, Train acc: 0.6108319374651033
[2024-12-24 17:37:46] - INFO: Epoch: 1, Batch[165/227], Train loss :1.913, Train acc: 0.6188907193849533
[2024-12-24 17:37:48] - INFO: Epoch: 1, Batch[166/227], Train loss :2.104, Train acc: 0.5827298050139276
[2024-12-24 17:37:51] - INFO: Epoch: 1, Batch[167/227], Train loss :1.998, Train acc: 0.5932671081677704
[2024-12-24 17:37:53] - INFO: Epoch: 1, Batch[168/227], Train loss :1.866, Train acc: 0.6174927113702624
[2024-12-24 17:37:56] - INFO: Epoch: 1, Batch[169/227], Train loss :2.037, Train acc: 0.5954997383568812
[2024-12-24 17:37:58] - INFO: Epoch: 1, Batch[170/227], Train loss :1.911, Train acc: 0.6082359488035615
[2024-12-24 17:38:01] - INFO: Epoch: 1, Batch[171/227], Train loss :2.001, Train acc: 0.5950759559979046
[2024-12-24 17:38:03] - INFO: Epoch: 1, Batch[172/227], Train loss :2.051, Train acc: 0.5822784810126582
[2024-12-24 17:38:05] - INFO: Epoch: 1, Batch[173/227], Train loss :1.879, Train acc: 0.6198300283286119
[2024-12-24 17:38:07] - INFO: Epoch: 1, Batch[174/227], Train loss :1.943, Train acc: 0.6042382588774341
[2024-12-24 17:38:09] - INFO: Epoch: 1, Batch[175/227], Train loss :1.832, Train acc: 0.6217765042979942
[2024-12-24 17:38:12] - INFO: Epoch: 1, Batch[176/227], Train loss :2.012, Train acc: 0.5991237677984665
[2024-12-24 17:38:14] - INFO: Epoch: 1, Batch[177/227], Train loss :1.980, Train acc: 0.602808988764045
[2024-12-24 17:38:16] - INFO: Epoch: 1, Batch[178/227], Train loss :2.011, Train acc: 0.5797333333333333
[2024-12-24 17:38:18] - INFO: Epoch: 1, Batch[179/227], Train loss :1.833, Train acc: 0.6134499726626572
[2024-12-24 17:38:20] - INFO: Epoch: 1, Batch[180/227], Train loss :1.862, Train acc: 0.6159954622802042
[2024-12-24 17:38:22] - INFO: Epoch: 1, Batch[181/227], Train loss :1.811, Train acc: 0.6300345224395857
[2024-12-24 17:38:24] - INFO: Epoch: 1, Batch[182/227], Train loss :1.897, Train acc: 0.6143344709897611
[2024-12-24 17:38:26] - INFO: Epoch: 1, Batch[183/227], Train loss :1.914, Train acc: 0.6019929660023446
[2024-12-24 17:38:28] - INFO: Epoch: 1, Batch[184/227], Train loss :2.092, Train acc: 0.5728
[2024-12-24 17:38:30] - INFO: Epoch: 1, Batch[185/227], Train loss :1.968, Train acc: 0.5921787709497207
[2024-12-24 17:38:32] - INFO: Epoch: 1, Batch[186/227], Train loss :1.901, Train acc: 0.595146166574738
[2024-12-24 17:38:34] - INFO: Epoch: 1, Batch[187/227], Train loss :2.015, Train acc: 0.5914315569487983
[2024-12-24 17:38:36] - INFO: Epoch: 1, Batch[188/227], Train loss :1.905, Train acc: 0.625069949636262
[2024-12-24 17:38:39] - INFO: Epoch: 1, Batch[189/227], Train loss :1.843, Train acc: 0.6112359550561798
[2024-12-24 17:38:41] - INFO: Epoch: 1, Batch[190/227], Train loss :1.870, Train acc: 0.6095396561286744
[2024-12-24 17:38:43] - INFO: Epoch: 1, Batch[191/227], Train loss :2.015, Train acc: 0.5865331107401224
[2024-12-24 17:38:45] - INFO: Epoch: 1, Batch[192/227], Train loss :1.907, Train acc: 0.5986547085201793
[2024-12-24 17:38:47] - INFO: Epoch: 1, Batch[193/227], Train loss :1.931, Train acc: 0.5978082191780822
[2024-12-24 17:38:49] - INFO: Epoch: 1, Batch[194/227], Train loss :1.923, Train acc: 0.6093394077448747
[2024-12-24 17:38:52] - INFO: Epoch: 1, Batch[195/227], Train loss :1.799, Train acc: 0.6307779030439684
[2024-12-24 17:38:54] - INFO: Epoch: 1, Batch[196/227], Train loss :1.901, Train acc: 0.6037219485495348
[2024-12-24 17:38:56] - INFO: Epoch: 1, Batch[197/227], Train loss :1.954, Train acc: 0.6056745801968731
[2024-12-24 17:38:58] - INFO: Epoch: 1, Batch[198/227], Train loss :2.015, Train acc: 0.5918949771689498
[2024-12-24 17:39:00] - INFO: Epoch: 1, Batch[199/227], Train loss :1.929, Train acc: 0.6038043478260869
[2024-12-24 17:39:02] - INFO: Epoch: 1, Batch[200/227], Train loss :2.044, Train acc: 0.584180790960452
[2024-12-24 17:39:04] - INFO: Epoch: 1, Batch[201/227], Train loss :1.910, Train acc: 0.6005633802816901
[2024-12-24 17:39:06] - INFO: Epoch: 1, Batch[202/227], Train loss :1.858, Train acc: 0.6109012412304371
[2024-12-24 17:39:08] - INFO: Epoch: 1, Batch[203/227], Train loss :1.872, Train acc: 0.6095505617977528
[2024-12-24 17:39:10] - INFO: Epoch: 1, Batch[204/227], Train loss :1.912, Train acc: 0.610648918469218
[2024-12-24 17:39:12] - INFO: Epoch: 1, Batch[205/227], Train loss :1.964, Train acc: 0.5993502977801841
[2024-12-24 17:39:15] - INFO: Epoch: 1, Batch[206/227], Train loss :1.867, Train acc: 0.6077348066298343
[2024-12-24 17:39:17] - INFO: Epoch: 1, Batch[207/227], Train loss :1.922, Train acc: 0.5923460898502496
[2024-12-24 17:39:20] - INFO: Epoch: 1, Batch[208/227], Train loss :1.899, Train acc: 0.6057529610829103
[2024-12-24 17:39:22] - INFO: Epoch: 1, Batch[209/227], Train loss :1.926, Train acc: 0.6090621707060063
[2024-12-24 17:39:24] - INFO: Epoch: 1, Batch[210/227], Train loss :1.859, Train acc: 0.5997770345596433
[2024-12-24 17:39:26] - INFO: Epoch: 1, Batch[211/227], Train loss :1.887, Train acc: 0.6049450549450549
[2024-12-24 17:39:28] - INFO: Epoch: 1, Batch[212/227], Train loss :1.839, Train acc: 0.6222596964586846
[2024-12-24 17:39:31] - INFO: Epoch: 1, Batch[213/227], Train loss :2.000, Train acc: 0.5839822024471635
[2024-12-24 17:39:33] - INFO: Epoch: 1, Batch[214/227], Train loss :1.796, Train acc: 0.631106120157215
[2024-12-24 17:39:34] - INFO: Epoch: 1, Batch[215/227], Train loss :1.930, Train acc: 0.6048913043478261
[2024-12-24 17:39:37] - INFO: Epoch: 1, Batch[216/227], Train loss :1.863, Train acc: 0.6138952164009112
[2024-12-24 17:39:39] - INFO: Epoch: 1, Batch[217/227], Train loss :1.996, Train acc: 0.5882996172772007
[2024-12-24 17:39:41] - INFO: Epoch: 1, Batch[218/227], Train loss :1.830, Train acc: 0.6167763157894737
[2024-12-24 17:39:43] - INFO: Epoch: 1, Batch[219/227], Train loss :1.921, Train acc: 0.5906463478717814
[2024-12-24 17:39:45] - INFO: Epoch: 1, Batch[220/227], Train loss :1.833, Train acc: 0.6297327394209354
[2024-12-24 17:39:47] - INFO: Epoch: 1, Batch[221/227], Train loss :1.862, Train acc: 0.6195358877495952
[2024-12-24 17:39:49] - INFO: Epoch: 1, Batch[222/227], Train loss :1.874, Train acc: 0.6161559888579388
[2024-12-24 17:39:52] - INFO: Epoch: 1, Batch[223/227], Train loss :1.921, Train acc: 0.6016348773841962
[2024-12-24 17:39:54] - INFO: Epoch: 1, Batch[224/227], Train loss :1.909, Train acc: 0.6181613085166384
[2024-12-24 17:39:56] - INFO: Epoch: 1, Batch[225/227], Train loss :1.896, Train acc: 0.6116402116402117
[2024-12-24 17:39:58] - INFO: Epoch: 1, Batch[226/227], Train loss :2.010, Train acc: 0.5806451612903226
[2024-12-24 17:39:58] - INFO: Epoch: 1, Train loss: 1.941, Epoch time = 504.563s
[2024-12-24 17:40:03] - INFO: Accuracy on validation0.558
[2024-12-24 17:40:06] - INFO: Epoch: 2, Batch[0/227], Train loss :1.906, Train acc: 0.6029492080830148
[2024-12-24 17:40:08] - INFO: Epoch: 2, Batch[1/227], Train loss :1.879, Train acc: 0.6083382266588373
[2024-12-24 17:40:10] - INFO: Epoch: 2, Batch[2/227], Train loss :1.890, Train acc: 0.6294773928361714
[2024-12-24 17:40:13] - INFO: Epoch: 2, Batch[3/227], Train loss :2.049, Train acc: 0.5761111111111111
[2024-12-24 17:40:15] - INFO: Epoch: 2, Batch[4/227], Train loss :1.861, Train acc: 0.6183333333333333
[2024-12-24 17:40:17] - INFO: Epoch: 2, Batch[5/227], Train loss :1.832, Train acc: 0.6087912087912087
[2024-12-24 17:40:19] - INFO: Epoch: 2, Batch[6/227], Train loss :1.885, Train acc: 0.6041202672605791
[2024-12-24 17:40:22] - INFO: Epoch: 2, Batch[7/227], Train loss :1.825, Train acc: 0.6081007115489874
[2024-12-24 17:40:24] - INFO: Epoch: 2, Batch[8/227], Train loss :1.802, Train acc: 0.626951995373048
[2024-12-24 17:40:26] - INFO: Epoch: 2, Batch[9/227], Train loss :1.786, Train acc: 0.6224944320712695
[2024-12-24 17:40:28] - INFO: Epoch: 2, Batch[10/227], Train loss :1.853, Train acc: 0.6161449752883031
[2024-12-24 17:40:30] - INFO: Epoch: 2, Batch[11/227], Train loss :1.871, Train acc: 0.6187175043327556
[2024-12-24 17:40:32] - INFO: Epoch: 2, Batch[12/227], Train loss :1.882, Train acc: 0.6179326099371788
[2024-12-24 17:40:34] - INFO: Epoch: 2, Batch[13/227], Train loss :1.906, Train acc: 0.6095617529880478
[2024-12-24 17:40:36] - INFO: Epoch: 2, Batch[14/227], Train loss :1.931, Train acc: 0.6105499438832772
[2024-12-24 17:40:38] - INFO: Epoch: 2, Batch[15/227], Train loss :1.620, Train acc: 0.6603995299647474
[2024-12-24 17:40:40] - INFO: Epoch: 2, Batch[16/227], Train loss :1.996, Train acc: 0.589433131535498
[2024-12-24 17:40:42] - INFO: Epoch: 2, Batch[17/227], Train loss :1.966, Train acc: 0.5993377483443708
[2024-12-24 17:40:44] - INFO: Epoch: 2, Batch[18/227], Train loss :1.763, Train acc: 0.632821075740944
[2024-12-24 17:40:46] - INFO: Epoch: 2, Batch[19/227], Train loss :1.779, Train acc: 0.6251372118551043
[2024-12-24 17:40:48] - INFO: Epoch: 2, Batch[20/227], Train loss :1.678, Train acc: 0.6349480968858131
[2024-12-24 17:40:51] - INFO: Epoch: 2, Batch[21/227], Train loss :1.729, Train acc: 0.6399108138238573
[2024-12-24 17:40:53] - INFO: Epoch: 2, Batch[22/227], Train loss :1.850, Train acc: 0.6157354618015963
[2024-12-24 17:40:55] - INFO: Epoch: 2, Batch[23/227], Train loss :1.861, Train acc: 0.6182618261826183
[2024-12-24 17:40:57] - INFO: Epoch: 2, Batch[24/227], Train loss :1.821, Train acc: 0.6200221238938053
[2024-12-24 17:41:00] - INFO: Epoch: 2, Batch[25/227], Train loss :1.930, Train acc: 0.5978494623655914
[2024-12-24 17:41:02] - INFO: Epoch: 2, Batch[26/227], Train loss :2.009, Train acc: 0.5821880153930731
[2024-12-24 17:41:04] - INFO: Epoch: 2, Batch[27/227], Train loss :1.785, Train acc: 0.6397306397306397
[2024-12-24 17:41:06] - INFO: Epoch: 2, Batch[28/227], Train loss :1.877, Train acc: 0.6019822639540949
[2024-12-24 17:41:09] - INFO: Epoch: 2, Batch[29/227], Train loss :2.001, Train acc: 0.5990466101694916
[2024-12-24 17:41:11] - INFO: Epoch: 2, Batch[30/227], Train loss :1.824, Train acc: 0.615990990990991
[2024-12-24 17:41:14] - INFO: Epoch: 2, Batch[31/227], Train loss :1.869, Train acc: 0.6228165938864629
[2024-12-24 17:41:16] - INFO: Epoch: 2, Batch[32/227], Train loss :1.859, Train acc: 0.6077994428969359
[2024-12-24 17:41:18] - INFO: Epoch: 2, Batch[33/227], Train loss :1.926, Train acc: 0.6061093247588425
[2024-12-24 17:41:21] - INFO: Epoch: 2, Batch[34/227], Train loss :1.668, Train acc: 0.6476949345475241
[2024-12-24 17:41:23] - INFO: Epoch: 2, Batch[35/227], Train loss :1.873, Train acc: 0.6081521739130434
[2024-12-24 17:41:25] - INFO: Epoch: 2, Batch[36/227], Train loss :1.889, Train acc: 0.6168478260869565
[2024-12-24 17:41:27] - INFO: Epoch: 2, Batch[37/227], Train loss :1.827, Train acc: 0.6326297775242442
[2024-12-24 17:41:30] - INFO: Epoch: 2, Batch[38/227], Train loss :1.851, Train acc: 0.6209039548022599
[2024-12-24 17:41:32] - INFO: Epoch: 2, Batch[39/227], Train loss :1.824, Train acc: 0.6226826608505998
[2024-12-24 17:41:34] - INFO: Epoch: 2, Batch[40/227], Train loss :1.757, Train acc: 0.6270422535211267
[2024-12-24 17:41:36] - INFO: Epoch: 2, Batch[41/227], Train loss :1.902, Train acc: 0.6261363636363636
[2024-12-24 17:41:39] - INFO: Epoch: 2, Batch[42/227], Train loss :1.990, Train acc: 0.5821064552661381
[2024-12-24 17:41:41] - INFO: Epoch: 2, Batch[43/227], Train loss :1.880, Train acc: 0.6089568541780448
[2024-12-24 17:41:43] - INFO: Epoch: 2, Batch[44/227], Train loss :1.868, Train acc: 0.5947995666305526
[2024-12-24 17:41:46] - INFO: Epoch: 2, Batch[45/227], Train loss :1.882, Train acc: 0.6197339246119734
[2024-12-24 17:41:48] - INFO: Epoch: 2, Batch[46/227], Train loss :2.006, Train acc: 0.5890038105606968
[2024-12-24 17:41:51] - INFO: Epoch: 2, Batch[47/227], Train loss :1.929, Train acc: 0.607789358200768
[2024-12-24 17:41:53] - INFO: Epoch: 2, Batch[48/227], Train loss :1.823, Train acc: 0.6128674431503051
[2024-12-24 17:41:55] - INFO: Epoch: 2, Batch[49/227], Train loss :1.867, Train acc: 0.6190476190476191
[2024-12-24 17:41:57] - INFO: Epoch: 2, Batch[50/227], Train loss :1.879, Train acc: 0.6012168141592921
[2024-12-24 17:42:00] - INFO: Epoch: 2, Batch[51/227], Train loss :1.926, Train acc: 0.6145038167938931
[2024-12-24 17:42:03] - INFO: Epoch: 2, Batch[52/227], Train loss :1.931, Train acc: 0.599242833964305
[2024-12-24 17:42:05] - INFO: Epoch: 2, Batch[53/227], Train loss :1.724, Train acc: 0.6447368421052632
[2024-12-24 17:42:07] - INFO: Epoch: 2, Batch[54/227], Train loss :1.819, Train acc: 0.6214953271028038
[2024-12-24 17:42:10] - INFO: Epoch: 2, Batch[55/227], Train loss :1.882, Train acc: 0.611771363893605
[2024-12-24 17:42:12] - INFO: Epoch: 2, Batch[56/227], Train loss :1.826, Train acc: 0.6205381658429434
[2024-12-24 17:42:14] - INFO: Epoch: 2, Batch[57/227], Train loss :1.849, Train acc: 0.6126327557294577
[2024-12-24 17:42:16] - INFO: Epoch: 2, Batch[58/227], Train loss :1.761, Train acc: 0.6219931271477663
[2024-12-24 17:42:19] - INFO: Epoch: 2, Batch[59/227], Train loss :1.923, Train acc: 0.5962596259625963
[2024-12-24 17:42:22] - INFO: Epoch: 2, Batch[60/227], Train loss :1.895, Train acc: 0.6104618809126322
[2024-12-24 17:42:23] - INFO: Epoch: 2, Batch[61/227], Train loss :1.840, Train acc: 0.6129411764705882
[2024-12-24 17:42:26] - INFO: Epoch: 2, Batch[62/227], Train loss :1.938, Train acc: 0.5927505330490405
[2024-12-24 17:42:28] - INFO: Epoch: 2, Batch[63/227], Train loss :1.894, Train acc: 0.6149885583524027
[2024-12-24 17:42:30] - INFO: Epoch: 2, Batch[64/227], Train loss :2.085, Train acc: 0.5810147299509002
[2024-12-24 17:42:32] - INFO: Epoch: 2, Batch[65/227], Train loss :1.844, Train acc: 0.6158977209560867
[2024-12-24 17:42:34] - INFO: Epoch: 2, Batch[66/227], Train loss :1.830, Train acc: 0.6165207877461707
[2024-12-24 17:42:36] - INFO: Epoch: 2, Batch[67/227], Train loss :1.843, Train acc: 0.6254935138183869
[2024-12-24 17:42:38] - INFO: Epoch: 2, Batch[68/227], Train loss :1.839, Train acc: 0.6115231032515688
[2024-12-24 17:42:40] - INFO: Epoch: 2, Batch[69/227], Train loss :1.883, Train acc: 0.6024811218985976
[2024-12-24 17:42:42] - INFO: Epoch: 2, Batch[70/227], Train loss :1.798, Train acc: 0.6130653266331658
[2024-12-24 17:42:44] - INFO: Epoch: 2, Batch[71/227], Train loss :1.798, Train acc: 0.6278174821330401
[2024-12-24 17:42:46] - INFO: Epoch: 2, Batch[72/227], Train loss :1.850, Train acc: 0.6067354698533406
[2024-12-24 17:42:48] - INFO: Epoch: 2, Batch[73/227], Train loss :1.789, Train acc: 0.629424778761062
[2024-12-24 17:42:51] - INFO: Epoch: 2, Batch[74/227], Train loss :1.847, Train acc: 0.6211143695014663
[2024-12-24 17:42:53] - INFO: Epoch: 2, Batch[75/227], Train loss :1.966, Train acc: 0.5952254641909814
[2024-12-24 17:42:55] - INFO: Epoch: 2, Batch[76/227], Train loss :1.905, Train acc: 0.6061554512258738
[2024-12-24 17:42:57] - INFO: Epoch: 2, Batch[77/227], Train loss :1.919, Train acc: 0.6040114613180516
[2024-12-24 17:42:59] - INFO: Epoch: 2, Batch[78/227], Train loss :1.905, Train acc: 0.6127232142857143
[2024-12-24 17:43:01] - INFO: Epoch: 2, Batch[79/227], Train loss :1.966, Train acc: 0.60568669527897
[2024-12-24 17:43:04] - INFO: Epoch: 2, Batch[80/227], Train loss :1.931, Train acc: 0.5947712418300654
[2024-12-24 17:43:07] - INFO: Epoch: 2, Batch[81/227], Train loss :1.923, Train acc: 0.5990836197021764
[2024-12-24 17:43:09] - INFO: Epoch: 2, Batch[82/227], Train loss :1.973, Train acc: 0.6055096418732783
[2024-12-24 17:43:12] - INFO: Epoch: 2, Batch[83/227], Train loss :1.934, Train acc: 0.6076839826839827
[2024-12-24 17:43:14] - INFO: Epoch: 2, Batch[84/227], Train loss :1.840, Train acc: 0.6122004357298475
[2024-12-24 17:43:17] - INFO: Epoch: 2, Batch[85/227], Train loss :1.878, Train acc: 0.6151685393258427
[2024-12-24 17:43:19] - INFO: Epoch: 2, Batch[86/227], Train loss :1.878, Train acc: 0.6092140921409214
[2024-12-24 17:43:21] - INFO: Epoch: 2, Batch[87/227], Train loss :1.932, Train acc: 0.5955882352941176
[2024-12-24 17:43:23] - INFO: Epoch: 2, Batch[88/227], Train loss :1.976, Train acc: 0.5904079382579934
[2024-12-24 17:43:26] - INFO: Epoch: 2, Batch[89/227], Train loss :1.872, Train acc: 0.6079698438341411
[2024-12-24 17:43:28] - INFO: Epoch: 2, Batch[90/227], Train loss :1.913, Train acc: 0.6092017738359202
[2024-12-24 17:43:30] - INFO: Epoch: 2, Batch[91/227], Train loss :1.866, Train acc: 0.6252796420581656
[2024-12-24 17:43:32] - INFO: Epoch: 2, Batch[92/227], Train loss :1.878, Train acc: 0.6046643913538111
[2024-12-24 17:43:35] - INFO: Epoch: 2, Batch[93/227], Train loss :1.831, Train acc: 0.6234972677595628
[2024-12-24 17:43:37] - INFO: Epoch: 2, Batch[94/227], Train loss :1.969, Train acc: 0.596723044397463
[2024-12-24 17:43:40] - INFO: Epoch: 2, Batch[95/227], Train loss :1.861, Train acc: 0.5996543778801844
[2024-12-24 17:43:41] - INFO: Epoch: 2, Batch[96/227], Train loss :1.948, Train acc: 0.5891171571349251
[2024-12-24 17:43:44] - INFO: Epoch: 2, Batch[97/227], Train loss :1.843, Train acc: 0.6197718631178707
[2024-12-24 17:43:45] - INFO: Epoch: 2, Batch[98/227], Train loss :1.790, Train acc: 0.6256267409470752
[2024-12-24 17:43:47] - INFO: Epoch: 2, Batch[99/227], Train loss :1.865, Train acc: 0.621160409556314
[2024-12-24 17:43:50] - INFO: Epoch: 2, Batch[100/227], Train loss :1.919, Train acc: 0.6119648737650933
[2024-12-24 17:43:52] - INFO: Epoch: 2, Batch[101/227], Train loss :1.783, Train acc: 0.6228668941979523
[2024-12-24 17:43:54] - INFO: Epoch: 2, Batch[102/227], Train loss :1.847, Train acc: 0.6093660765276985
[2024-12-24 17:43:56] - INFO: Epoch: 2, Batch[103/227], Train loss :1.864, Train acc: 0.6225988700564972
[2024-12-24 17:43:59] - INFO: Epoch: 2, Batch[104/227], Train loss :1.910, Train acc: 0.6023102310231023
[2024-12-24 17:44:01] - INFO: Epoch: 2, Batch[105/227], Train loss :1.894, Train acc: 0.6028681742967458
[2024-12-24 17:44:04] - INFO: Epoch: 2, Batch[106/227], Train loss :1.939, Train acc: 0.604052573932092
[2024-12-24 17:44:06] - INFO: Epoch: 2, Batch[107/227], Train loss :2.039, Train acc: 0.594075699396599
[2024-12-24 17:44:08] - INFO: Epoch: 2, Batch[108/227], Train loss :1.822, Train acc: 0.62046783625731
[2024-12-24 17:44:11] - INFO: Epoch: 2, Batch[109/227], Train loss :1.917, Train acc: 0.5950323974082073
[2024-12-24 17:44:13] - INFO: Epoch: 2, Batch[110/227], Train loss :1.849, Train acc: 0.608282036933408
[2024-12-24 17:44:15] - INFO: Epoch: 2, Batch[111/227], Train loss :1.918, Train acc: 0.6086009798584648
[2024-12-24 17:44:17] - INFO: Epoch: 2, Batch[112/227], Train loss :1.902, Train acc: 0.6013179571663921
[2024-12-24 17:44:20] - INFO: Epoch: 2, Batch[113/227], Train loss :1.908, Train acc: 0.6101223581757509
[2024-12-24 17:44:23] - INFO: Epoch: 2, Batch[114/227], Train loss :1.956, Train acc: 0.6035313001605136
[2024-12-24 17:44:25] - INFO: Epoch: 2, Batch[115/227], Train loss :1.994, Train acc: 0.5996740901683868
[2024-12-24 17:44:27] - INFO: Epoch: 2, Batch[116/227], Train loss :1.933, Train acc: 0.5930416447021613
[2024-12-24 17:44:29] - INFO: Epoch: 2, Batch[117/227], Train loss :1.773, Train acc: 0.6127964699393271
[2024-12-24 17:44:32] - INFO: Epoch: 2, Batch[118/227], Train loss :2.010, Train acc: 0.6
[2024-12-24 17:44:34] - INFO: Epoch: 2, Batch[119/227], Train loss :2.018, Train acc: 0.5857220118983234
[2024-12-24 17:44:36] - INFO: Epoch: 2, Batch[120/227], Train loss :1.891, Train acc: 0.6096196868008948
[2024-12-24 17:44:40] - INFO: Epoch: 2, Batch[121/227], Train loss :1.885, Train acc: 0.6212471131639723
[2024-12-24 17:44:42] - INFO: Epoch: 2, Batch[122/227], Train loss :1.708, Train acc: 0.6382737081203862
[2024-12-24 17:44:43] - INFO: Epoch: 2, Batch[123/227], Train loss :1.880, Train acc: 0.6077441077441077
[2024-12-24 17:44:46] - INFO: Epoch: 2, Batch[124/227], Train loss :1.912, Train acc: 0.6026860660324567
[2024-12-24 17:44:47] - INFO: Epoch: 2, Batch[125/227], Train loss :1.779, Train acc: 0.621465666474322
[2024-12-24 17:44:50] - INFO: Epoch: 2, Batch[126/227], Train loss :1.943, Train acc: 0.6018671059857221
[2024-12-24 17:44:53] - INFO: Epoch: 2, Batch[127/227], Train loss :1.870, Train acc: 0.6097826086956522
[2024-12-24 17:44:55] - INFO: Epoch: 2, Batch[128/227], Train loss :2.004, Train acc: 0.6027777777777777
[2024-12-24 17:44:58] - INFO: Epoch: 2, Batch[129/227], Train loss :1.809, Train acc: 0.609967497291441
[2024-12-24 17:45:00] - INFO: Epoch: 2, Batch[130/227], Train loss :1.930, Train acc: 0.6058791507893304
[2024-12-24 17:45:02] - INFO: Epoch: 2, Batch[131/227], Train loss :1.900, Train acc: 0.6038251366120219
[2024-12-24 17:45:04] - INFO: Epoch: 2, Batch[132/227], Train loss :1.800, Train acc: 0.628393665158371
[2024-12-24 17:45:06] - INFO: Epoch: 2, Batch[133/227], Train loss :1.995, Train acc: 0.5913621262458472
[2024-12-24 17:45:09] - INFO: Epoch: 2, Batch[134/227], Train loss :1.795, Train acc: 0.629862700228833
[2024-12-24 17:45:11] - INFO: Epoch: 2, Batch[135/227], Train loss :1.965, Train acc: 0.6068965517241379
[2024-12-24 17:45:13] - INFO: Epoch: 2, Batch[136/227], Train loss :1.965, Train acc: 0.5959206174200662
[2024-12-24 17:45:16] - INFO: Epoch: 2, Batch[137/227], Train loss :1.993, Train acc: 0.5964125560538116
[2024-12-24 17:45:18] - INFO: Epoch: 2, Batch[138/227], Train loss :1.912, Train acc: 0.6004390779363337
[2024-12-24 17:45:21] - INFO: Epoch: 2, Batch[139/227], Train loss :1.915, Train acc: 0.6112938596491229
[2024-12-24 17:45:23] - INFO: Epoch: 2, Batch[140/227], Train loss :1.858, Train acc: 0.6226826608505998
[2024-12-24 17:45:25] - INFO: Epoch: 2, Batch[141/227], Train loss :1.862, Train acc: 0.6145059965733867
[2024-12-24 17:45:27] - INFO: Epoch: 2, Batch[142/227], Train loss :1.889, Train acc: 0.5931642778390298
[2024-12-24 17:45:30] - INFO: Epoch: 2, Batch[143/227], Train loss :1.813, Train acc: 0.6280454791553871
[2024-12-24 17:45:31] - INFO: Epoch: 2, Batch[144/227], Train loss :1.895, Train acc: 0.6023622047244095
[2024-12-24 17:45:34] - INFO: Epoch: 2, Batch[145/227], Train loss :2.041, Train acc: 0.5846560846560847
[2024-12-24 17:45:36] - INFO: Epoch: 2, Batch[146/227], Train loss :1.857, Train acc: 0.6088154269972452
[2024-12-24 17:45:38] - INFO: Epoch: 2, Batch[147/227], Train loss :1.941, Train acc: 0.5925333333333334
[2024-12-24 17:45:40] - INFO: Epoch: 2, Batch[148/227], Train loss :1.849, Train acc: 0.6311383928571429
[2024-12-24 17:45:42] - INFO: Epoch: 2, Batch[149/227], Train loss :1.888, Train acc: 0.6095457159286947
[2024-12-24 17:45:43] - INFO: Epoch: 2, Batch[150/227], Train loss :1.844, Train acc: 0.6091825307950728
[2024-12-24 17:45:45] - INFO: Epoch: 2, Batch[151/227], Train loss :2.018, Train acc: 0.5967302452316077
[2024-12-24 17:45:48] - INFO: Epoch: 2, Batch[152/227], Train loss :1.872, Train acc: 0.6007950028392959
[2024-12-24 17:45:50] - INFO: Epoch: 2, Batch[153/227], Train loss :2.102, Train acc: 0.5797958087049974
[2024-12-24 17:45:53] - INFO: Epoch: 2, Batch[154/227], Train loss :1.848, Train acc: 0.6180257510729614
[2024-12-24 17:45:54] - INFO: Epoch: 2, Batch[155/227], Train loss :1.797, Train acc: 0.6318711826762909
[2024-12-24 17:45:57] - INFO: Epoch: 2, Batch[156/227], Train loss :1.960, Train acc: 0.6004415011037527
[2024-12-24 17:45:59] - INFO: Epoch: 2, Batch[157/227], Train loss :1.857, Train acc: 0.6047565118912798
[2024-12-24 17:46:01] - INFO: Epoch: 2, Batch[158/227], Train loss :1.949, Train acc: 0.5967470555243971
[2024-12-24 17:46:03] - INFO: Epoch: 2, Batch[159/227], Train loss :1.971, Train acc: 0.6114754098360655
[2024-12-24 17:46:05] - INFO: Epoch: 2, Batch[160/227], Train loss :1.942, Train acc: 0.6063249727371864
[2024-12-24 17:46:07] - INFO: Epoch: 2, Batch[161/227], Train loss :1.955, Train acc: 0.6002239641657335
[2024-12-24 17:46:10] - INFO: Epoch: 2, Batch[162/227], Train loss :1.888, Train acc: 0.606312292358804
[2024-12-24 17:46:12] - INFO: Epoch: 2, Batch[163/227], Train loss :1.840, Train acc: 0.6077257889009793
[2024-12-24 17:46:14] - INFO: Epoch: 2, Batch[164/227], Train loss :1.888, Train acc: 0.6008515167642363
[2024-12-24 17:46:17] - INFO: Epoch: 2, Batch[165/227], Train loss :1.826, Train acc: 0.6185682326621924
[2024-12-24 17:46:19] - INFO: Epoch: 2, Batch[166/227], Train loss :1.861, Train acc: 0.6076294277929155
[2024-12-24 17:46:22] - INFO: Epoch: 2, Batch[167/227], Train loss :2.049, Train acc: 0.5812143227815257
[2024-12-24 17:46:24] - INFO: Epoch: 2, Batch[168/227], Train loss :2.011, Train acc: 0.5951192457016085
[2024-12-24 17:46:26] - INFO: Epoch: 2, Batch[169/227], Train loss :1.825, Train acc: 0.6192893401015228
[2024-12-24 17:46:28] - INFO: Epoch: 2, Batch[170/227], Train loss :1.918, Train acc: 0.6021917808219178
[2024-12-24 17:46:30] - INFO: Epoch: 2, Batch[171/227], Train loss :1.807, Train acc: 0.6304347826086957
[2024-12-24 17:46:32] - INFO: Epoch: 2, Batch[172/227], Train loss :1.843, Train acc: 0.6124927703875073
[2024-12-24 17:46:35] - INFO: Epoch: 2, Batch[173/227], Train loss :1.916, Train acc: 0.5956618464961068
[2024-12-24 17:46:37] - INFO: Epoch: 2, Batch[174/227], Train loss :1.924, Train acc: 0.6010899182561308
[2024-12-24 17:46:40] - INFO: Epoch: 2, Batch[175/227], Train loss :1.987, Train acc: 0.5993589743589743
[2024-12-24 17:46:43] - INFO: Epoch: 2, Batch[176/227], Train loss :1.849, Train acc: 0.62109375
[2024-12-24 17:46:44] - INFO: Epoch: 2, Batch[177/227], Train loss :1.890, Train acc: 0.6274397244546498
[2024-12-24 17:46:47] - INFO: Epoch: 2, Batch[178/227], Train loss :1.868, Train acc: 0.6173108328796951
[2024-12-24 17:46:49] - INFO: Epoch: 2, Batch[179/227], Train loss :1.922, Train acc: 0.6046128500823723
[2024-12-24 17:46:52] - INFO: Epoch: 2, Batch[180/227], Train loss :1.872, Train acc: 0.6150442477876106
[2024-12-24 17:46:54] - INFO: Epoch: 2, Batch[181/227], Train loss :1.859, Train acc: 0.6186813186813187
[2024-12-24 17:46:56] - INFO: Epoch: 2, Batch[182/227], Train loss :1.868, Train acc: 0.6080316742081447
[2024-12-24 17:46:58] - INFO: Epoch: 2, Batch[183/227], Train loss :1.812, Train acc: 0.6267487409065473
[2024-12-24 17:47:00] - INFO: Epoch: 2, Batch[184/227], Train loss :1.982, Train acc: 0.5951192457016085
[2024-12-24 17:47:02] - INFO: Epoch: 2, Batch[185/227], Train loss :2.026, Train acc: 0.5901551631888711
[2024-12-24 17:47:05] - INFO: Epoch: 2, Batch[186/227], Train loss :2.013, Train acc: 0.5907626208378088
[2024-12-24 17:47:08] - INFO: Epoch: 2, Batch[187/227], Train loss :1.984, Train acc: 0.6173669467787115
[2024-12-24 17:47:11] - INFO: Epoch: 2, Batch[188/227], Train loss :1.966, Train acc: 0.5924932975871313
[2024-12-24 17:47:13] - INFO: Epoch: 2, Batch[189/227], Train loss :1.970, Train acc: 0.5952773201537617
[2024-12-24 17:47:15] - INFO: Epoch: 2, Batch[190/227], Train loss :2.007, Train acc: 0.590027700831025
[2024-12-24 17:47:17] - INFO: Epoch: 2, Batch[191/227], Train loss :1.899, Train acc: 0.6027173913043479
[2024-12-24 17:47:20] - INFO: Epoch: 2, Batch[192/227], Train loss :1.956, Train acc: 0.5954113038612199
[2024-12-24 17:47:23] - INFO: Epoch: 2, Batch[193/227], Train loss :1.791, Train acc: 0.6191478169384534
[2024-12-24 17:47:26] - INFO: Epoch: 2, Batch[194/227], Train loss :1.917, Train acc: 0.5991091314031181
[2024-12-24 17:47:27] - INFO: Epoch: 2, Batch[195/227], Train loss :1.820, Train acc: 0.62248322147651
[2024-12-24 17:47:30] - INFO: Epoch: 2, Batch[196/227], Train loss :2.062, Train acc: 0.5828288387451843
[2024-12-24 17:47:32] - INFO: Epoch: 2, Batch[197/227], Train loss :1.867, Train acc: 0.6040697674418605
[2024-12-24 17:47:34] - INFO: Epoch: 2, Batch[198/227], Train loss :1.896, Train acc: 0.5981152993348116
[2024-12-24 17:47:36] - INFO: Epoch: 2, Batch[199/227], Train loss :1.861, Train acc: 0.6006600660066007
[2024-12-24 17:47:40] - INFO: Epoch: 2, Batch[200/227], Train loss :1.822, Train acc: 0.6128133704735376
[2024-12-24 17:47:42] - INFO: Epoch: 2, Batch[201/227], Train loss :1.952, Train acc: 0.5890557939914163
[2024-12-24 17:47:44] - INFO: Epoch: 2, Batch[202/227], Train loss :1.937, Train acc: 0.602017937219731
[2024-12-24 17:47:46] - INFO: Epoch: 2, Batch[203/227], Train loss :1.952, Train acc: 0.5928652321630804
[2024-12-24 17:47:48] - INFO: Epoch: 2, Batch[204/227], Train loss :1.998, Train acc: 0.5935656836461126
[2024-12-24 17:47:50] - INFO: Epoch: 2, Batch[205/227], Train loss :1.791, Train acc: 0.6199220923761826
[2024-12-24 17:47:52] - INFO: Epoch: 2, Batch[206/227], Train loss :1.736, Train acc: 0.642068564787914
[2024-12-24 17:47:54] - INFO: Epoch: 2, Batch[207/227], Train loss :1.785, Train acc: 0.6211143695014663
[2024-12-24 17:47:57] - INFO: Epoch: 2, Batch[208/227], Train loss :2.017, Train acc: 0.5950684931506849
[2024-12-24 17:47:58] - INFO: Epoch: 2, Batch[209/227], Train loss :1.822, Train acc: 0.6133333333333333
[2024-12-24 17:48:01] - INFO: Epoch: 2, Batch[210/227], Train loss :1.863, Train acc: 0.624225352112676
[2024-12-24 17:48:03] - INFO: Epoch: 2, Batch[211/227], Train loss :1.923, Train acc: 0.5975215517241379
[2024-12-24 17:48:05] - INFO: Epoch: 2, Batch[212/227], Train loss :1.739, Train acc: 0.6398860398860399
[2024-12-24 17:48:07] - INFO: Epoch: 2, Batch[213/227], Train loss :1.929, Train acc: 0.6047297297297297
[2024-12-24 17:48:10] - INFO: Epoch: 2, Batch[214/227], Train loss :1.808, Train acc: 0.6236914600550965
[2024-12-24 17:48:11] - INFO: Epoch: 2, Batch[215/227], Train loss :1.971, Train acc: 0.5869209809264305
[2024-12-24 17:48:13] - INFO: Epoch: 2, Batch[216/227], Train loss :1.953, Train acc: 0.5912568306010929
[2024-12-24 17:48:15] - INFO: Epoch: 2, Batch[217/227], Train loss :1.871, Train acc: 0.6109558412520961
[2024-12-24 17:48:17] - INFO: Epoch: 2, Batch[218/227], Train loss :1.909, Train acc: 0.6087662337662337
[2024-12-24 17:48:19] - INFO: Epoch: 2, Batch[219/227], Train loss :1.817, Train acc: 0.6097560975609756
[2024-12-24 17:48:22] - INFO: Epoch: 2, Batch[220/227], Train loss :1.933, Train acc: 0.6014823261117446
[2024-12-24 17:48:24] - INFO: Epoch: 2, Batch[221/227], Train loss :1.833, Train acc: 0.6186770428015564
[2024-12-24 17:48:26] - INFO: Epoch: 2, Batch[222/227], Train loss :1.851, Train acc: 0.6056105610561056
[2024-12-24 17:48:28] - INFO: Epoch: 2, Batch[223/227], Train loss :1.923, Train acc: 0.6030491247882552
[2024-12-24 17:48:30] - INFO: Epoch: 2, Batch[224/227], Train loss :1.983, Train acc: 0.5979492714517
[2024-12-24 17:48:32] - INFO: Epoch: 2, Batch[225/227], Train loss :1.931, Train acc: 0.6050420168067226
[2024-12-24 17:48:33] - INFO: Epoch: 2, Batch[226/227], Train loss :1.968, Train acc: 0.6063303659742829
[2024-12-24 17:48:33] - INFO: Epoch: 2, Train loss: 1.889, Epoch time = 509.883s
[2024-12-24 17:48:36] - INFO: Epoch: 3, Batch[0/227], Train loss :1.779, Train acc: 0.6289237668161435
[2024-12-24 17:48:39] - INFO: Epoch: 3, Batch[1/227], Train loss :1.929, Train acc: 0.6009825327510917
[2024-12-24 17:48:41] - INFO: Epoch: 3, Batch[2/227], Train loss :1.794, Train acc: 0.5956043956043956
[2024-12-24 17:48:43] - INFO: Epoch: 3, Batch[3/227], Train loss :1.764, Train acc: 0.6365105008077544
[2024-12-24 17:48:45] - INFO: Epoch: 3, Batch[4/227], Train loss :1.735, Train acc: 0.6229872293170461
[2024-12-24 17:48:47] - INFO: Epoch: 3, Batch[5/227], Train loss :1.728, Train acc: 0.6342301087578707
[2024-12-24 17:48:51] - INFO: Epoch: 3, Batch[6/227], Train loss :1.802, Train acc: 0.6287323943661972
[2024-12-24 17:48:53] - INFO: Epoch: 3, Batch[7/227], Train loss :1.847, Train acc: 0.609055770292656
[2024-12-24 17:48:56] - INFO: Epoch: 3, Batch[8/227], Train loss :1.842, Train acc: 0.6075880758807588
[2024-12-24 17:48:58] - INFO: Epoch: 3, Batch[9/227], Train loss :1.824, Train acc: 0.6304470854555744
[2024-12-24 17:49:00] - INFO: Epoch: 3, Batch[10/227], Train loss :1.803, Train acc: 0.6135105204872646
[2024-12-24 17:49:03] - INFO: Epoch: 3, Batch[11/227], Train loss :1.752, Train acc: 0.6316989737742303
[2024-12-24 17:49:05] - INFO: Epoch: 3, Batch[12/227], Train loss :1.779, Train acc: 0.6258992805755396
[2024-12-24 17:49:07] - INFO: Epoch: 3, Batch[13/227], Train loss :1.853, Train acc: 0.6095890410958904
[2024-12-24 17:49:10] - INFO: Epoch: 3, Batch[14/227], Train loss :1.853, Train acc: 0.6218487394957983
[2024-12-24 17:49:11] - INFO: Epoch: 3, Batch[15/227], Train loss :1.872, Train acc: 0.6034968979131415
[2024-12-24 17:49:13] - INFO: Epoch: 3, Batch[16/227], Train loss :1.762, Train acc: 0.6373873873873874
[2024-12-24 17:49:15] - INFO: Epoch: 3, Batch[17/227], Train loss :1.930, Train acc: 0.6062639821029083
[2024-12-24 17:49:17] - INFO: Epoch: 3, Batch[18/227], Train loss :1.734, Train acc: 0.6291028446389497
[2024-12-24 17:49:20] - INFO: Epoch: 3, Batch[19/227], Train loss :1.858, Train acc: 0.6263548203080433
[2024-12-24 17:49:23] - INFO: Epoch: 3, Batch[20/227], Train loss :1.907, Train acc: 0.601184068891281
[2024-12-24 17:49:25] - INFO: Epoch: 3, Batch[21/227], Train loss :1.793, Train acc: 0.6197802197802198
[2024-12-24 17:49:27] - INFO: Epoch: 3, Batch[22/227], Train loss :1.862, Train acc: 0.6059479553903345
[2024-12-24 17:49:29] - INFO: Epoch: 3, Batch[23/227], Train loss :1.746, Train acc: 0.639751552795031
[2024-12-24 17:49:31] - INFO: Epoch: 3, Batch[24/227], Train loss :1.881, Train acc: 0.6182224706539966
[2024-12-24 17:49:34] - INFO: Epoch: 3, Batch[25/227], Train loss :1.791, Train acc: 0.623342175066313
[2024-12-24 17:49:37] - INFO: Epoch: 3, Batch[26/227], Train loss :1.907, Train acc: 0.6167048054919908
[2024-12-24 17:49:39] - INFO: Epoch: 3, Batch[27/227], Train loss :1.836, Train acc: 0.6183333333333333
[2024-12-24 17:49:41] - INFO: Epoch: 3, Batch[28/227], Train loss :1.759, Train acc: 0.6287964004499438
[2024-12-24 17:49:43] - INFO: Epoch: 3, Batch[29/227], Train loss :1.796, Train acc: 0.6238279095421952
[2024-12-24 17:49:46] - INFO: Epoch: 3, Batch[30/227], Train loss :1.820, Train acc: 0.6184649610678532
[2024-12-24 17:49:47] - INFO: Epoch: 3, Batch[31/227], Train loss :1.820, Train acc: 0.6090808416389811
[2024-12-24 17:49:50] - INFO: Epoch: 3, Batch[32/227], Train loss :1.759, Train acc: 0.634090909090909
[2024-12-24 17:49:53] - INFO: Epoch: 3, Batch[33/227], Train loss :1.740, Train acc: 0.6299885974914481
[2024-12-24 17:49:55] - INFO: Epoch: 3, Batch[34/227], Train loss :1.867, Train acc: 0.6103312745648513
[2024-12-24 17:49:57] - INFO: Epoch: 3, Batch[35/227], Train loss :1.769, Train acc: 0.6290322580645161
[2024-12-24 17:50:00] - INFO: Epoch: 3, Batch[36/227], Train loss :1.863, Train acc: 0.6066481994459834
[2024-12-24 17:50:02] - INFO: Epoch: 3, Batch[37/227], Train loss :1.818, Train acc: 0.6318711826762909
[2024-12-24 17:50:04] - INFO: Epoch: 3, Batch[38/227], Train loss :1.759, Train acc: 0.6276243093922652
[2024-12-24 17:50:06] - INFO: Epoch: 3, Batch[39/227], Train loss :1.768, Train acc: 0.6381909547738693
[2024-12-24 17:50:09] - INFO: Epoch: 3, Batch[40/227], Train loss :1.806, Train acc: 0.6250696378830084
[2024-12-24 17:50:11] - INFO: Epoch: 3, Batch[41/227], Train loss :1.709, Train acc: 0.6496390893947807
[2024-12-24 17:50:13] - INFO: Epoch: 3, Batch[42/227], Train loss :1.766, Train acc: 0.6281087333718912
[2024-12-24 17:50:15] - INFO: Epoch: 3, Batch[43/227], Train loss :1.879, Train acc: 0.5979381443298969
[2024-12-24 17:50:17] - INFO: Epoch: 3, Batch[44/227], Train loss :1.786, Train acc: 0.618510158013544
[2024-12-24 17:50:20] - INFO: Epoch: 3, Batch[45/227], Train loss :1.819, Train acc: 0.6071428571428571
[2024-12-24 17:50:22] - INFO: Epoch: 3, Batch[46/227], Train loss :1.856, Train acc: 0.6324595649749024
[2024-12-24 17:50:24] - INFO: Epoch: 3, Batch[47/227], Train loss :1.860, Train acc: 0.6071428571428571
[2024-12-24 17:50:26] - INFO: Epoch: 3, Batch[48/227], Train loss :1.915, Train acc: 0.6076839826839827
[2024-12-24 17:50:28] - INFO: Epoch: 3, Batch[49/227], Train loss :1.735, Train acc: 0.6366630076838639
[2024-12-24 17:50:30] - INFO: Epoch: 3, Batch[50/227], Train loss :1.836, Train acc: 0.6220604703247481
[2024-12-24 17:50:32] - INFO: Epoch: 3, Batch[51/227], Train loss :1.690, Train acc: 0.6276889134031991
[2024-12-24 17:50:35] - INFO: Epoch: 3, Batch[52/227], Train loss :1.921, Train acc: 0.6055798687089715
[2024-12-24 17:50:37] - INFO: Epoch: 3, Batch[53/227], Train loss :1.854, Train acc: 0.609811751283514
[2024-12-24 17:50:40] - INFO: Epoch: 3, Batch[54/227], Train loss :1.845, Train acc: 0.6178378378378379
[2024-12-24 17:50:42] - INFO: Epoch: 3, Batch[55/227], Train loss :1.826, Train acc: 0.6036816459122902
[2024-12-24 17:50:44] - INFO: Epoch: 3, Batch[56/227], Train loss :1.745, Train acc: 0.6334246575342466
[2024-12-24 17:50:46] - INFO: Epoch: 3, Batch[57/227], Train loss :1.778, Train acc: 0.6303854875283447
[2024-12-24 17:50:49] - INFO: Epoch: 3, Batch[58/227], Train loss :1.937, Train acc: 0.594210813762971
[2024-12-24 17:50:52] - INFO: Epoch: 3, Batch[59/227], Train loss :1.769, Train acc: 0.6181719848566792
[2024-12-24 17:50:54] - INFO: Epoch: 3, Batch[60/227], Train loss :1.930, Train acc: 0.6057906458797327
[2024-12-24 17:50:56] - INFO: Epoch: 3, Batch[61/227], Train loss :1.911, Train acc: 0.6036892118501956
[2024-12-24 17:50:58] - INFO: Epoch: 3, Batch[62/227], Train loss :1.901, Train acc: 0.6107456140350878
[2024-12-24 17:51:00] - INFO: Epoch: 3, Batch[63/227], Train loss :1.701, Train acc: 0.6304464766003227
[2024-12-24 17:51:02] - INFO: Epoch: 3, Batch[64/227], Train loss :1.896, Train acc: 0.6142301278488049
[2024-12-24 17:51:04] - INFO: Epoch: 3, Batch[65/227], Train loss :1.938, Train acc: 0.6035940803382663
[2024-12-24 17:51:06] - INFO: Epoch: 3, Batch[66/227], Train loss :1.620, Train acc: 0.6575963718820862
[2024-12-24 17:51:08] - INFO: Epoch: 3, Batch[67/227], Train loss :1.805, Train acc: 0.6231884057971014
[2024-12-24 17:51:11] - INFO: Epoch: 3, Batch[68/227], Train loss :1.827, Train acc: 0.6081009994739611
[2024-12-24 17:51:13] - INFO: Epoch: 3, Batch[69/227], Train loss :1.822, Train acc: 0.6158503026967529
[2024-12-24 17:51:16] - INFO: Epoch: 3, Batch[70/227], Train loss :1.705, Train acc: 0.6264929424538545
[2024-12-24 17:51:18] - INFO: Epoch: 3, Batch[71/227], Train loss :1.879, Train acc: 0.6107123136388736
[2024-12-24 17:51:20] - INFO: Epoch: 3, Batch[72/227], Train loss :1.611, Train acc: 0.64472190692395
[2024-12-24 17:51:22] - INFO: Epoch: 3, Batch[73/227], Train loss :1.703, Train acc: 0.63948973932335
[2024-12-24 17:51:24] - INFO: Epoch: 3, Batch[74/227], Train loss :1.690, Train acc: 0.6384742951907131
[2024-12-24 17:51:26] - INFO: Epoch: 3, Batch[75/227], Train loss :1.700, Train acc: 0.6450719822812846
[2024-12-24 17:51:28] - INFO: Epoch: 3, Batch[76/227], Train loss :1.876, Train acc: 0.6091205211726385
[2024-12-24 17:51:30] - INFO: Epoch: 3, Batch[77/227], Train loss :1.774, Train acc: 0.6191275167785235
[2024-12-24 17:51:32] - INFO: Epoch: 3, Batch[78/227], Train loss :1.834, Train acc: 0.6244318181818181
[2024-12-24 17:51:34] - INFO: Epoch: 3, Batch[79/227], Train loss :1.747, Train acc: 0.6260387811634349
[2024-12-24 17:51:37] - INFO: Epoch: 3, Batch[80/227], Train loss :1.896, Train acc: 0.609106052193226
[2024-12-24 17:51:39] - INFO: Epoch: 3, Batch[81/227], Train loss :1.814, Train acc: 0.6081903707802988
[2024-12-24 17:51:41] - INFO: Epoch: 3, Batch[82/227], Train loss :1.838, Train acc: 0.6185002736726875
[2024-12-24 17:51:43] - INFO: Epoch: 3, Batch[83/227], Train loss :1.832, Train acc: 0.6121313299944352
[2024-12-24 17:51:45] - INFO: Epoch: 3, Batch[84/227], Train loss :1.783, Train acc: 0.6168067226890757
[2024-12-24 17:51:47] - INFO: Epoch: 3, Batch[85/227], Train loss :1.749, Train acc: 0.6416093170989942
[2024-12-24 17:51:50] - INFO: Epoch: 3, Batch[86/227], Train loss :1.819, Train acc: 0.6170921198668147
[2024-12-24 17:51:53] - INFO: Epoch: 3, Batch[87/227], Train loss :1.864, Train acc: 0.6145038167938931
[2024-12-24 17:51:55] - INFO: Epoch: 3, Batch[88/227], Train loss :1.805, Train acc: 0.6204379562043796
[2024-12-24 17:51:57] - INFO: Epoch: 3, Batch[89/227], Train loss :1.828, Train acc: 0.6198257080610022
[2024-12-24 17:51:59] - INFO: Epoch: 3, Batch[90/227], Train loss :1.743, Train acc: 0.6359060402684564
[2024-12-24 17:52:01] - INFO: Epoch: 3, Batch[91/227], Train loss :1.785, Train acc: 0.6145598194130926
[2024-12-24 17:52:03] - INFO: Epoch: 3, Batch[92/227], Train loss :1.722, Train acc: 0.6219382321618744
[2024-12-24 17:52:05] - INFO: Epoch: 3, Batch[93/227], Train loss :1.715, Train acc: 0.636881047239613
[2024-12-24 17:52:07] - INFO: Epoch: 3, Batch[94/227], Train loss :1.891, Train acc: 0.6062948647156268
[2024-12-24 17:52:10] - INFO: Epoch: 3, Batch[95/227], Train loss :1.840, Train acc: 0.6161952301719357
[2024-12-24 17:52:13] - INFO: Epoch: 3, Batch[96/227], Train loss :1.894, Train acc: 0.6040156162855549
[2024-12-24 17:52:16] - INFO: Epoch: 3, Batch[97/227], Train loss :1.855, Train acc: 0.6104868913857678
[2024-12-24 17:52:18] - INFO: Epoch: 3, Batch[98/227], Train loss :1.800, Train acc: 0.6385331781140862
[2024-12-24 17:52:21] - INFO: Epoch: 3, Batch[99/227], Train loss :1.815, Train acc: 0.6229685807150596
[2024-12-24 17:52:23] - INFO: Epoch: 3, Batch[100/227], Train loss :1.972, Train acc: 0.5965281430825881
[2024-12-24 17:52:25] - INFO: Epoch: 3, Batch[101/227], Train loss :1.815, Train acc: 0.6200227531285551
[2024-12-24 17:52:27] - INFO: Epoch: 3, Batch[102/227], Train loss :1.816, Train acc: 0.6141425389755011
[2024-12-24 17:52:28] - INFO: Epoch: 3, Batch[103/227], Train loss :1.823, Train acc: 0.6226203807390818
[2024-12-24 17:52:31] - INFO: Epoch: 3, Batch[104/227], Train loss :1.939, Train acc: 0.6145945945945946
[2024-12-24 17:52:33] - INFO: Epoch: 3, Batch[105/227], Train loss :1.803, Train acc: 0.6139101861993428
[2024-12-24 17:52:35] - INFO: Epoch: 3, Batch[106/227], Train loss :1.786, Train acc: 0.6334440753045404
[2024-12-24 17:52:38] - INFO: Epoch: 3, Batch[107/227], Train loss :1.797, Train acc: 0.6114318706697459
[2024-12-24 17:52:40] - INFO: Epoch: 3, Batch[108/227], Train loss :1.826, Train acc: 0.6186487995533222
[2024-12-24 17:52:42] - INFO: Epoch: 3, Batch[109/227], Train loss :1.771, Train acc: 0.621606334841629
[2024-12-24 17:52:44] - INFO: Epoch: 3, Batch[110/227], Train loss :1.827, Train acc: 0.6047278724573941
[2024-12-24 17:52:46] - INFO: Epoch: 3, Batch[111/227], Train loss :1.917, Train acc: 0.5987825124515772
[2024-12-24 17:52:48] - INFO: Epoch: 3, Batch[112/227], Train loss :1.793, Train acc: 0.6196487376509331
[2024-12-24 17:52:51] - INFO: Epoch: 3, Batch[113/227], Train loss :1.734, Train acc: 0.6265536723163841
[2024-12-24 17:52:53] - INFO: Epoch: 3, Batch[114/227], Train loss :1.780, Train acc: 0.6187535092644582
[2024-12-24 17:52:56] - INFO: Epoch: 3, Batch[115/227], Train loss :1.907, Train acc: 0.6018569087930092
[2024-12-24 17:52:58] - INFO: Epoch: 3, Batch[116/227], Train loss :1.897, Train acc: 0.6026200873362445
[2024-12-24 17:53:00] - INFO: Epoch: 3, Batch[117/227], Train loss :1.842, Train acc: 0.6008381351492929
[2024-12-24 17:53:02] - INFO: Epoch: 3, Batch[118/227], Train loss :1.772, Train acc: 0.6297943301834352
[2024-12-24 17:53:04] - INFO: Epoch: 3, Batch[119/227], Train loss :1.712, Train acc: 0.6355353075170843
[2024-12-24 17:53:06] - INFO: Epoch: 3, Batch[120/227], Train loss :1.780, Train acc: 0.629757785467128
[2024-12-24 17:53:08] - INFO: Epoch: 3, Batch[121/227], Train loss :1.876, Train acc: 0.5915655690352397
[2024-12-24 17:53:10] - INFO: Epoch: 3, Batch[122/227], Train loss :1.794, Train acc: 0.6227678571428571
[2024-12-24 17:53:13] - INFO: Epoch: 3, Batch[123/227], Train loss :1.813, Train acc: 0.6198439241917503
[2024-12-24 17:53:15] - INFO: Epoch: 3, Batch[124/227], Train loss :1.733, Train acc: 0.6336796063422635
[2024-12-24 17:53:17] - INFO: Epoch: 3, Batch[125/227], Train loss :1.912, Train acc: 0.6157253599114064
[2024-12-24 17:53:20] - INFO: Epoch: 3, Batch[126/227], Train loss :1.823, Train acc: 0.5972927241962775
[2024-12-24 17:53:22] - INFO: Epoch: 3, Batch[127/227], Train loss :1.905, Train acc: 0.6065217391304348
[2024-12-24 17:53:24] - INFO: Epoch: 3, Batch[128/227], Train loss :1.934, Train acc: 0.5901098901098901
[2024-12-24 17:53:26] - INFO: Epoch: 3, Batch[129/227], Train loss :1.748, Train acc: 0.6273041474654378
[2024-12-24 17:53:28] - INFO: Epoch: 3, Batch[130/227], Train loss :1.846, Train acc: 0.6243654822335025
[2024-12-24 17:53:30] - INFO: Epoch: 3, Batch[131/227], Train loss :1.840, Train acc: 0.6172701949860724
[2024-12-24 17:53:32] - INFO: Epoch: 3, Batch[132/227], Train loss :1.722, Train acc: 0.6178307779670642
[2024-12-24 17:53:34] - INFO: Epoch: 3, Batch[133/227], Train loss :1.802, Train acc: 0.6260434056761269
[2024-12-24 17:53:36] - INFO: Epoch: 3, Batch[134/227], Train loss :1.835, Train acc: 0.5946696279844531
[2024-12-24 17:53:38] - INFO: Epoch: 3, Batch[135/227], Train loss :1.970, Train acc: 0.5945330296127562
[2024-12-24 17:53:40] - INFO: Epoch: 3, Batch[136/227], Train loss :1.829, Train acc: 0.6182505399568035
[2024-12-24 17:53:42] - INFO: Epoch: 3, Batch[137/227], Train loss :1.793, Train acc: 0.6156840934371524
[2024-12-24 17:53:45] - INFO: Epoch: 3, Batch[138/227], Train loss :1.785, Train acc: 0.6196013289036545
[2024-12-24 17:53:47] - INFO: Epoch: 3, Batch[139/227], Train loss :1.730, Train acc: 0.6435754189944134
[2024-12-24 17:53:51] - INFO: Epoch: 3, Batch[140/227], Train loss :1.809, Train acc: 0.6183783783783784
[2024-12-24 17:53:53] - INFO: Epoch: 3, Batch[141/227], Train loss :1.778, Train acc: 0.6100558659217877
[2024-12-24 17:53:55] - INFO: Epoch: 3, Batch[142/227], Train loss :1.968, Train acc: 0.5953062392673154
[2024-12-24 17:53:57] - INFO: Epoch: 3, Batch[143/227], Train loss :1.957, Train acc: 0.5938345051379124
[2024-12-24 17:54:00] - INFO: Epoch: 3, Batch[144/227], Train loss :1.844, Train acc: 0.6155495978552279
[2024-12-24 17:54:01] - INFO: Epoch: 3, Batch[145/227], Train loss :1.873, Train acc: 0.6048615036743923
[2024-12-24 17:54:04] - INFO: Epoch: 3, Batch[146/227], Train loss :1.894, Train acc: 0.612617598229109
[2024-12-24 17:54:06] - INFO: Epoch: 3, Batch[147/227], Train loss :1.995, Train acc: 0.5938345051379124
[2024-12-24 17:54:09] - INFO: Epoch: 3, Batch[148/227], Train loss :1.822, Train acc: 0.6177802944507361
[2024-12-24 17:54:11] - INFO: Epoch: 3, Batch[149/227], Train loss :1.795, Train acc: 0.6160809371671991
[2024-12-24 17:54:13] - INFO: Epoch: 3, Batch[150/227], Train loss :1.763, Train acc: 0.6154714850367025
[2024-12-24 17:54:15] - INFO: Epoch: 3, Batch[151/227], Train loss :1.894, Train acc: 0.6017897091722595
[2024-12-24 17:54:18] - INFO: Epoch: 3, Batch[152/227], Train loss :1.846, Train acc: 0.615633423180593
[2024-12-24 17:54:20] - INFO: Epoch: 3, Batch[153/227], Train loss :1.894, Train acc: 0.6100056850483229
[2024-12-24 17:54:22] - INFO: Epoch: 3, Batch[154/227], Train loss :1.854, Train acc: 0.6084401709401709
[2024-12-24 17:54:25] - INFO: Epoch: 3, Batch[155/227], Train loss :1.902, Train acc: 0.5978567399887197
[2024-12-24 17:54:27] - INFO: Epoch: 3, Batch[156/227], Train loss :1.845, Train acc: 0.6052356020942409
[2024-12-24 17:54:29] - INFO: Epoch: 3, Batch[157/227], Train loss :1.894, Train acc: 0.59375
[2024-12-24 17:54:31] - INFO: Epoch: 3, Batch[158/227], Train loss :1.760, Train acc: 0.6342566421707179
[2024-12-24 17:54:33] - INFO: Epoch: 3, Batch[159/227], Train loss :1.856, Train acc: 0.6102062975027145
[2024-12-24 17:54:35] - INFO: Epoch: 3, Batch[160/227], Train loss :1.913, Train acc: 0.5914477073673364
[2024-12-24 17:54:37] - INFO: Epoch: 3, Batch[161/227], Train loss :1.749, Train acc: 0.6292004634994206
[2024-12-24 17:54:40] - INFO: Epoch: 3, Batch[162/227], Train loss :1.798, Train acc: 0.6320960698689956
[2024-12-24 17:54:42] - INFO: Epoch: 3, Batch[163/227], Train loss :1.962, Train acc: 0.6030368763557483
[2024-12-24 17:54:43] - INFO: Epoch: 3, Batch[164/227], Train loss :1.955, Train acc: 0.5936675461741425
[2024-12-24 17:54:45] - INFO: Epoch: 3, Batch[165/227], Train loss :1.868, Train acc: 0.6025857223159078
[2024-12-24 17:54:47] - INFO: Epoch: 3, Batch[166/227], Train loss :1.685, Train acc: 0.6337016574585635
[2024-12-24 17:54:50] - INFO: Epoch: 3, Batch[167/227], Train loss :1.979, Train acc: 0.6018826135105205
[2024-12-24 17:54:52] - INFO: Epoch: 3, Batch[168/227], Train loss :1.753, Train acc: 0.6379907621247113
[2024-12-24 17:54:55] - INFO: Epoch: 3, Batch[169/227], Train loss :1.856, Train acc: 0.622610595303113
[2024-12-24 17:54:57] - INFO: Epoch: 3, Batch[170/227], Train loss :1.791, Train acc: 0.6148901981788967
[2024-12-24 17:54:59] - INFO: Epoch: 3, Batch[171/227], Train loss :1.867, Train acc: 0.6130653266331658
[2024-12-24 17:55:01] - INFO: Epoch: 3, Batch[172/227], Train loss :1.909, Train acc: 0.5938676707907478
[2024-12-24 17:55:03] - INFO: Epoch: 3, Batch[173/227], Train loss :1.992, Train acc: 0.5929705215419501
[2024-12-24 17:55:06] - INFO: Epoch: 3, Batch[174/227], Train loss :1.775, Train acc: 0.6230337078651685
[2024-12-24 17:55:09] - INFO: Epoch: 3, Batch[175/227], Train loss :1.789, Train acc: 0.6127104834329169
[2024-12-24 17:55:11] - INFO: Epoch: 3, Batch[176/227], Train loss :1.989, Train acc: 0.6040156162855549
[2024-12-24 17:55:13] - INFO: Epoch: 3, Batch[177/227], Train loss :2.004, Train acc: 0.597623089983022
[2024-12-24 17:55:16] - INFO: Epoch: 3, Batch[178/227], Train loss :1.799, Train acc: 0.6206896551724138
[2024-12-24 17:55:18] - INFO: Epoch: 3, Batch[179/227], Train loss :1.934, Train acc: 0.5993537964458805
[2024-12-24 17:55:21] - INFO: Epoch: 3, Batch[180/227], Train loss :1.865, Train acc: 0.6165626772546795
[2024-12-24 17:55:23] - INFO: Epoch: 3, Batch[181/227], Train loss :1.949, Train acc: 0.5958904109589042
[2024-12-24 17:55:26] - INFO: Epoch: 3, Batch[182/227], Train loss :2.003, Train acc: 0.5996563573883161
[2024-12-24 17:55:27] - INFO: Epoch: 3, Batch[183/227], Train loss :1.880, Train acc: 0.5971459934138309
[2024-12-24 17:55:30] - INFO: Epoch: 3, Batch[184/227], Train loss :1.734, Train acc: 0.6239217941345601
[2024-12-24 17:55:32] - INFO: Epoch: 3, Batch[185/227], Train loss :1.848, Train acc: 0.6094198736358415
[2024-12-24 17:55:34] - INFO: Epoch: 3, Batch[186/227], Train loss :1.908, Train acc: 0.6090712742980562
[2024-12-24 17:55:37] - INFO: Epoch: 3, Batch[187/227], Train loss :1.895, Train acc: 0.5988826815642458
[2024-12-24 17:55:40] - INFO: Epoch: 3, Batch[188/227], Train loss :1.798, Train acc: 0.6329647182727751
[2024-12-24 17:55:42] - INFO: Epoch: 3, Batch[189/227], Train loss :1.854, Train acc: 0.6121912769311614
[2024-12-24 17:55:43] - INFO: Epoch: 3, Batch[190/227], Train loss :1.763, Train acc: 0.6158675799086758
[2024-12-24 17:55:46] - INFO: Epoch: 3, Batch[191/227], Train loss :1.844, Train acc: 0.612289287656335
[2024-12-24 17:55:48] - INFO: Epoch: 3, Batch[192/227], Train loss :1.939, Train acc: 0.6048728813559322
[2024-12-24 17:55:50] - INFO: Epoch: 3, Batch[193/227], Train loss :1.881, Train acc: 0.6218627997769102
[2024-12-24 17:55:52] - INFO: Epoch: 3, Batch[194/227], Train loss :1.956, Train acc: 0.5939726027397261
[2024-12-24 17:55:54] - INFO: Epoch: 3, Batch[195/227], Train loss :1.907, Train acc: 0.6074270557029178
[2024-12-24 17:55:57] - INFO: Epoch: 3, Batch[196/227], Train loss :1.847, Train acc: 0.613599568267674
[2024-12-24 17:55:58] - INFO: Epoch: 3, Batch[197/227], Train loss :1.827, Train acc: 0.6145893164847789
[2024-12-24 17:56:00] - INFO: Epoch: 3, Batch[198/227], Train loss :1.770, Train acc: 0.6150881057268722
[2024-12-24 17:56:03] - INFO: Epoch: 3, Batch[199/227], Train loss :1.885, Train acc: 0.5962962962962963
[2024-12-24 17:56:04] - INFO: Epoch: 3, Batch[200/227], Train loss :1.822, Train acc: 0.6172006745362564
[2024-12-24 17:56:06] - INFO: Epoch: 3, Batch[201/227], Train loss :1.828, Train acc: 0.6154684095860566
[2024-12-24 17:56:09] - INFO: Epoch: 3, Batch[202/227], Train loss :1.826, Train acc: 0.6052060737527115
[2024-12-24 17:56:11] - INFO: Epoch: 3, Batch[203/227], Train loss :1.920, Train acc: 0.6117850953206239
[2024-12-24 17:56:13] - INFO: Epoch: 3, Batch[204/227], Train loss :1.947, Train acc: 0.5976627712854758
[2024-12-24 17:56:15] - INFO: Epoch: 3, Batch[205/227], Train loss :1.697, Train acc: 0.6271855611957134
[2024-12-24 17:56:18] - INFO: Epoch: 3, Batch[206/227], Train loss :1.931, Train acc: 0.6039660056657223
[2024-12-24 17:56:20] - INFO: Epoch: 3, Batch[207/227], Train loss :1.904, Train acc: 0.6029492080830148
[2024-12-24 17:56:23] - INFO: Epoch: 3, Batch[208/227], Train loss :1.817, Train acc: 0.6126878130217028
[2024-12-24 17:56:25] - INFO: Epoch: 3, Batch[209/227], Train loss :1.833, Train acc: 0.6222101252041372
[2024-12-24 17:56:27] - INFO: Epoch: 3, Batch[210/227], Train loss :1.991, Train acc: 0.5930425179458862
[2024-12-24 17:56:30] - INFO: Epoch: 3, Batch[211/227], Train loss :2.034, Train acc: 0.5883947939262473
[2024-12-24 17:56:32] - INFO: Epoch: 3, Batch[212/227], Train loss :1.844, Train acc: 0.6147356580427447
[2024-12-24 17:56:34] - INFO: Epoch: 3, Batch[213/227], Train loss :1.855, Train acc: 0.6149204607789358
[2024-12-24 17:56:36] - INFO: Epoch: 3, Batch[214/227], Train loss :1.643, Train acc: 0.6349570200573066
[2024-12-24 17:56:39] - INFO: Epoch: 3, Batch[215/227], Train loss :1.989, Train acc: 0.5867944621938233
[2024-12-24 17:56:41] - INFO: Epoch: 3, Batch[216/227], Train loss :1.892, Train acc: 0.5943753380205516
[2024-12-24 17:56:44] - INFO: Epoch: 3, Batch[217/227], Train loss :1.852, Train acc: 0.6111111111111112
[2024-12-24 17:56:46] - INFO: Epoch: 3, Batch[218/227], Train loss :1.755, Train acc: 0.6386986301369864
[2024-12-24 17:56:48] - INFO: Epoch: 3, Batch[219/227], Train loss :1.770, Train acc: 0.6192634560906516
[2024-12-24 17:56:51] - INFO: Epoch: 3, Batch[220/227], Train loss :1.922, Train acc: 0.5991140642303433
[2024-12-24 17:56:53] - INFO: Epoch: 3, Batch[221/227], Train loss :1.759, Train acc: 0.6353711790393013
[2024-12-24 17:56:55] - INFO: Epoch: 3, Batch[222/227], Train loss :1.730, Train acc: 0.6344444444444445
[2024-12-24 17:56:58] - INFO: Epoch: 3, Batch[223/227], Train loss :1.959, Train acc: 0.5918023582257159
[2024-12-24 17:57:00] - INFO: Epoch: 3, Batch[224/227], Train loss :1.871, Train acc: 0.5993227990970654
[2024-12-24 17:57:02] - INFO: Epoch: 3, Batch[225/227], Train loss :1.806, Train acc: 0.6249288560045532
[2024-12-24 17:57:03] - INFO: Epoch: 3, Batch[226/227], Train loss :1.818, Train acc: 0.6214285714285714
[2024-12-24 17:57:03] - INFO: Epoch: 3, Train loss: 1.832, Epoch time = 509.716s
[2024-12-24 17:57:09] - INFO: Accuracy on validation0.560
[2024-12-24 17:57:12] - INFO: Epoch: 4, Batch[0/227], Train loss :1.603, Train acc: 0.6491422246817931
[2024-12-24 17:57:14] - INFO: Epoch: 4, Batch[1/227], Train loss :1.775, Train acc: 0.6274403470715835
[2024-12-24 17:57:16] - INFO: Epoch: 4, Batch[2/227], Train loss :1.673, Train acc: 0.6383442265795207
[2024-12-24 17:57:17] - INFO: Epoch: 4, Batch[3/227], Train loss :1.667, Train acc: 0.6550925925925926
[2024-12-24 17:57:20] - INFO: Epoch: 4, Batch[4/227], Train loss :1.639, Train acc: 0.6354875283446711
[2024-12-24 17:57:22] - INFO: Epoch: 4, Batch[5/227], Train loss :1.732, Train acc: 0.6331685886875343
[2024-12-24 17:57:25] - INFO: Epoch: 4, Batch[6/227], Train loss :1.694, Train acc: 0.6410684474123539
[2024-12-24 17:57:27] - INFO: Epoch: 4, Batch[7/227], Train loss :1.815, Train acc: 0.6221148684916801
[2024-12-24 17:57:29] - INFO: Epoch: 4, Batch[8/227], Train loss :1.705, Train acc: 0.630384167636787
[2024-12-24 17:57:32] - INFO: Epoch: 4, Batch[9/227], Train loss :1.668, Train acc: 0.6321647189760712
[2024-12-24 17:57:34] - INFO: Epoch: 4, Batch[10/227], Train loss :1.697, Train acc: 0.639344262295082
[2024-12-24 17:57:36] - INFO: Epoch: 4, Batch[11/227], Train loss :1.700, Train acc: 0.6312255176273084
[2024-12-24 17:57:39] - INFO: Epoch: 4, Batch[12/227], Train loss :1.645, Train acc: 0.6499180775532496
[2024-12-24 17:57:41] - INFO: Epoch: 4, Batch[13/227], Train loss :1.794, Train acc: 0.6220994475138122
[2024-12-24 17:57:43] - INFO: Epoch: 4, Batch[14/227], Train loss :1.776, Train acc: 0.6237085372485046
[2024-12-24 17:57:45] - INFO: Epoch: 4, Batch[15/227], Train loss :1.705, Train acc: 0.632183908045977
[2024-12-24 17:57:47] - INFO: Epoch: 4, Batch[16/227], Train loss :1.671, Train acc: 0.6386840612592173
[2024-12-24 17:57:50] - INFO: Epoch: 4, Batch[17/227], Train loss :1.686, Train acc: 0.6374458874458875
[2024-12-24 17:57:53] - INFO: Epoch: 4, Batch[18/227], Train loss :1.745, Train acc: 0.617055947854427
[2024-12-24 17:57:56] - INFO: Epoch: 4, Batch[19/227], Train loss :1.769, Train acc: 0.6257828810020877
[2024-12-24 17:57:58] - INFO: Epoch: 4, Batch[20/227], Train loss :1.715, Train acc: 0.6376021798365122
[2024-12-24 17:58:00] - INFO: Epoch: 4, Batch[21/227], Train loss :1.764, Train acc: 0.6352413019079686
[2024-12-24 17:58:02] - INFO: Epoch: 4, Batch[22/227], Train loss :1.755, Train acc: 0.6283672347443651
[2024-12-24 17:58:04] - INFO: Epoch: 4, Batch[23/227], Train loss :1.681, Train acc: 0.6393534002229655
[2024-12-24 17:58:07] - INFO: Epoch: 4, Batch[24/227], Train loss :1.930, Train acc: 0.6029094827586207
[2024-12-24 17:58:09] - INFO: Epoch: 4, Batch[25/227], Train loss :1.732, Train acc: 0.627906976744186
[2024-12-24 17:58:12] - INFO: Epoch: 4, Batch[26/227], Train loss :1.791, Train acc: 0.6174920969441517
[2024-12-24 17:58:13] - INFO: Epoch: 4, Batch[27/227], Train loss :1.838, Train acc: 0.6020293122886133
[2024-12-24 17:58:16] - INFO: Epoch: 4, Batch[28/227], Train loss :1.857, Train acc: 0.6048
[2024-12-24 17:58:18] - INFO: Epoch: 4, Batch[29/227], Train loss :1.675, Train acc: 0.6455331412103746
[2024-12-24 17:58:20] - INFO: Epoch: 4, Batch[30/227], Train loss :1.844, Train acc: 0.5952512424075097
[2024-12-24 17:58:23] - INFO: Epoch: 4, Batch[31/227], Train loss :1.654, Train acc: 0.658675799086758
[2024-12-24 17:58:25] - INFO: Epoch: 4, Batch[32/227], Train loss :1.700, Train acc: 0.6349115801483172
[2024-12-24 17:58:27] - INFO: Epoch: 4, Batch[33/227], Train loss :1.743, Train acc: 0.64
[2024-12-24 17:58:28] - INFO: Epoch: 4, Batch[34/227], Train loss :1.642, Train acc: 0.6628242074927954
[2024-12-24 17:58:31] - INFO: Epoch: 4, Batch[35/227], Train loss :1.767, Train acc: 0.6274278987639789
[2024-12-24 17:58:33] - INFO: Epoch: 4, Batch[36/227], Train loss :1.703, Train acc: 0.6331712587790383
[2024-12-24 17:58:35] - INFO: Epoch: 4, Batch[37/227], Train loss :1.753, Train acc: 0.6316076294277929
[2024-12-24 17:58:37] - INFO: Epoch: 4, Batch[38/227], Train loss :1.692, Train acc: 0.6400924321201618
[2024-12-24 17:58:40] - INFO: Epoch: 4, Batch[39/227], Train loss :1.837, Train acc: 0.6045016077170418
[2024-12-24 17:58:42] - INFO: Epoch: 4, Batch[40/227], Train loss :1.776, Train acc: 0.6203150461705594
[2024-12-24 17:58:44] - INFO: Epoch: 4, Batch[41/227], Train loss :1.742, Train acc: 0.6243508366993653
[2024-12-24 17:58:47] - INFO: Epoch: 4, Batch[42/227], Train loss :1.778, Train acc: 0.6238630283574104
[2024-12-24 17:58:49] - INFO: Epoch: 4, Batch[43/227], Train loss :1.836, Train acc: 0.615857063093244
[2024-12-24 17:58:51] - INFO: Epoch: 4, Batch[44/227], Train loss :1.672, Train acc: 0.646092865232163
[2024-12-24 17:58:53] - INFO: Epoch: 4, Batch[45/227], Train loss :1.870, Train acc: 0.6156443444006753
[2024-12-24 17:58:56] - INFO: Epoch: 4, Batch[46/227], Train loss :1.739, Train acc: 0.6268823201338539
[2024-12-24 17:58:58] - INFO: Epoch: 4, Batch[47/227], Train loss :1.771, Train acc: 0.6242490442381212
[2024-12-24 17:59:00] - INFO: Epoch: 4, Batch[48/227], Train loss :1.640, Train acc: 0.664179104477612
[2024-12-24 17:59:02] - INFO: Epoch: 4, Batch[49/227], Train loss :1.703, Train acc: 0.6310788149804359
[2024-12-24 17:59:04] - INFO: Epoch: 4, Batch[50/227], Train loss :1.715, Train acc: 0.6321014892443464
[2024-12-24 17:59:06] - INFO: Epoch: 4, Batch[51/227], Train loss :1.704, Train acc: 0.6500283607487237
[2024-12-24 17:59:08] - INFO: Epoch: 4, Batch[52/227], Train loss :1.678, Train acc: 0.6542372881355932
[2024-12-24 17:59:10] - INFO: Epoch: 4, Batch[53/227], Train loss :1.774, Train acc: 0.6254847645429363
[2024-12-24 17:59:13] - INFO: Epoch: 4, Batch[54/227], Train loss :1.829, Train acc: 0.6205016357688113
[2024-12-24 17:59:14] - INFO: Epoch: 4, Batch[55/227], Train loss :1.806, Train acc: 0.6073585941790225
[2024-12-24 17:59:17] - INFO: Epoch: 4, Batch[56/227], Train loss :1.704, Train acc: 0.6370328425821065
[2024-12-24 17:59:19] - INFO: Epoch: 4, Batch[57/227], Train loss :1.719, Train acc: 0.6358665937670859
[2024-12-24 17:59:21] - INFO: Epoch: 4, Batch[58/227], Train loss :1.729, Train acc: 0.6282121377802078
[2024-12-24 17:59:24] - INFO: Epoch: 4, Batch[59/227], Train loss :1.889, Train acc: 0.599572877736252
[2024-12-24 17:59:26] - INFO: Epoch: 4, Batch[60/227], Train loss :1.775, Train acc: 0.6228540772532188
[2024-12-24 17:59:27] - INFO: Epoch: 4, Batch[61/227], Train loss :1.603, Train acc: 0.656047197640118
[2024-12-24 17:59:30] - INFO: Epoch: 4, Batch[62/227], Train loss :1.722, Train acc: 0.6387940841865757
[2024-12-24 17:59:31] - INFO: Epoch: 4, Batch[63/227], Train loss :1.702, Train acc: 0.6440202133632791
[2024-12-24 17:59:34] - INFO: Epoch: 4, Batch[64/227], Train loss :1.652, Train acc: 0.641914191419142
[2024-12-24 17:59:36] - INFO: Epoch: 4, Batch[65/227], Train loss :1.743, Train acc: 0.6279607163489312
[2024-12-24 17:59:38] - INFO: Epoch: 4, Batch[66/227], Train loss :1.891, Train acc: 0.6061269146608315
[2024-12-24 17:59:40] - INFO: Epoch: 4, Batch[67/227], Train loss :1.794, Train acc: 0.609967497291441
[2024-12-24 17:59:42] - INFO: Epoch: 4, Batch[68/227], Train loss :1.731, Train acc: 0.6512676056338028
[2024-12-24 17:59:44] - INFO: Epoch: 4, Batch[69/227], Train loss :1.665, Train acc: 0.6399779127553837
[2024-12-24 17:59:47] - INFO: Epoch: 4, Batch[70/227], Train loss :1.794, Train acc: 0.6059602649006622
[2024-12-24 17:59:49] - INFO: Epoch: 4, Batch[71/227], Train loss :1.924, Train acc: 0.6046137339055794
[2024-12-24 17:59:51] - INFO: Epoch: 4, Batch[72/227], Train loss :1.663, Train acc: 0.6441363373772386
[2024-12-24 17:59:54] - INFO: Epoch: 4, Batch[73/227], Train loss :1.743, Train acc: 0.6245989304812835
[2024-12-24 17:59:56] - INFO: Epoch: 4, Batch[74/227], Train loss :1.722, Train acc: 0.6290672451193059
[2024-12-24 17:59:58] - INFO: Epoch: 4, Batch[75/227], Train loss :1.807, Train acc: 0.619351100811124
[2024-12-24 18:00:00] - INFO: Epoch: 4, Batch[76/227], Train loss :1.779, Train acc: 0.6309385863267671
[2024-12-24 18:00:01] - INFO: Epoch: 4, Batch[77/227], Train loss :1.648, Train acc: 0.6354570637119114
[2024-12-24 18:00:03] - INFO: Epoch: 4, Batch[78/227], Train loss :1.678, Train acc: 0.6349118942731278
[2024-12-24 18:00:05] - INFO: Epoch: 4, Batch[79/227], Train loss :1.811, Train acc: 0.6072766185125735
[2024-12-24 18:00:08] - INFO: Epoch: 4, Batch[80/227], Train loss :1.858, Train acc: 0.608719052744887
[2024-12-24 18:00:10] - INFO: Epoch: 4, Batch[81/227], Train loss :1.695, Train acc: 0.6374331550802139
[2024-12-24 18:00:12] - INFO: Epoch: 4, Batch[82/227], Train loss :1.756, Train acc: 0.6309794988610479
[2024-12-24 18:00:14] - INFO: Epoch: 4, Batch[83/227], Train loss :1.708, Train acc: 0.6436909394107838
[2024-12-24 18:00:17] - INFO: Epoch: 4, Batch[84/227], Train loss :1.734, Train acc: 0.6285397001665741
[2024-12-24 18:00:19] - INFO: Epoch: 4, Batch[85/227], Train loss :1.907, Train acc: 0.5998877665544332
[2024-12-24 18:00:22] - INFO: Epoch: 4, Batch[86/227], Train loss :1.760, Train acc: 0.6304100227790432
[2024-12-24 18:00:24] - INFO: Epoch: 4, Batch[87/227], Train loss :1.824, Train acc: 0.602377093462993
[2024-12-24 18:00:26] - INFO: Epoch: 4, Batch[88/227], Train loss :1.739, Train acc: 0.6269884805266045
[2024-12-24 18:00:28] - INFO: Epoch: 4, Batch[89/227], Train loss :1.775, Train acc: 0.6037333333333333
[2024-12-24 18:00:31] - INFO: Epoch: 4, Batch[90/227], Train loss :1.732, Train acc: 0.6265788028555739
[2024-12-24 18:00:33] - INFO: Epoch: 4, Batch[91/227], Train loss :1.785, Train acc: 0.6178067318132465
[2024-12-24 18:00:36] - INFO: Epoch: 4, Batch[92/227], Train loss :1.758, Train acc: 0.6145552560646901
[2024-12-24 18:00:38] - INFO: Epoch: 4, Batch[93/227], Train loss :1.829, Train acc: 0.6135073779795687
[2024-12-24 18:00:40] - INFO: Epoch: 4, Batch[94/227], Train loss :1.689, Train acc: 0.6420581655480985
[2024-12-24 18:00:42] - INFO: Epoch: 4, Batch[95/227], Train loss :1.733, Train acc: 0.6410398230088495
[2024-12-24 18:00:44] - INFO: Epoch: 4, Batch[96/227], Train loss :1.684, Train acc: 0.6339686098654709
[2024-12-24 18:00:46] - INFO: Epoch: 4, Batch[97/227], Train loss :1.738, Train acc: 0.6203007518796992
[2024-12-24 18:00:49] - INFO: Epoch: 4, Batch[98/227], Train loss :1.810, Train acc: 0.6057529610829103
[2024-12-24 18:00:52] - INFO: Epoch: 4, Batch[99/227], Train loss :1.681, Train acc: 0.6276537833424061
[2024-12-24 18:00:54] - INFO: Epoch: 4, Batch[100/227], Train loss :1.868, Train acc: 0.6102819237147595
[2024-12-24 18:00:56] - INFO: Epoch: 4, Batch[101/227], Train loss :1.784, Train acc: 0.628494623655914
[2024-12-24 18:00:58] - INFO: Epoch: 4, Batch[102/227], Train loss :1.827, Train acc: 0.6126373626373627
[2024-12-24 18:01:01] - INFO: Epoch: 4, Batch[103/227], Train loss :1.873, Train acc: 0.6172566371681416
[2024-12-24 18:01:03] - INFO: Epoch: 4, Batch[104/227], Train loss :1.713, Train acc: 0.6321388577827548
[2024-12-24 18:01:05] - INFO: Epoch: 4, Batch[105/227], Train loss :1.793, Train acc: 0.6171832517672649
[2024-12-24 18:01:07] - INFO: Epoch: 4, Batch[106/227], Train loss :1.732, Train acc: 0.6195530726256984
[2024-12-24 18:01:10] - INFO: Epoch: 4, Batch[107/227], Train loss :1.827, Train acc: 0.6172707889125799
[2024-12-24 18:01:12] - INFO: Epoch: 4, Batch[108/227], Train loss :1.806, Train acc: 0.620021528525296
[2024-12-24 18:01:14] - INFO: Epoch: 4, Batch[109/227], Train loss :1.769, Train acc: 0.6382737081203862
[2024-12-24 18:01:16] - INFO: Epoch: 4, Batch[110/227], Train loss :1.776, Train acc: 0.6161238654564869
[2024-12-24 18:01:18] - INFO: Epoch: 4, Batch[111/227], Train loss :1.830, Train acc: 0.623608017817372
[2024-12-24 18:01:21] - INFO: Epoch: 4, Batch[112/227], Train loss :1.771, Train acc: 0.6213333333333333
[2024-12-24 18:01:23] - INFO: Epoch: 4, Batch[113/227], Train loss :1.733, Train acc: 0.6314907872696818
[2024-12-24 18:01:26] - INFO: Epoch: 4, Batch[114/227], Train loss :1.797, Train acc: 0.6287128712871287
[2024-12-24 18:01:28] - INFO: Epoch: 4, Batch[115/227], Train loss :1.833, Train acc: 0.6139852188743604
[2024-12-24 18:01:29] - INFO: Epoch: 4, Batch[116/227], Train loss :1.768, Train acc: 0.627030162412993
[2024-12-24 18:01:32] - INFO: Epoch: 4, Batch[117/227], Train loss :1.657, Train acc: 0.641273450824332
[2024-12-24 18:01:34] - INFO: Epoch: 4, Batch[118/227], Train loss :1.797, Train acc: 0.6120218579234973
[2024-12-24 18:01:36] - INFO: Epoch: 4, Batch[119/227], Train loss :1.814, Train acc: 0.6076502732240437
[2024-12-24 18:01:38] - INFO: Epoch: 4, Batch[120/227], Train loss :1.691, Train acc: 0.6193905817174515
[2024-12-24 18:01:40] - INFO: Epoch: 4, Batch[121/227], Train loss :1.771, Train acc: 0.6158833063209076
[2024-12-24 18:01:42] - INFO: Epoch: 4, Batch[122/227], Train loss :1.757, Train acc: 0.6214876033057851
[2024-12-24 18:01:44] - INFO: Epoch: 4, Batch[123/227], Train loss :1.671, Train acc: 0.6222098214285714
[2024-12-24 18:01:47] - INFO: Epoch: 4, Batch[124/227], Train loss :1.753, Train acc: 0.6386363636363637
[2024-12-24 18:01:50] - INFO: Epoch: 4, Batch[125/227], Train loss :1.733, Train acc: 0.6226096737907761
[2024-12-24 18:01:52] - INFO: Epoch: 4, Batch[126/227], Train loss :1.864, Train acc: 0.6127262753702688
[2024-12-24 18:01:54] - INFO: Epoch: 4, Batch[127/227], Train loss :1.790, Train acc: 0.6410550458715596
[2024-12-24 18:01:57] - INFO: Epoch: 4, Batch[128/227], Train loss :1.795, Train acc: 0.6061257388500806
[2024-12-24 18:01:59] - INFO: Epoch: 4, Batch[129/227], Train loss :1.831, Train acc: 0.62157629960872
[2024-12-24 18:02:02] - INFO: Epoch: 4, Batch[130/227], Train loss :1.817, Train acc: 0.6166941241076331
[2024-12-24 18:02:04] - INFO: Epoch: 4, Batch[131/227], Train loss :1.755, Train acc: 0.6191512513601741
[2024-12-24 18:02:06] - INFO: Epoch: 4, Batch[132/227], Train loss :1.809, Train acc: 0.6271468144044321
[2024-12-24 18:02:08] - INFO: Epoch: 4, Batch[133/227], Train loss :1.814, Train acc: 0.6111111111111112
[2024-12-24 18:02:11] - INFO: Epoch: 4, Batch[134/227], Train loss :1.685, Train acc: 0.6383442265795207
[2024-12-24 18:02:13] - INFO: Epoch: 4, Batch[135/227], Train loss :1.809, Train acc: 0.6106870229007634
[2024-12-24 18:02:14] - INFO: Epoch: 4, Batch[136/227], Train loss :1.774, Train acc: 0.6237458193979933
[2024-12-24 18:02:17] - INFO: Epoch: 4, Batch[137/227], Train loss :1.862, Train acc: 0.6029016657710908
[2024-12-24 18:02:20] - INFO: Epoch: 4, Batch[138/227], Train loss :1.911, Train acc: 0.601528384279476
[2024-12-24 18:02:22] - INFO: Epoch: 4, Batch[139/227], Train loss :1.672, Train acc: 0.64257481648786
[2024-12-24 18:02:24] - INFO: Epoch: 4, Batch[140/227], Train loss :1.876, Train acc: 0.6123688569850911
[2024-12-24 18:02:26] - INFO: Epoch: 4, Batch[141/227], Train loss :1.713, Train acc: 0.6296728971962616
[2024-12-24 18:02:28] - INFO: Epoch: 4, Batch[142/227], Train loss :1.721, Train acc: 0.6286353467561522
[2024-12-24 18:02:31] - INFO: Epoch: 4, Batch[143/227], Train loss :1.690, Train acc: 0.637855579868709
[2024-12-24 18:02:34] - INFO: Epoch: 4, Batch[144/227], Train loss :1.699, Train acc: 0.6379804934021801
[2024-12-24 18:02:36] - INFO: Epoch: 4, Batch[145/227], Train loss :1.837, Train acc: 0.6120892018779343
[2024-12-24 18:02:38] - INFO: Epoch: 4, Batch[146/227], Train loss :1.774, Train acc: 0.6231182795698925
[2024-12-24 18:02:40] - INFO: Epoch: 4, Batch[147/227], Train loss :1.625, Train acc: 0.6338411316648531
[2024-12-24 18:02:42] - INFO: Epoch: 4, Batch[148/227], Train loss :1.743, Train acc: 0.6216676120249575
[2024-12-24 18:02:46] - INFO: Epoch: 4, Batch[149/227], Train loss :1.777, Train acc: 0.6167698368036015
[2024-12-24 18:02:47] - INFO: Epoch: 4, Batch[150/227], Train loss :1.764, Train acc: 0.6122797043774872
[2024-12-24 18:02:49] - INFO: Epoch: 4, Batch[151/227], Train loss :1.796, Train acc: 0.6080773606370876
[2024-12-24 18:02:52] - INFO: Epoch: 4, Batch[152/227], Train loss :1.942, Train acc: 0.5991091314031181
[2024-12-24 18:02:54] - INFO: Epoch: 4, Batch[153/227], Train loss :1.878, Train acc: 0.613262599469496
[2024-12-24 18:02:56] - INFO: Epoch: 4, Batch[154/227], Train loss :1.895, Train acc: 0.6081609837898267
[2024-12-24 18:02:58] - INFO: Epoch: 4, Batch[155/227], Train loss :1.727, Train acc: 0.628680479825518
[2024-12-24 18:03:01] - INFO: Epoch: 4, Batch[156/227], Train loss :1.703, Train acc: 0.629608938547486
[2024-12-24 18:03:03] - INFO: Epoch: 4, Batch[157/227], Train loss :1.806, Train acc: 0.6189439303211758
[2024-12-24 18:03:05] - INFO: Epoch: 4, Batch[158/227], Train loss :1.838, Train acc: 0.6152137701277068
[2024-12-24 18:03:07] - INFO: Epoch: 4, Batch[159/227], Train loss :1.753, Train acc: 0.6068783068783069
[2024-12-24 18:03:10] - INFO: Epoch: 4, Batch[160/227], Train loss :1.798, Train acc: 0.6312328767123287
[2024-12-24 18:03:12] - INFO: Epoch: 4, Batch[161/227], Train loss :1.708, Train acc: 0.6359525155455059
[2024-12-24 18:03:15] - INFO: Epoch: 4, Batch[162/227], Train loss :1.898, Train acc: 0.596661281637049
[2024-12-24 18:03:17] - INFO: Epoch: 4, Batch[163/227], Train loss :1.757, Train acc: 0.6153005464480874
[2024-12-24 18:03:19] - INFO: Epoch: 4, Batch[164/227], Train loss :1.748, Train acc: 0.6366090712742981
[2024-12-24 18:03:22] - INFO: Epoch: 4, Batch[165/227], Train loss :1.883, Train acc: 0.6126027397260274
[2024-12-24 18:03:24] - INFO: Epoch: 4, Batch[166/227], Train loss :1.740, Train acc: 0.6303479749001711
[2024-12-24 18:03:26] - INFO: Epoch: 4, Batch[167/227], Train loss :1.830, Train acc: 0.616557734204793
[2024-12-24 18:03:28] - INFO: Epoch: 4, Batch[168/227], Train loss :1.751, Train acc: 0.6379603399433428
[2024-12-24 18:03:30] - INFO: Epoch: 4, Batch[169/227], Train loss :1.772, Train acc: 0.6225127913587265
[2024-12-24 18:03:32] - INFO: Epoch: 4, Batch[170/227], Train loss :1.670, Train acc: 0.642656162070906
[2024-12-24 18:03:34] - INFO: Epoch: 4, Batch[171/227], Train loss :1.759, Train acc: 0.6126798082045818
[2024-12-24 18:03:36] - INFO: Epoch: 4, Batch[172/227], Train loss :1.799, Train acc: 0.613747954173486
[2024-12-24 18:03:38] - INFO: Epoch: 4, Batch[173/227], Train loss :1.737, Train acc: 0.6384180790960452
[2024-12-24 18:03:41] - INFO: Epoch: 4, Batch[174/227], Train loss :1.855, Train acc: 0.6172506738544474
[2024-12-24 18:03:43] - INFO: Epoch: 4, Batch[175/227], Train loss :1.842, Train acc: 0.607103218645949
[2024-12-24 18:03:45] - INFO: Epoch: 4, Batch[176/227], Train loss :1.689, Train acc: 0.6302797586396051
[2024-12-24 18:03:47] - INFO: Epoch: 4, Batch[177/227], Train loss :1.815, Train acc: 0.605827377680044
[2024-12-24 18:03:49] - INFO: Epoch: 4, Batch[178/227], Train loss :1.816, Train acc: 0.625
[2024-12-24 18:03:52] - INFO: Epoch: 4, Batch[179/227], Train loss :1.877, Train acc: 0.6040861402540033
[2024-12-24 18:03:54] - INFO: Epoch: 4, Batch[180/227], Train loss :1.799, Train acc: 0.6150134048257373
[2024-12-24 18:03:56] - INFO: Epoch: 4, Batch[181/227], Train loss :1.990, Train acc: 0.5925315760571115
[2024-12-24 18:03:59] - INFO: Epoch: 4, Batch[182/227], Train loss :1.762, Train acc: 0.6265466816647919
[2024-12-24 18:04:01] - INFO: Epoch: 4, Batch[183/227], Train loss :1.833, Train acc: 0.6154261057173679
[2024-12-24 18:04:03] - INFO: Epoch: 4, Batch[184/227], Train loss :1.829, Train acc: 0.6128491620111732
[2024-12-24 18:04:05] - INFO: Epoch: 4, Batch[185/227], Train loss :1.792, Train acc: 0.6005665722379604
[2024-12-24 18:04:07] - INFO: Epoch: 4, Batch[186/227], Train loss :1.841, Train acc: 0.6158160403813797
[2024-12-24 18:04:09] - INFO: Epoch: 4, Batch[187/227], Train loss :1.757, Train acc: 0.6212885154061625
[2024-12-24 18:04:11] - INFO: Epoch: 4, Batch[188/227], Train loss :1.760, Train acc: 0.621477937267411
[2024-12-24 18:04:14] - INFO: Epoch: 4, Batch[189/227], Train loss :1.888, Train acc: 0.6044420941300899
[2024-12-24 18:04:15] - INFO: Epoch: 4, Batch[190/227], Train loss :1.747, Train acc: 0.6414141414141414
[2024-12-24 18:04:17] - INFO: Epoch: 4, Batch[191/227], Train loss :1.947, Train acc: 0.6004343105320304
[2024-12-24 18:04:20] - INFO: Epoch: 4, Batch[192/227], Train loss :1.887, Train acc: 0.6230425055928411
[2024-12-24 18:04:23] - INFO: Epoch: 4, Batch[193/227], Train loss :1.690, Train acc: 0.6399317406143344
[2024-12-24 18:04:25] - INFO: Epoch: 4, Batch[194/227], Train loss :1.700, Train acc: 0.6420034149117815
[2024-12-24 18:04:27] - INFO: Epoch: 4, Batch[195/227], Train loss :1.799, Train acc: 0.6216216216216216
[2024-12-24 18:04:30] - INFO: Epoch: 4, Batch[196/227], Train loss :1.842, Train acc: 0.6134220743205768
[2024-12-24 18:04:32] - INFO: Epoch: 4, Batch[197/227], Train loss :1.870, Train acc: 0.6103104212860311
[2024-12-24 18:04:34] - INFO: Epoch: 4, Batch[198/227], Train loss :1.641, Train acc: 0.6413288288288288
[2024-12-24 18:04:36] - INFO: Epoch: 4, Batch[199/227], Train loss :1.819, Train acc: 0.6240896358543417
[2024-12-24 18:04:38] - INFO: Epoch: 4, Batch[200/227], Train loss :1.761, Train acc: 0.6136612021857923
[2024-12-24 18:04:40] - INFO: Epoch: 4, Batch[201/227], Train loss :1.775, Train acc: 0.6284741917186614
[2024-12-24 18:04:42] - INFO: Epoch: 4, Batch[202/227], Train loss :1.785, Train acc: 0.6289871292669278
[2024-12-24 18:04:44] - INFO: Epoch: 4, Batch[203/227], Train loss :1.744, Train acc: 0.6216814159292036
[2024-12-24 18:04:46] - INFO: Epoch: 4, Batch[204/227], Train loss :1.749, Train acc: 0.6326530612244898
[2024-12-24 18:04:48] - INFO: Epoch: 4, Batch[205/227], Train loss :1.940, Train acc: 0.5950644980370162
[2024-12-24 18:04:51] - INFO: Epoch: 4, Batch[206/227], Train loss :1.749, Train acc: 0.6354223433242506
[2024-12-24 18:04:53] - INFO: Epoch: 4, Batch[207/227], Train loss :1.791, Train acc: 0.6252747252747253
[2024-12-24 18:04:55] - INFO: Epoch: 4, Batch[208/227], Train loss :1.694, Train acc: 0.6436024162548051
[2024-12-24 18:04:57] - INFO: Epoch: 4, Batch[209/227], Train loss :1.753, Train acc: 0.6223653395784543
[2024-12-24 18:04:59] - INFO: Epoch: 4, Batch[210/227], Train loss :1.726, Train acc: 0.6295662100456622
[2024-12-24 18:05:01] - INFO: Epoch: 4, Batch[211/227], Train loss :1.819, Train acc: 0.6070844686648501
[2024-12-24 18:05:03] - INFO: Epoch: 4, Batch[212/227], Train loss :1.698, Train acc: 0.6349557522123894
[2024-12-24 18:05:05] - INFO: Epoch: 4, Batch[213/227], Train loss :1.907, Train acc: 0.5993485342019544
[2024-12-24 18:05:07] - INFO: Epoch: 4, Batch[214/227], Train loss :1.708, Train acc: 0.6285060103033773
[2024-12-24 18:05:10] - INFO: Epoch: 4, Batch[215/227], Train loss :1.923, Train acc: 0.5966386554621849
[2024-12-24 18:05:12] - INFO: Epoch: 4, Batch[216/227], Train loss :1.875, Train acc: 0.6125202374527793
[2024-12-24 18:05:14] - INFO: Epoch: 4, Batch[217/227], Train loss :1.812, Train acc: 0.6118568232662193
[2024-12-24 18:05:16] - INFO: Epoch: 4, Batch[218/227], Train loss :1.778, Train acc: 0.5957330415754923
[2024-12-24 18:05:19] - INFO: Epoch: 4, Batch[219/227], Train loss :1.803, Train acc: 0.6217070600632244
[2024-12-24 18:05:22] - INFO: Epoch: 4, Batch[220/227], Train loss :1.722, Train acc: 0.636881047239613
[2024-12-24 18:05:24] - INFO: Epoch: 4, Batch[221/227], Train loss :1.800, Train acc: 0.6085526315789473
[2024-12-24 18:05:26] - INFO: Epoch: 4, Batch[222/227], Train loss :1.924, Train acc: 0.6009719222462203
[2024-12-24 18:05:28] - INFO: Epoch: 4, Batch[223/227], Train loss :1.815, Train acc: 0.6273867975995635
[2024-12-24 18:05:30] - INFO: Epoch: 4, Batch[224/227], Train loss :1.875, Train acc: 0.6059767570558937
[2024-12-24 18:05:31] - INFO: Epoch: 4, Batch[225/227], Train loss :1.852, Train acc: 0.6025569760978321
[2024-12-24 18:05:33] - INFO: Epoch: 4, Batch[226/227], Train loss :1.682, Train acc: 0.6646403242147924
[2024-12-24 18:05:33] - INFO: Epoch: 4, Train loss: 1.769, Epoch time = 502.908s
[2024-12-24 18:05:36] - INFO: Epoch: 5, Batch[0/227], Train loss :1.612, Train acc: 0.641602634467618
[2024-12-24 18:05:38] - INFO: Epoch: 5, Batch[1/227], Train loss :1.625, Train acc: 0.647982062780269
[2024-12-24 18:05:41] - INFO: Epoch: 5, Batch[2/227], Train loss :1.749, Train acc: 0.6198257080610022
[2024-12-24 18:05:43] - INFO: Epoch: 5, Batch[3/227], Train loss :1.768, Train acc: 0.619530416221985
[2024-12-24 18:05:45] - INFO: Epoch: 5, Batch[4/227], Train loss :1.591, Train acc: 0.642148277875073
[2024-12-24 18:05:47] - INFO: Epoch: 5, Batch[5/227], Train loss :1.709, Train acc: 0.6370588235294118
[2024-12-24 18:05:49] - INFO: Epoch: 5, Batch[6/227], Train loss :1.669, Train acc: 0.6447007138934652
[2024-12-24 18:05:52] - INFO: Epoch: 5, Batch[7/227], Train loss :1.585, Train acc: 0.6551525618883132
[2024-12-24 18:05:54] - INFO: Epoch: 5, Batch[8/227], Train loss :1.741, Train acc: 0.6132596685082873
[2024-12-24 18:05:56] - INFO: Epoch: 5, Batch[9/227], Train loss :1.779, Train acc: 0.6098505810736027
[2024-12-24 18:05:58] - INFO: Epoch: 5, Batch[10/227], Train loss :1.693, Train acc: 0.6373500856653341
[2024-12-24 18:06:00] - INFO: Epoch: 5, Batch[11/227], Train loss :1.666, Train acc: 0.631810676940011
[2024-12-24 18:06:02] - INFO: Epoch: 5, Batch[12/227], Train loss :1.670, Train acc: 0.628226249313564
[2024-12-24 18:06:04] - INFO: Epoch: 5, Batch[13/227], Train loss :1.613, Train acc: 0.642122360584732
[2024-12-24 18:06:06] - INFO: Epoch: 5, Batch[14/227], Train loss :1.590, Train acc: 0.6462053571428571
[2024-12-24 18:06:09] - INFO: Epoch: 5, Batch[15/227], Train loss :1.669, Train acc: 0.6358635863586358
[2024-12-24 18:06:11] - INFO: Epoch: 5, Batch[16/227], Train loss :1.646, Train acc: 0.6203599550056242
[2024-12-24 18:06:13] - INFO: Epoch: 5, Batch[17/227], Train loss :1.560, Train acc: 0.6590389016018307
[2024-12-24 18:06:16] - INFO: Epoch: 5, Batch[18/227], Train loss :1.606, Train acc: 0.6501668520578421
[2024-12-24 18:06:18] - INFO: Epoch: 5, Batch[19/227], Train loss :1.781, Train acc: 0.6108165429480382
[2024-12-24 18:06:20] - INFO: Epoch: 5, Batch[20/227], Train loss :1.651, Train acc: 0.631700288184438
[2024-12-24 18:06:23] - INFO: Epoch: 5, Batch[21/227], Train loss :1.574, Train acc: 0.6393805309734514
[2024-12-24 18:06:25] - INFO: Epoch: 5, Batch[22/227], Train loss :1.659, Train acc: 0.644122383252818
[2024-12-24 18:06:27] - INFO: Epoch: 5, Batch[23/227], Train loss :1.689, Train acc: 0.6366150442477876
[2024-12-24 18:06:29] - INFO: Epoch: 5, Batch[24/227], Train loss :1.571, Train acc: 0.6453362255965293
[2024-12-24 18:06:31] - INFO: Epoch: 5, Batch[25/227], Train loss :1.679, Train acc: 0.6290050590219224
[2024-12-24 18:06:33] - INFO: Epoch: 5, Batch[26/227], Train loss :1.667, Train acc: 0.6399108138238573
[2024-12-24 18:06:35] - INFO: Epoch: 5, Batch[27/227], Train loss :1.714, Train acc: 0.627906976744186
[2024-12-24 18:06:38] - INFO: Epoch: 5, Batch[28/227], Train loss :1.607, Train acc: 0.6465090709180868
[2024-12-24 18:06:40] - INFO: Epoch: 5, Batch[29/227], Train loss :1.619, Train acc: 0.6456473214285714
[2024-12-24 18:06:42] - INFO: Epoch: 5, Batch[30/227], Train loss :1.784, Train acc: 0.623059866962306
[2024-12-24 18:06:45] - INFO: Epoch: 5, Batch[31/227], Train loss :1.732, Train acc: 0.630793819925413
[2024-12-24 18:06:48] - INFO: Epoch: 5, Batch[32/227], Train loss :1.771, Train acc: 0.6204059243006034
[2024-12-24 18:06:51] - INFO: Epoch: 5, Batch[33/227], Train loss :1.613, Train acc: 0.6466591166477916
[2024-12-24 18:06:53] - INFO: Epoch: 5, Batch[34/227], Train loss :1.625, Train acc: 0.645091693635383
[2024-12-24 18:06:55] - INFO: Epoch: 5, Batch[35/227], Train loss :1.633, Train acc: 0.6372067648663393
[2024-12-24 18:06:57] - INFO: Epoch: 5, Batch[36/227], Train loss :1.605, Train acc: 0.6423804226918799
[2024-12-24 18:07:00] - INFO: Epoch: 5, Batch[37/227], Train loss :1.558, Train acc: 0.6530494821634062
[2024-12-24 18:07:02] - INFO: Epoch: 5, Batch[38/227], Train loss :1.707, Train acc: 0.6306203756402959
[2024-12-24 18:07:04] - INFO: Epoch: 5, Batch[39/227], Train loss :1.590, Train acc: 0.660245183887916
[2024-12-24 18:07:06] - INFO: Epoch: 5, Batch[40/227], Train loss :1.712, Train acc: 0.6360637713029137
[2024-12-24 18:07:08] - INFO: Epoch: 5, Batch[41/227], Train loss :1.789, Train acc: 0.6193149915777653
[2024-12-24 18:07:10] - INFO: Epoch: 5, Batch[42/227], Train loss :1.752, Train acc: 0.6146788990825688
[2024-12-24 18:07:12] - INFO: Epoch: 5, Batch[43/227], Train loss :1.753, Train acc: 0.6219312602291326
[2024-12-24 18:07:15] - INFO: Epoch: 5, Batch[44/227], Train loss :1.559, Train acc: 0.6542523624235687
[2024-12-24 18:07:17] - INFO: Epoch: 5, Batch[45/227], Train loss :1.707, Train acc: 0.6411267605633802
[2024-12-24 18:07:19] - INFO: Epoch: 5, Batch[46/227], Train loss :1.758, Train acc: 0.6348593491450635
[2024-12-24 18:07:22] - INFO: Epoch: 5, Batch[47/227], Train loss :1.751, Train acc: 0.6278350515463917
[2024-12-24 18:07:24] - INFO: Epoch: 5, Batch[48/227], Train loss :1.728, Train acc: 0.6303879310344828
[2024-12-24 18:07:26] - INFO: Epoch: 5, Batch[49/227], Train loss :1.689, Train acc: 0.6366594360086768
[2024-12-24 18:07:28] - INFO: Epoch: 5, Batch[50/227], Train loss :1.808, Train acc: 0.6164535853251807
[2024-12-24 18:07:31] - INFO: Epoch: 5, Batch[51/227], Train loss :1.807, Train acc: 0.6082635983263598
[2024-12-24 18:07:33] - INFO: Epoch: 5, Batch[52/227], Train loss :1.602, Train acc: 0.6485260770975056
[2024-12-24 18:07:35] - INFO: Epoch: 5, Batch[53/227], Train loss :1.576, Train acc: 0.6413478012564249
[2024-12-24 18:07:37] - INFO: Epoch: 5, Batch[54/227], Train loss :1.731, Train acc: 0.6284090909090909
[2024-12-24 18:07:39] - INFO: Epoch: 5, Batch[55/227], Train loss :1.784, Train acc: 0.6205156950672646
[2024-12-24 18:07:41] - INFO: Epoch: 5, Batch[56/227], Train loss :1.712, Train acc: 0.6214324178782983
[2024-12-24 18:07:44] - INFO: Epoch: 5, Batch[57/227], Train loss :1.714, Train acc: 0.629374337221633
[2024-12-24 18:07:46] - INFO: Epoch: 5, Batch[58/227], Train loss :1.688, Train acc: 0.6308615049073064
[2024-12-24 18:07:48] - INFO: Epoch: 5, Batch[59/227], Train loss :1.620, Train acc: 0.6516656925774401
[2024-12-24 18:07:51] - INFO: Epoch: 5, Batch[60/227], Train loss :1.591, Train acc: 0.6492236917768832
[2024-12-24 18:07:53] - INFO: Epoch: 5, Batch[61/227], Train loss :1.639, Train acc: 0.6369747899159663
[2024-12-24 18:07:55] - INFO: Epoch: 5, Batch[62/227], Train loss :1.752, Train acc: 0.6169074371321562
[2024-12-24 18:07:57] - INFO: Epoch: 5, Batch[63/227], Train loss :1.655, Train acc: 0.6401326699834162
[2024-12-24 18:08:00] - INFO: Epoch: 5, Batch[64/227], Train loss :1.749, Train acc: 0.6202804746494067
[2024-12-24 18:08:02] - INFO: Epoch: 5, Batch[65/227], Train loss :1.665, Train acc: 0.6429378531073446
[2024-12-24 18:08:04] - INFO: Epoch: 5, Batch[66/227], Train loss :1.708, Train acc: 0.6497206703910614
[2024-12-24 18:08:06] - INFO: Epoch: 5, Batch[67/227], Train loss :1.555, Train acc: 0.6516916250693289
[2024-12-24 18:08:08] - INFO: Epoch: 5, Batch[68/227], Train loss :1.674, Train acc: 0.6301742919389978
[2024-12-24 18:08:11] - INFO: Epoch: 5, Batch[69/227], Train loss :1.641, Train acc: 0.6317218902770234
[2024-12-24 18:08:14] - INFO: Epoch: 5, Batch[70/227], Train loss :1.674, Train acc: 0.6388732394366197
[2024-12-24 18:08:16] - INFO: Epoch: 5, Batch[71/227], Train loss :1.740, Train acc: 0.6186534216335541
[2024-12-24 18:08:18] - INFO: Epoch: 5, Batch[72/227], Train loss :1.719, Train acc: 0.6393530997304582
[2024-12-24 18:08:21] - INFO: Epoch: 5, Batch[73/227], Train loss :1.703, Train acc: 0.6131386861313869
[2024-12-24 18:08:23] - INFO: Epoch: 5, Batch[74/227], Train loss :1.688, Train acc: 0.6365168539325843
[2024-12-24 18:08:25] - INFO: Epoch: 5, Batch[75/227], Train loss :1.739, Train acc: 0.6279712548369265
[2024-12-24 18:08:27] - INFO: Epoch: 5, Batch[76/227], Train loss :1.736, Train acc: 0.6143326039387309
[2024-12-24 18:08:30] - INFO: Epoch: 5, Batch[77/227], Train loss :1.712, Train acc: 0.6252860411899314
[2024-12-24 18:08:32] - INFO: Epoch: 5, Batch[78/227], Train loss :1.803, Train acc: 0.6098104793756968
[2024-12-24 18:08:34] - INFO: Epoch: 5, Batch[79/227], Train loss :1.721, Train acc: 0.6348476135710178
[2024-12-24 18:08:36] - INFO: Epoch: 5, Batch[80/227], Train loss :1.519, Train acc: 0.6565096952908587
[2024-12-24 18:08:38] - INFO: Epoch: 5, Batch[81/227], Train loss :1.769, Train acc: 0.6138344226579521
[2024-12-24 18:08:40] - INFO: Epoch: 5, Batch[82/227], Train loss :1.672, Train acc: 0.6355979786636721
[2024-12-24 18:08:42] - INFO: Epoch: 5, Batch[83/227], Train loss :1.711, Train acc: 0.6368684064408662
[2024-12-24 18:08:44] - INFO: Epoch: 5, Batch[84/227], Train loss :1.739, Train acc: 0.6161837069436851
[2024-12-24 18:08:46] - INFO: Epoch: 5, Batch[85/227], Train loss :1.636, Train acc: 0.6365650969529086
[2024-12-24 18:08:48] - INFO: Epoch: 5, Batch[86/227], Train loss :1.781, Train acc: 0.6139664804469274
[2024-12-24 18:08:52] - INFO: Epoch: 5, Batch[87/227], Train loss :1.649, Train acc: 0.6323529411764706
[2024-12-24 18:08:54] - INFO: Epoch: 5, Batch[88/227], Train loss :1.675, Train acc: 0.6222841225626741
[2024-12-24 18:08:56] - INFO: Epoch: 5, Batch[89/227], Train loss :1.657, Train acc: 0.6420811892510006
[2024-12-24 18:08:58] - INFO: Epoch: 5, Batch[90/227], Train loss :1.670, Train acc: 0.6240855374226224
[2024-12-24 18:09:00] - INFO: Epoch: 5, Batch[91/227], Train loss :1.715, Train acc: 0.6342281879194631
[2024-12-24 18:09:02] - INFO: Epoch: 5, Batch[92/227], Train loss :1.767, Train acc: 0.6288258208124652
[2024-12-24 18:09:05] - INFO: Epoch: 5, Batch[93/227], Train loss :1.838, Train acc: 0.6179713340683572
[2024-12-24 18:09:08] - INFO: Epoch: 5, Batch[94/227], Train loss :1.813, Train acc: 0.6122112211221122
[2024-12-24 18:09:10] - INFO: Epoch: 5, Batch[95/227], Train loss :1.816, Train acc: 0.6135857461024499
[2024-12-24 18:09:12] - INFO: Epoch: 5, Batch[96/227], Train loss :1.675, Train acc: 0.6285554935861685
[2024-12-24 18:09:14] - INFO: Epoch: 5, Batch[97/227], Train loss :1.685, Train acc: 0.6375207526286663
[2024-12-24 18:09:17] - INFO: Epoch: 5, Batch[98/227], Train loss :1.700, Train acc: 0.62348401323043
[2024-12-24 18:09:19] - INFO: Epoch: 5, Batch[99/227], Train loss :1.627, Train acc: 0.6404185022026432
[2024-12-24 18:09:21] - INFO: Epoch: 5, Batch[100/227], Train loss :1.704, Train acc: 0.6276483050847458
[2024-12-24 18:09:23] - INFO: Epoch: 5, Batch[101/227], Train loss :1.543, Train acc: 0.66535654126895
[2024-12-24 18:09:25] - INFO: Epoch: 5, Batch[102/227], Train loss :1.596, Train acc: 0.6514285714285715
[2024-12-24 18:09:27] - INFO: Epoch: 5, Batch[103/227], Train loss :1.708, Train acc: 0.6251366120218579
[2024-12-24 18:09:29] - INFO: Epoch: 5, Batch[104/227], Train loss :1.691, Train acc: 0.6323935876174682
[2024-12-24 18:09:31] - INFO: Epoch: 5, Batch[105/227], Train loss :1.738, Train acc: 0.6116764863417247
[2024-12-24 18:09:33] - INFO: Epoch: 5, Batch[106/227], Train loss :1.757, Train acc: 0.6282051282051282
[2024-12-24 18:09:35] - INFO: Epoch: 5, Batch[107/227], Train loss :1.670, Train acc: 0.6379498364231189
[2024-12-24 18:09:37] - INFO: Epoch: 5, Batch[108/227], Train loss :1.668, Train acc: 0.6354679802955665
[2024-12-24 18:09:40] - INFO: Epoch: 5, Batch[109/227], Train loss :1.773, Train acc: 0.6270822138635143
[2024-12-24 18:09:42] - INFO: Epoch: 5, Batch[110/227], Train loss :1.694, Train acc: 0.6359953703703703
[2024-12-24 18:09:44] - INFO: Epoch: 5, Batch[111/227], Train loss :1.756, Train acc: 0.6212857914640735
[2024-12-24 18:09:47] - INFO: Epoch: 5, Batch[112/227], Train loss :1.613, Train acc: 0.6418499717992103
[2024-12-24 18:09:49] - INFO: Epoch: 5, Batch[113/227], Train loss :1.746, Train acc: 0.6222606689734718
[2024-12-24 18:09:51] - INFO: Epoch: 5, Batch[114/227], Train loss :1.632, Train acc: 0.6300992282249173
[2024-12-24 18:09:53] - INFO: Epoch: 5, Batch[115/227], Train loss :1.718, Train acc: 0.6184631803628602
[2024-12-24 18:09:56] - INFO: Epoch: 5, Batch[116/227], Train loss :1.778, Train acc: 0.6143473570658037
[2024-12-24 18:09:58] - INFO: Epoch: 5, Batch[117/227], Train loss :1.670, Train acc: 0.6291105121293801
[2024-12-24 18:10:01] - INFO: Epoch: 5, Batch[118/227], Train loss :1.787, Train acc: 0.6151436031331593
[2024-12-24 18:10:03] - INFO: Epoch: 5, Batch[119/227], Train loss :1.614, Train acc: 0.6270096463022508
[2024-12-24 18:10:05] - INFO: Epoch: 5, Batch[120/227], Train loss :1.707, Train acc: 0.6275787187839305
[2024-12-24 18:10:08] - INFO: Epoch: 5, Batch[121/227], Train loss :1.708, Train acc: 0.6301524562394127
[2024-12-24 18:10:10] - INFO: Epoch: 5, Batch[122/227], Train loss :1.652, Train acc: 0.6423357664233577
[2024-12-24 18:10:12] - INFO: Epoch: 5, Batch[123/227], Train loss :1.649, Train acc: 0.6475722858701582
[2024-12-24 18:10:14] - INFO: Epoch: 5, Batch[124/227], Train loss :1.673, Train acc: 0.6362135388002201
[2024-12-24 18:10:17] - INFO: Epoch: 5, Batch[125/227], Train loss :1.598, Train acc: 0.6372767857142857
[2024-12-24 18:10:18] - INFO: Epoch: 5, Batch[126/227], Train loss :1.584, Train acc: 0.6433084434233199
[2024-12-24 18:10:21] - INFO: Epoch: 5, Batch[127/227], Train loss :1.733, Train acc: 0.6334792122538293
[2024-12-24 18:10:23] - INFO: Epoch: 5, Batch[128/227], Train loss :1.768, Train acc: 0.6116191500806886
[2024-12-24 18:10:26] - INFO: Epoch: 5, Batch[129/227], Train loss :1.745, Train acc: 0.619281045751634
[2024-12-24 18:10:28] - INFO: Epoch: 5, Batch[130/227], Train loss :1.696, Train acc: 0.6253326237360298
[2024-12-24 18:10:30] - INFO: Epoch: 5, Batch[131/227], Train loss :1.613, Train acc: 0.6400226757369615
[2024-12-24 18:10:32] - INFO: Epoch: 5, Batch[132/227], Train loss :1.712, Train acc: 0.6360110803324099
[2024-12-24 18:10:34] - INFO: Epoch: 5, Batch[133/227], Train loss :1.733, Train acc: 0.6337177375068643
[2024-12-24 18:10:36] - INFO: Epoch: 5, Batch[134/227], Train loss :1.655, Train acc: 0.6397550111358574
[2024-12-24 18:10:38] - INFO: Epoch: 5, Batch[135/227], Train loss :1.656, Train acc: 0.6394176931690929
[2024-12-24 18:10:40] - INFO: Epoch: 5, Batch[136/227], Train loss :1.633, Train acc: 0.6321243523316062
[2024-12-24 18:10:42] - INFO: Epoch: 5, Batch[137/227], Train loss :1.726, Train acc: 0.6273122959738846
[2024-12-24 18:10:44] - INFO: Epoch: 5, Batch[138/227], Train loss :1.772, Train acc: 0.6168999481596682
[2024-12-24 18:10:46] - INFO: Epoch: 5, Batch[139/227], Train loss :1.714, Train acc: 0.6278130409694171
[2024-12-24 18:10:49] - INFO: Epoch: 5, Batch[140/227], Train loss :1.653, Train acc: 0.6390270867882808
[2024-12-24 18:10:51] - INFO: Epoch: 5, Batch[141/227], Train loss :1.684, Train acc: 0.6270903010033445
[2024-12-24 18:10:53] - INFO: Epoch: 5, Batch[142/227], Train loss :1.790, Train acc: 0.6190214403518417
[2024-12-24 18:10:56] - INFO: Epoch: 5, Batch[143/227], Train loss :1.684, Train acc: 0.637308533916849
[2024-12-24 18:10:58] - INFO: Epoch: 5, Batch[144/227], Train loss :1.630, Train acc: 0.6366982124079916
[2024-12-24 18:11:01] - INFO: Epoch: 5, Batch[145/227], Train loss :1.770, Train acc: 0.6303291958985429
[2024-12-24 18:11:03] - INFO: Epoch: 5, Batch[146/227], Train loss :1.813, Train acc: 0.6190992946283234
[2024-12-24 18:11:05] - INFO: Epoch: 5, Batch[147/227], Train loss :1.766, Train acc: 0.6344262295081967
[2024-12-24 18:11:07] - INFO: Epoch: 5, Batch[148/227], Train loss :1.759, Train acc: 0.6247203579418344
[2024-12-24 18:11:09] - INFO: Epoch: 5, Batch[149/227], Train loss :1.732, Train acc: 0.625
[2024-12-24 18:11:11] - INFO: Epoch: 5, Batch[150/227], Train loss :1.727, Train acc: 0.6199563794983642
[2024-12-24 18:11:14] - INFO: Epoch: 5, Batch[151/227], Train loss :1.739, Train acc: 0.6361619523017193
[2024-12-24 18:11:16] - INFO: Epoch: 5, Batch[152/227], Train loss :1.708, Train acc: 0.6299168975069253
[2024-12-24 18:11:19] - INFO: Epoch: 5, Batch[153/227], Train loss :1.666, Train acc: 0.6491841491841492
[2024-12-24 18:11:21] - INFO: Epoch: 5, Batch[154/227], Train loss :1.639, Train acc: 0.6372275013974288
[2024-12-24 18:11:24] - INFO: Epoch: 5, Batch[155/227], Train loss :1.749, Train acc: 0.6281908990011099
[2024-12-24 18:11:26] - INFO: Epoch: 5, Batch[156/227], Train loss :1.694, Train acc: 0.6311787072243346
[2024-12-24 18:11:28] - INFO: Epoch: 5, Batch[157/227], Train loss :1.678, Train acc: 0.6297129994372538
[2024-12-24 18:11:30] - INFO: Epoch: 5, Batch[158/227], Train loss :1.780, Train acc: 0.6257634647418101
[2024-12-24 18:11:31] - INFO: Epoch: 5, Batch[159/227], Train loss :1.783, Train acc: 0.6219303255282695
[2024-12-24 18:11:34] - INFO: Epoch: 5, Batch[160/227], Train loss :1.624, Train acc: 0.6500847936687394
[2024-12-24 18:11:36] - INFO: Epoch: 5, Batch[161/227], Train loss :1.696, Train acc: 0.6283632286995515
[2024-12-24 18:11:38] - INFO: Epoch: 5, Batch[162/227], Train loss :1.818, Train acc: 0.6014023732470335
[2024-12-24 18:11:40] - INFO: Epoch: 5, Batch[163/227], Train loss :1.733, Train acc: 0.6176148796498906
[2024-12-24 18:11:42] - INFO: Epoch: 5, Batch[164/227], Train loss :1.848, Train acc: 0.6107123136388736
[2024-12-24 18:11:44] - INFO: Epoch: 5, Batch[165/227], Train loss :1.830, Train acc: 0.6101333333333333
[2024-12-24 18:11:47] - INFO: Epoch: 5, Batch[166/227], Train loss :1.756, Train acc: 0.6211213935764834
[2024-12-24 18:11:49] - INFO: Epoch: 5, Batch[167/227], Train loss :1.665, Train acc: 0.6485310119695321
[2024-12-24 18:11:51] - INFO: Epoch: 5, Batch[168/227], Train loss :1.678, Train acc: 0.6466821885913854
[2024-12-24 18:11:54] - INFO: Epoch: 5, Batch[169/227], Train loss :1.700, Train acc: 0.6277777777777778
[2024-12-24 18:11:56] - INFO: Epoch: 5, Batch[170/227], Train loss :1.774, Train acc: 0.6246560264171711
[2024-12-24 18:11:58] - INFO: Epoch: 5, Batch[171/227], Train loss :1.807, Train acc: 0.6090858104318564
[2024-12-24 18:12:00] - INFO: Epoch: 5, Batch[172/227], Train loss :1.747, Train acc: 0.6274065685164213
[2024-12-24 18:12:02] - INFO: Epoch: 5, Batch[173/227], Train loss :1.710, Train acc: 0.6367638965327462
[2024-12-24 18:12:04] - INFO: Epoch: 5, Batch[174/227], Train loss :1.674, Train acc: 0.6355979786636721
[2024-12-24 18:12:07] - INFO: Epoch: 5, Batch[175/227], Train loss :1.658, Train acc: 0.6397616468039004
[2024-12-24 18:12:09] - INFO: Epoch: 5, Batch[176/227], Train loss :1.726, Train acc: 0.6194690265486725
[2024-12-24 18:12:11] - INFO: Epoch: 5, Batch[177/227], Train loss :1.746, Train acc: 0.6217105263157895
[2024-12-24 18:12:14] - INFO: Epoch: 5, Batch[178/227], Train loss :1.603, Train acc: 0.6343201754385965
[2024-12-24 18:12:15] - INFO: Epoch: 5, Batch[179/227], Train loss :1.678, Train acc: 0.6332425068119891
[2024-12-24 18:12:18] - INFO: Epoch: 5, Batch[180/227], Train loss :1.777, Train acc: 0.6150881057268722
[2024-12-24 18:12:21] - INFO: Epoch: 5, Batch[181/227], Train loss :1.670, Train acc: 0.6304585152838428
[2024-12-24 18:12:23] - INFO: Epoch: 5, Batch[182/227], Train loss :1.671, Train acc: 0.6487804878048781
[2024-12-24 18:12:25] - INFO: Epoch: 5, Batch[183/227], Train loss :1.646, Train acc: 0.6336418072945019
[2024-12-24 18:12:28] - INFO: Epoch: 5, Batch[184/227], Train loss :1.741, Train acc: 0.6193619361936193
[2024-12-24 18:12:30] - INFO: Epoch: 5, Batch[185/227], Train loss :1.621, Train acc: 0.6520998864926221
[2024-12-24 18:12:32] - INFO: Epoch: 5, Batch[186/227], Train loss :1.752, Train acc: 0.6047278724573941
[2024-12-24 18:12:34] - INFO: Epoch: 5, Batch[187/227], Train loss :1.774, Train acc: 0.6176808266360505
[2024-12-24 18:12:36] - INFO: Epoch: 5, Batch[188/227], Train loss :1.831, Train acc: 0.6034858387799564
[2024-12-24 18:12:39] - INFO: Epoch: 5, Batch[189/227], Train loss :1.791, Train acc: 0.618192026951151
[2024-12-24 18:12:42] - INFO: Epoch: 5, Batch[190/227], Train loss :1.625, Train acc: 0.6455840455840456
[2024-12-24 18:12:43] - INFO: Epoch: 5, Batch[191/227], Train loss :1.679, Train acc: 0.6428571428571429
[2024-12-24 18:12:46] - INFO: Epoch: 5, Batch[192/227], Train loss :1.686, Train acc: 0.6388274336283186
[2024-12-24 18:12:48] - INFO: Epoch: 5, Batch[193/227], Train loss :1.767, Train acc: 0.6301742919389978
[2024-12-24 18:12:51] - INFO: Epoch: 5, Batch[194/227], Train loss :1.826, Train acc: 0.6007604562737643
[2024-12-24 18:12:54] - INFO: Epoch: 5, Batch[195/227], Train loss :1.621, Train acc: 0.6572769953051644
[2024-12-24 18:12:56] - INFO: Epoch: 5, Batch[196/227], Train loss :1.740, Train acc: 0.6440771349862259
[2024-12-24 18:12:58] - INFO: Epoch: 5, Batch[197/227], Train loss :1.872, Train acc: 0.6060436485730274
[2024-12-24 18:13:00] - INFO: Epoch: 5, Batch[198/227], Train loss :1.761, Train acc: 0.6241935483870967
[2024-12-24 18:13:02] - INFO: Epoch: 5, Batch[199/227], Train loss :1.785, Train acc: 0.6122228231476474
[2024-12-24 18:13:04] - INFO: Epoch: 5, Batch[200/227], Train loss :1.712, Train acc: 0.6311159978009896
[2024-12-24 18:13:06] - INFO: Epoch: 5, Batch[201/227], Train loss :1.750, Train acc: 0.6215772179627601
[2024-12-24 18:13:08] - INFO: Epoch: 5, Batch[202/227], Train loss :1.829, Train acc: 0.6126568466993999
[2024-12-24 18:13:11] - INFO: Epoch: 5, Batch[203/227], Train loss :1.827, Train acc: 0.6114346214928533
[2024-12-24 18:13:12] - INFO: Epoch: 5, Batch[204/227], Train loss :1.649, Train acc: 0.6402144772117963
[2024-12-24 18:13:14] - INFO: Epoch: 5, Batch[205/227], Train loss :1.758, Train acc: 0.6207881210736722
[2024-12-24 18:13:16] - INFO: Epoch: 5, Batch[206/227], Train loss :1.936, Train acc: 0.5890557939914163
[2024-12-24 18:13:18] - INFO: Epoch: 5, Batch[207/227], Train loss :1.684, Train acc: 0.6308492201039861
[2024-12-24 18:13:21] - INFO: Epoch: 5, Batch[208/227], Train loss :1.780, Train acc: 0.6056644880174292
[2024-12-24 18:13:23] - INFO: Epoch: 5, Batch[209/227], Train loss :1.644, Train acc: 0.6443682104059463
[2024-12-24 18:13:25] - INFO: Epoch: 5, Batch[210/227], Train loss :1.765, Train acc: 0.624859392575928
[2024-12-24 18:13:27] - INFO: Epoch: 5, Batch[211/227], Train loss :1.634, Train acc: 0.6496519721577726
[2024-12-24 18:13:28] - INFO: Epoch: 5, Batch[212/227], Train loss :1.757, Train acc: 0.6268571428571429
[2024-12-24 18:13:31] - INFO: Epoch: 5, Batch[213/227], Train loss :1.731, Train acc: 0.6190744920993227
[2024-12-24 18:13:33] - INFO: Epoch: 5, Batch[214/227], Train loss :1.659, Train acc: 0.6358126721763085
[2024-12-24 18:13:35] - INFO: Epoch: 5, Batch[215/227], Train loss :1.922, Train acc: 0.5958034235229155
[2024-12-24 18:13:37] - INFO: Epoch: 5, Batch[216/227], Train loss :1.807, Train acc: 0.6195230171935663
[2024-12-24 18:13:40] - INFO: Epoch: 5, Batch[217/227], Train loss :1.719, Train acc: 0.6197718631178707
[2024-12-24 18:13:41] - INFO: Epoch: 5, Batch[218/227], Train loss :1.726, Train acc: 0.6269972451790634
[2024-12-24 18:13:43] - INFO: Epoch: 5, Batch[219/227], Train loss :1.682, Train acc: 0.6383981154299175
[2024-12-24 18:13:46] - INFO: Epoch: 5, Batch[220/227], Train loss :1.604, Train acc: 0.6465116279069767
[2024-12-24 18:13:48] - INFO: Epoch: 5, Batch[221/227], Train loss :1.812, Train acc: 0.6148028092922745
[2024-12-24 18:13:50] - INFO: Epoch: 5, Batch[222/227], Train loss :1.718, Train acc: 0.6292197011621472
[2024-12-24 18:13:53] - INFO: Epoch: 5, Batch[223/227], Train loss :1.755, Train acc: 0.6321776814734561
[2024-12-24 18:13:55] - INFO: Epoch: 5, Batch[224/227], Train loss :1.739, Train acc: 0.6280522430437252
[2024-12-24 18:13:56] - INFO: Epoch: 5, Batch[225/227], Train loss :1.692, Train acc: 0.6421596783457783
[2024-12-24 18:13:58] - INFO: Epoch: 5, Batch[226/227], Train loss :1.755, Train acc: 0.6313686313686314
[2024-12-24 18:13:58] - INFO: Epoch: 5, Train loss: 1.704, Epoch time = 504.884s
[2024-12-24 18:14:03] - INFO: Accuracy on validation0.560
[2024-12-24 18:14:06] - INFO: Epoch: 6, Batch[0/227], Train loss :1.599, Train acc: 0.6470588235294118
[2024-12-24 18:14:09] - INFO: Epoch: 6, Batch[1/227], Train loss :1.636, Train acc: 0.6409116175653141
[2024-12-24 18:14:11] - INFO: Epoch: 6, Batch[2/227], Train loss :1.516, Train acc: 0.6510710259301015
[2024-12-24 18:14:14] - INFO: Epoch: 6, Batch[3/227], Train loss :1.627, Train acc: 0.6351423965609887
[2024-12-24 18:14:16] - INFO: Epoch: 6, Batch[4/227], Train loss :1.671, Train acc: 0.6275644397685429
[2024-12-24 18:14:18] - INFO: Epoch: 6, Batch[5/227], Train loss :1.549, Train acc: 0.6423240938166311
[2024-12-24 18:14:21] - INFO: Epoch: 6, Batch[6/227], Train loss :1.555, Train acc: 0.6457204767063922
[2024-12-24 18:14:23] - INFO: Epoch: 6, Batch[7/227], Train loss :1.610, Train acc: 0.6378845116028062
[2024-12-24 18:14:25] - INFO: Epoch: 6, Batch[8/227], Train loss :1.669, Train acc: 0.6299006795608991
[2024-12-24 18:14:28] - INFO: Epoch: 6, Batch[9/227], Train loss :1.574, Train acc: 0.6471610660486674
[2024-12-24 18:14:30] - INFO: Epoch: 6, Batch[10/227], Train loss :1.587, Train acc: 0.6541436464088398
[2024-12-24 18:14:32] - INFO: Epoch: 6, Batch[11/227], Train loss :1.675, Train acc: 0.613588110403397
[2024-12-24 18:14:34] - INFO: Epoch: 6, Batch[12/227], Train loss :1.705, Train acc: 0.6293895191788222
[2024-12-24 18:14:36] - INFO: Epoch: 6, Batch[13/227], Train loss :1.522, Train acc: 0.6541689983212088
[2024-12-24 18:14:39] - INFO: Epoch: 6, Batch[14/227], Train loss :1.567, Train acc: 0.651175505740842
[2024-12-24 18:14:41] - INFO: Epoch: 6, Batch[15/227], Train loss :1.645, Train acc: 0.6363636363636364
[2024-12-24 18:14:43] - INFO: Epoch: 6, Batch[16/227], Train loss :1.622, Train acc: 0.6494845360824743
[2024-12-24 18:14:45] - INFO: Epoch: 6, Batch[17/227], Train loss :1.501, Train acc: 0.6564718732932824
[2024-12-24 18:14:47] - INFO: Epoch: 6, Batch[18/227], Train loss :1.716, Train acc: 0.6054897739504844
[2024-12-24 18:14:50] - INFO: Epoch: 6, Batch[19/227], Train loss :1.568, Train acc: 0.6544943820224719
[2024-12-24 18:14:53] - INFO: Epoch: 6, Batch[20/227], Train loss :1.562, Train acc: 0.6547884187082406
[2024-12-24 18:14:55] - INFO: Epoch: 6, Batch[21/227], Train loss :1.620, Train acc: 0.6367591082109842
[2024-12-24 18:14:57] - INFO: Epoch: 6, Batch[22/227], Train loss :1.577, Train acc: 0.6591802358225716
[2024-12-24 18:14:59] - INFO: Epoch: 6, Batch[23/227], Train loss :1.616, Train acc: 0.6392333709131905
[2024-12-24 18:15:01] - INFO: Epoch: 6, Batch[24/227], Train loss :1.588, Train acc: 0.6500566251415628
[2024-12-24 18:15:03] - INFO: Epoch: 6, Batch[25/227], Train loss :1.546, Train acc: 0.6666666666666666
[2024-12-24 18:15:06] - INFO: Epoch: 6, Batch[26/227], Train loss :1.567, Train acc: 0.6598564329099945
[2024-12-24 18:15:09] - INFO: Epoch: 6, Batch[27/227], Train loss :1.570, Train acc: 0.6382047071702244
[2024-12-24 18:15:12] - INFO: Epoch: 6, Batch[28/227], Train loss :1.497, Train acc: 0.655819084390513
[2024-12-24 18:15:14] - INFO: Epoch: 6, Batch[29/227], Train loss :1.555, Train acc: 0.6404682274247492
[2024-12-24 18:15:16] - INFO: Epoch: 6, Batch[30/227], Train loss :1.610, Train acc: 0.6339977851605758
[2024-12-24 18:15:18] - INFO: Epoch: 6, Batch[31/227], Train loss :1.544, Train acc: 0.6456473214285714
[2024-12-24 18:15:21] - INFO: Epoch: 6, Batch[32/227], Train loss :1.743, Train acc: 0.6203848153926157
[2024-12-24 18:15:23] - INFO: Epoch: 6, Batch[33/227], Train loss :1.512, Train acc: 0.6717724288840262
[2024-12-24 18:15:25] - INFO: Epoch: 6, Batch[34/227], Train loss :1.591, Train acc: 0.6422366992399565
[2024-12-24 18:15:28] - INFO: Epoch: 6, Batch[35/227], Train loss :1.493, Train acc: 0.6668561682774303
[2024-12-24 18:15:30] - INFO: Epoch: 6, Batch[36/227], Train loss :1.640, Train acc: 0.6327000575705239
[2024-12-24 18:15:32] - INFO: Epoch: 6, Batch[37/227], Train loss :1.532, Train acc: 0.6564132327336042
[2024-12-24 18:15:34] - INFO: Epoch: 6, Batch[38/227], Train loss :1.621, Train acc: 0.6246705324196099
[2024-12-24 18:15:35] - INFO: Epoch: 6, Batch[39/227], Train loss :1.458, Train acc: 0.6834112149532711
[2024-12-24 18:15:38] - INFO: Epoch: 6, Batch[40/227], Train loss :1.535, Train acc: 0.6524701873935264
[2024-12-24 18:15:40] - INFO: Epoch: 6, Batch[41/227], Train loss :1.644, Train acc: 0.6387640449438202
[2024-12-24 18:15:42] - INFO: Epoch: 6, Batch[42/227], Train loss :1.657, Train acc: 0.63931718061674
[2024-12-24 18:15:45] - INFO: Epoch: 6, Batch[43/227], Train loss :1.571, Train acc: 0.6534090909090909
[2024-12-24 18:15:47] - INFO: Epoch: 6, Batch[44/227], Train loss :1.544, Train acc: 0.6512434933487565
[2024-12-24 18:15:50] - INFO: Epoch: 6, Batch[45/227], Train loss :1.573, Train acc: 0.6475675675675676
[2024-12-24 18:15:52] - INFO: Epoch: 6, Batch[46/227], Train loss :1.634, Train acc: 0.6392092257001647
[2024-12-24 18:15:55] - INFO: Epoch: 6, Batch[47/227], Train loss :1.604, Train acc: 0.6491728465487735
[2024-12-24 18:15:57] - INFO: Epoch: 6, Batch[48/227], Train loss :1.577, Train acc: 0.6454906409529212
[2024-12-24 18:15:59] - INFO: Epoch: 6, Batch[49/227], Train loss :1.661, Train acc: 0.645125348189415
[2024-12-24 18:16:01] - INFO: Epoch: 6, Batch[50/227], Train loss :1.653, Train acc: 0.6204458945078847
[2024-12-24 18:16:03] - INFO: Epoch: 6, Batch[51/227], Train loss :1.607, Train acc: 0.6523255813953488
[2024-12-24 18:16:05] - INFO: Epoch: 6, Batch[52/227], Train loss :1.743, Train acc: 0.636215334420881
[2024-12-24 18:16:07] - INFO: Epoch: 6, Batch[53/227], Train loss :1.595, Train acc: 0.6544532130777903
[2024-12-24 18:16:09] - INFO: Epoch: 6, Batch[54/227], Train loss :1.611, Train acc: 0.6447582835415535
[2024-12-24 18:16:11] - INFO: Epoch: 6, Batch[55/227], Train loss :1.681, Train acc: 0.6274738067520372
[2024-12-24 18:16:14] - INFO: Epoch: 6, Batch[56/227], Train loss :1.547, Train acc: 0.6503884572697003
[2024-12-24 18:16:17] - INFO: Epoch: 6, Batch[57/227], Train loss :1.626, Train acc: 0.6418478260869566
[2024-12-24 18:16:19] - INFO: Epoch: 6, Batch[58/227], Train loss :1.592, Train acc: 0.6517806670435274
[2024-12-24 18:16:21] - INFO: Epoch: 6, Batch[59/227], Train loss :1.599, Train acc: 0.6571753986332574
[2024-12-24 18:16:23] - INFO: Epoch: 6, Batch[60/227], Train loss :1.595, Train acc: 0.6405479452054794
[2024-12-24 18:16:25] - INFO: Epoch: 6, Batch[61/227], Train loss :1.613, Train acc: 0.6483214089157953
[2024-12-24 18:16:27] - INFO: Epoch: 6, Batch[62/227], Train loss :1.572, Train acc: 0.655793025871766
[2024-12-24 18:16:29] - INFO: Epoch: 6, Batch[63/227], Train loss :1.517, Train acc: 0.6557468073292615
[2024-12-24 18:16:32] - INFO: Epoch: 6, Batch[64/227], Train loss :1.539, Train acc: 0.6491228070175439
[2024-12-24 18:16:34] - INFO: Epoch: 6, Batch[65/227], Train loss :1.524, Train acc: 0.6530160486995019
[2024-12-24 18:16:36] - INFO: Epoch: 6, Batch[66/227], Train loss :1.619, Train acc: 0.6346578366445916
[2024-12-24 18:16:39] - INFO: Epoch: 6, Batch[67/227], Train loss :1.681, Train acc: 0.6230769230769231
[2024-12-24 18:16:41] - INFO: Epoch: 6, Batch[68/227], Train loss :1.720, Train acc: 0.6225910064239829
[2024-12-24 18:16:43] - INFO: Epoch: 6, Batch[69/227], Train loss :1.568, Train acc: 0.6612995974698103
[2024-12-24 18:16:44] - INFO: Epoch: 6, Batch[70/227], Train loss :1.550, Train acc: 0.6543075245365322
[2024-12-24 18:16:47] - INFO: Epoch: 6, Batch[71/227], Train loss :1.558, Train acc: 0.6470270270270271
[2024-12-24 18:16:50] - INFO: Epoch: 6, Batch[72/227], Train loss :1.576, Train acc: 0.6504109589041096
[2024-12-24 18:16:53] - INFO: Epoch: 6, Batch[73/227], Train loss :1.451, Train acc: 0.6745334796926454
[2024-12-24 18:16:55] - INFO: Epoch: 6, Batch[74/227], Train loss :1.675, Train acc: 0.6335403726708074
[2024-12-24 18:16:58] - INFO: Epoch: 6, Batch[75/227], Train loss :1.596, Train acc: 0.6668472372697725
[2024-12-24 18:17:00] - INFO: Epoch: 6, Batch[76/227], Train loss :1.639, Train acc: 0.631948192120885
[2024-12-24 18:17:02] - INFO: Epoch: 6, Batch[77/227], Train loss :1.641, Train acc: 0.6307352128247651
[2024-12-24 18:17:04] - INFO: Epoch: 6, Batch[78/227], Train loss :1.736, Train acc: 0.6223126089482859
[2024-12-24 18:17:06] - INFO: Epoch: 6, Batch[79/227], Train loss :1.613, Train acc: 0.6377551020408163
[2024-12-24 18:17:09] - INFO: Epoch: 6, Batch[80/227], Train loss :1.696, Train acc: 0.6216356107660456
[2024-12-24 18:17:11] - INFO: Epoch: 6, Batch[81/227], Train loss :1.653, Train acc: 0.6331049024775962
[2024-12-24 18:17:13] - INFO: Epoch: 6, Batch[82/227], Train loss :1.711, Train acc: 0.6326869806094183
[2024-12-24 18:17:16] - INFO: Epoch: 6, Batch[83/227], Train loss :1.660, Train acc: 0.6301969365426696
[2024-12-24 18:17:18] - INFO: Epoch: 6, Batch[84/227], Train loss :1.672, Train acc: 0.6388583973655324
[2024-12-24 18:17:20] - INFO: Epoch: 6, Batch[85/227], Train loss :1.547, Train acc: 0.6578220011055832
[2024-12-24 18:17:22] - INFO: Epoch: 6, Batch[86/227], Train loss :1.729, Train acc: 0.612783940834654
[2024-12-24 18:17:24] - INFO: Epoch: 6, Batch[87/227], Train loss :1.695, Train acc: 0.6338582677165354
[2024-12-24 18:17:26] - INFO: Epoch: 6, Batch[88/227], Train loss :1.501, Train acc: 0.6519799219185722
[2024-12-24 18:17:28] - INFO: Epoch: 6, Batch[89/227], Train loss :1.609, Train acc: 0.6389195148842337
[2024-12-24 18:17:31] - INFO: Epoch: 6, Batch[90/227], Train loss :1.686, Train acc: 0.6303030303030303
[2024-12-24 18:17:33] - INFO: Epoch: 6, Batch[91/227], Train loss :1.621, Train acc: 0.6384489350081922
[2024-12-24 18:17:35] - INFO: Epoch: 6, Batch[92/227], Train loss :1.669, Train acc: 0.6351720371381758
[2024-12-24 18:17:37] - INFO: Epoch: 6, Batch[93/227], Train loss :1.670, Train acc: 0.6223463687150838
[2024-12-24 18:17:40] - INFO: Epoch: 6, Batch[94/227], Train loss :1.490, Train acc: 0.656657223796034
[2024-12-24 18:17:42] - INFO: Epoch: 6, Batch[95/227], Train loss :1.570, Train acc: 0.6597582037996546
[2024-12-24 18:17:44] - INFO: Epoch: 6, Batch[96/227], Train loss :1.656, Train acc: 0.628792057363486
[2024-12-24 18:17:47] - INFO: Epoch: 6, Batch[97/227], Train loss :1.576, Train acc: 0.6560646900269542
[2024-12-24 18:17:49] - INFO: Epoch: 6, Batch[98/227], Train loss :1.642, Train acc: 0.6340352874217416
[2024-12-24 18:17:52] - INFO: Epoch: 6, Batch[99/227], Train loss :1.538, Train acc: 0.6503340757238307
[2024-12-24 18:17:54] - INFO: Epoch: 6, Batch[100/227], Train loss :1.650, Train acc: 0.6406581740976646
[2024-12-24 18:17:56] - INFO: Epoch: 6, Batch[101/227], Train loss :1.598, Train acc: 0.6450531022917831
[2024-12-24 18:17:58] - INFO: Epoch: 6, Batch[102/227], Train loss :1.604, Train acc: 0.6317204301075269
[2024-12-24 18:18:00] - INFO: Epoch: 6, Batch[103/227], Train loss :1.695, Train acc: 0.628804347826087
[2024-12-24 18:18:02] - INFO: Epoch: 6, Batch[104/227], Train loss :1.636, Train acc: 0.6519774011299435
[2024-12-24 18:18:04] - INFO: Epoch: 6, Batch[105/227], Train loss :1.734, Train acc: 0.6236798221234019
[2024-12-24 18:18:07] - INFO: Epoch: 6, Batch[106/227], Train loss :1.592, Train acc: 0.6422991071428571
[2024-12-24 18:18:09] - INFO: Epoch: 6, Batch[107/227], Train loss :1.659, Train acc: 0.6245059288537549
[2024-12-24 18:18:11] - INFO: Epoch: 6, Batch[108/227], Train loss :1.608, Train acc: 0.6560264171711613
[2024-12-24 18:18:13] - INFO: Epoch: 6, Batch[109/227], Train loss :1.642, Train acc: 0.6423804226918799
[2024-12-24 18:18:15] - INFO: Epoch: 6, Batch[110/227], Train loss :1.740, Train acc: 0.6227045075125208
[2024-12-24 18:18:17] - INFO: Epoch: 6, Batch[111/227], Train loss :1.661, Train acc: 0.6343732895457034
[2024-12-24 18:18:19] - INFO: Epoch: 6, Batch[112/227], Train loss :1.808, Train acc: 0.6079484425349087
[2024-12-24 18:18:22] - INFO: Epoch: 6, Batch[113/227], Train loss :1.763, Train acc: 0.6214084507042253
[2024-12-24 18:18:24] - INFO: Epoch: 6, Batch[114/227], Train loss :1.581, Train acc: 0.6416758544652701
[2024-12-24 18:18:26] - INFO: Epoch: 6, Batch[115/227], Train loss :1.639, Train acc: 0.6313725490196078
[2024-12-24 18:18:27] - INFO: Epoch: 6, Batch[116/227], Train loss :1.644, Train acc: 0.6304952698942682
[2024-12-24 18:18:30] - INFO: Epoch: 6, Batch[117/227], Train loss :1.692, Train acc: 0.6330734966592427
[2024-12-24 18:18:32] - INFO: Epoch: 6, Batch[118/227], Train loss :1.536, Train acc: 0.6553214478660183
[2024-12-24 18:18:34] - INFO: Epoch: 6, Batch[119/227], Train loss :1.545, Train acc: 0.6457750419697817
[2024-12-24 18:18:36] - INFO: Epoch: 6, Batch[120/227], Train loss :1.699, Train acc: 0.6352235550708834
[2024-12-24 18:18:38] - INFO: Epoch: 6, Batch[121/227], Train loss :1.666, Train acc: 0.6257309941520468
[2024-12-24 18:18:40] - INFO: Epoch: 6, Batch[122/227], Train loss :1.741, Train acc: 0.6257703081232493
[2024-12-24 18:18:43] - INFO: Epoch: 6, Batch[123/227], Train loss :1.688, Train acc: 0.6205931729155009
[2024-12-24 18:18:45] - INFO: Epoch: 6, Batch[124/227], Train loss :1.662, Train acc: 0.6331049024775962
[2024-12-24 18:18:48] - INFO: Epoch: 6, Batch[125/227], Train loss :1.700, Train acc: 0.6219912472647703
[2024-12-24 18:18:51] - INFO: Epoch: 6, Batch[126/227], Train loss :1.595, Train acc: 0.6382369776760161
[2024-12-24 18:18:53] - INFO: Epoch: 6, Batch[127/227], Train loss :1.567, Train acc: 0.6526195899772209
[2024-12-24 18:18:55] - INFO: Epoch: 6, Batch[128/227], Train loss :1.582, Train acc: 0.6521997621878716
[2024-12-24 18:18:57] - INFO: Epoch: 6, Batch[129/227], Train loss :1.681, Train acc: 0.6394444444444445
[2024-12-24 18:19:00] - INFO: Epoch: 6, Batch[130/227], Train loss :1.796, Train acc: 0.6069114470842333
[2024-12-24 18:19:02] - INFO: Epoch: 6, Batch[131/227], Train loss :1.655, Train acc: 0.6368983957219251
[2024-12-24 18:19:04] - INFO: Epoch: 6, Batch[132/227], Train loss :1.511, Train acc: 0.6568013659647126
[2024-12-24 18:19:07] - INFO: Epoch: 6, Batch[133/227], Train loss :1.703, Train acc: 0.6165075800112296
[2024-12-24 18:19:09] - INFO: Epoch: 6, Batch[134/227], Train loss :1.659, Train acc: 0.635809312638581
[2024-12-24 18:19:11] - INFO: Epoch: 6, Batch[135/227], Train loss :1.657, Train acc: 0.6354570637119114
[2024-12-24 18:19:14] - INFO: Epoch: 6, Batch[136/227], Train loss :1.691, Train acc: 0.6175963197239793
[2024-12-24 18:19:16] - INFO: Epoch: 6, Batch[137/227], Train loss :1.539, Train acc: 0.6563953488372093
[2024-12-24 18:19:18] - INFO: Epoch: 6, Batch[138/227], Train loss :1.637, Train acc: 0.6300164925783397
[2024-12-24 18:19:20] - INFO: Epoch: 6, Batch[139/227], Train loss :1.633, Train acc: 0.6313768513439386
[2024-12-24 18:19:22] - INFO: Epoch: 6, Batch[140/227], Train loss :1.647, Train acc: 0.6446991404011462
[2024-12-24 18:19:24] - INFO: Epoch: 6, Batch[141/227], Train loss :1.666, Train acc: 0.6354679802955665
[2024-12-24 18:19:27] - INFO: Epoch: 6, Batch[142/227], Train loss :1.704, Train acc: 0.6283333333333333
[2024-12-24 18:19:29] - INFO: Epoch: 6, Batch[143/227], Train loss :1.578, Train acc: 0.6505073280721533
[2024-12-24 18:19:31] - INFO: Epoch: 6, Batch[144/227], Train loss :1.691, Train acc: 0.6267409470752089
[2024-12-24 18:19:33] - INFO: Epoch: 6, Batch[145/227], Train loss :1.720, Train acc: 0.6307099614749587
[2024-12-24 18:19:36] - INFO: Epoch: 6, Batch[146/227], Train loss :1.746, Train acc: 0.6061899679829242
[2024-12-24 18:19:38] - INFO: Epoch: 6, Batch[147/227], Train loss :1.615, Train acc: 0.6364139457664637
[2024-12-24 18:19:40] - INFO: Epoch: 6, Batch[148/227], Train loss :1.664, Train acc: 0.6388577827547592
[2024-12-24 18:19:42] - INFO: Epoch: 6, Batch[149/227], Train loss :1.603, Train acc: 0.6464148877941981
[2024-12-24 18:19:44] - INFO: Epoch: 6, Batch[150/227], Train loss :1.569, Train acc: 0.631461923290717
[2024-12-24 18:19:46] - INFO: Epoch: 6, Batch[151/227], Train loss :1.658, Train acc: 0.6458449525934189
[2024-12-24 18:19:48] - INFO: Epoch: 6, Batch[152/227], Train loss :1.637, Train acc: 0.6401557285873193
[2024-12-24 18:19:51] - INFO: Epoch: 6, Batch[153/227], Train loss :1.665, Train acc: 0.6360022714366838
[2024-12-24 18:19:54] - INFO: Epoch: 6, Batch[154/227], Train loss :1.706, Train acc: 0.6317787418655098
[2024-12-24 18:19:56] - INFO: Epoch: 6, Batch[155/227], Train loss :1.720, Train acc: 0.6207849640685461
[2024-12-24 18:19:58] - INFO: Epoch: 6, Batch[156/227], Train loss :1.642, Train acc: 0.6192109777015438
[2024-12-24 18:20:00] - INFO: Epoch: 6, Batch[157/227], Train loss :1.724, Train acc: 0.6211477151965994
[2024-12-24 18:20:02] - INFO: Epoch: 6, Batch[158/227], Train loss :1.510, Train acc: 0.6697247706422018
[2024-12-24 18:20:04] - INFO: Epoch: 6, Batch[159/227], Train loss :1.681, Train acc: 0.617908787541713
[2024-12-24 18:20:07] - INFO: Epoch: 6, Batch[160/227], Train loss :1.651, Train acc: 0.6402203856749311
[2024-12-24 18:20:10] - INFO: Epoch: 6, Batch[161/227], Train loss :1.774, Train acc: 0.6180594503645541
[2024-12-24 18:20:12] - INFO: Epoch: 6, Batch[162/227], Train loss :1.704, Train acc: 0.6107419712070875
[2024-12-24 18:20:14] - INFO: Epoch: 6, Batch[163/227], Train loss :1.762, Train acc: 0.6083838940981798
[2024-12-24 18:20:17] - INFO: Epoch: 6, Batch[164/227], Train loss :1.683, Train acc: 0.628775398132894
[2024-12-24 18:20:18] - INFO: Epoch: 6, Batch[165/227], Train loss :1.687, Train acc: 0.6313159355913381
[2024-12-24 18:20:21] - INFO: Epoch: 6, Batch[166/227], Train loss :1.656, Train acc: 0.6312328767123287
[2024-12-24 18:20:23] - INFO: Epoch: 6, Batch[167/227], Train loss :1.573, Train acc: 0.6607142857142857
[2024-12-24 18:20:26] - INFO: Epoch: 6, Batch[168/227], Train loss :1.770, Train acc: 0.6090808416389811
[2024-12-24 18:20:28] - INFO: Epoch: 6, Batch[169/227], Train loss :1.643, Train acc: 0.6317747077577046
[2024-12-24 18:20:30] - INFO: Epoch: 6, Batch[170/227], Train loss :1.630, Train acc: 0.651073197578426
[2024-12-24 18:20:33] - INFO: Epoch: 6, Batch[171/227], Train loss :1.607, Train acc: 0.6408209806157354
[2024-12-24 18:20:34] - INFO: Epoch: 6, Batch[172/227], Train loss :1.641, Train acc: 0.639608520437536
[2024-12-24 18:20:37] - INFO: Epoch: 6, Batch[173/227], Train loss :1.607, Train acc: 0.6417171161254815
[2024-12-24 18:20:39] - INFO: Epoch: 6, Batch[174/227], Train loss :1.658, Train acc: 0.6315211422295443
[2024-12-24 18:20:42] - INFO: Epoch: 6, Batch[175/227], Train loss :1.685, Train acc: 0.6352413019079686
[2024-12-24 18:20:44] - INFO: Epoch: 6, Batch[176/227], Train loss :1.654, Train acc: 0.6474576271186441
[2024-12-24 18:20:46] - INFO: Epoch: 6, Batch[177/227], Train loss :1.665, Train acc: 0.6488636363636363
[2024-12-24 18:20:48] - INFO: Epoch: 6, Batch[178/227], Train loss :1.727, Train acc: 0.6229327453142227
[2024-12-24 18:20:51] - INFO: Epoch: 6, Batch[179/227], Train loss :1.596, Train acc: 0.6360505166475315
[2024-12-24 18:20:53] - INFO: Epoch: 6, Batch[180/227], Train loss :1.740, Train acc: 0.630242825607064
[2024-12-24 18:20:56] - INFO: Epoch: 6, Batch[181/227], Train loss :1.718, Train acc: 0.619934282584885
[2024-12-24 18:20:58] - INFO: Epoch: 6, Batch[182/227], Train loss :1.732, Train acc: 0.616260162601626
[2024-12-24 18:21:00] - INFO: Epoch: 6, Batch[183/227], Train loss :1.627, Train acc: 0.6356502242152466
[2024-12-24 18:21:03] - INFO: Epoch: 6, Batch[184/227], Train loss :1.698, Train acc: 0.6298491379310345
[2024-12-24 18:21:06] - INFO: Epoch: 6, Batch[185/227], Train loss :1.758, Train acc: 0.6365140650854937
[2024-12-24 18:21:07] - INFO: Epoch: 6, Batch[186/227], Train loss :1.605, Train acc: 0.6463768115942029
[2024-12-24 18:21:09] - INFO: Epoch: 6, Batch[187/227], Train loss :1.666, Train acc: 0.6395939086294417
[2024-12-24 18:21:11] - INFO: Epoch: 6, Batch[188/227], Train loss :1.769, Train acc: 0.6135741652983032
[2024-12-24 18:21:14] - INFO: Epoch: 6, Batch[189/227], Train loss :1.651, Train acc: 0.6316964285714286
[2024-12-24 18:21:16] - INFO: Epoch: 6, Batch[190/227], Train loss :1.630, Train acc: 0.6484507042253521
[2024-12-24 18:21:18] - INFO: Epoch: 6, Batch[191/227], Train loss :1.558, Train acc: 0.6517165005537099
[2024-12-24 18:21:21] - INFO: Epoch: 6, Batch[192/227], Train loss :1.740, Train acc: 0.6229321163719338
[2024-12-24 18:21:23] - INFO: Epoch: 6, Batch[193/227], Train loss :1.699, Train acc: 0.617237687366167
[2024-12-24 18:21:25] - INFO: Epoch: 6, Batch[194/227], Train loss :1.720, Train acc: 0.6111411573823688
[2024-12-24 18:21:27] - INFO: Epoch: 6, Batch[195/227], Train loss :1.765, Train acc: 0.6206322795341098
[2024-12-24 18:21:29] - INFO: Epoch: 6, Batch[196/227], Train loss :1.687, Train acc: 0.6317919075144509
[2024-12-24 18:21:31] - INFO: Epoch: 6, Batch[197/227], Train loss :1.581, Train acc: 0.6483578708946772
[2024-12-24 18:21:33] - INFO: Epoch: 6, Batch[198/227], Train loss :1.723, Train acc: 0.6182224706539966
[2024-12-24 18:21:35] - INFO: Epoch: 6, Batch[199/227], Train loss :1.745, Train acc: 0.6123814835471277
[2024-12-24 18:21:37] - INFO: Epoch: 6, Batch[200/227], Train loss :1.695, Train acc: 0.6235102925243771
[2024-12-24 18:21:40] - INFO: Epoch: 6, Batch[201/227], Train loss :1.707, Train acc: 0.6418152350081038
[2024-12-24 18:21:43] - INFO: Epoch: 6, Batch[202/227], Train loss :1.664, Train acc: 0.6227578475336323
[2024-12-24 18:21:45] - INFO: Epoch: 6, Batch[203/227], Train loss :1.683, Train acc: 0.6237413884472708
[2024-12-24 18:21:47] - INFO: Epoch: 6, Batch[204/227], Train loss :1.575, Train acc: 0.6535433070866141
[2024-12-24 18:21:49] - INFO: Epoch: 6, Batch[205/227], Train loss :1.716, Train acc: 0.6208425720620843
[2024-12-24 18:21:52] - INFO: Epoch: 6, Batch[206/227], Train loss :1.728, Train acc: 0.6281489594742606
[2024-12-24 18:21:54] - INFO: Epoch: 6, Batch[207/227], Train loss :1.606, Train acc: 0.6409544950055494
[2024-12-24 18:21:56] - INFO: Epoch: 6, Batch[208/227], Train loss :1.693, Train acc: 0.624332977588047
[2024-12-24 18:21:58] - INFO: Epoch: 6, Batch[209/227], Train loss :1.731, Train acc: 0.6243683323975294
[2024-12-24 18:22:01] - INFO: Epoch: 6, Batch[210/227], Train loss :1.723, Train acc: 0.633695652173913
[2024-12-24 18:22:03] - INFO: Epoch: 6, Batch[211/227], Train loss :1.542, Train acc: 0.6494486360998258
[2024-12-24 18:22:05] - INFO: Epoch: 6, Batch[212/227], Train loss :1.575, Train acc: 0.6424032351242056
[2024-12-24 18:22:07] - INFO: Epoch: 6, Batch[213/227], Train loss :1.670, Train acc: 0.6320960698689956
[2024-12-24 18:22:10] - INFO: Epoch: 6, Batch[214/227], Train loss :1.785, Train acc: 0.6106623586429726
[2024-12-24 18:22:12] - INFO: Epoch: 6, Batch[215/227], Train loss :1.699, Train acc: 0.6315217391304347
[2024-12-24 18:22:14] - INFO: Epoch: 6, Batch[216/227], Train loss :1.575, Train acc: 0.640625
[2024-12-24 18:22:17] - INFO: Epoch: 6, Batch[217/227], Train loss :1.642, Train acc: 0.6442151004888648
[2024-12-24 18:22:19] - INFO: Epoch: 6, Batch[218/227], Train loss :1.618, Train acc: 0.633011911514464
[2024-12-24 18:22:21] - INFO: Epoch: 6, Batch[219/227], Train loss :1.739, Train acc: 0.6307692307692307
[2024-12-24 18:22:23] - INFO: Epoch: 6, Batch[220/227], Train loss :1.698, Train acc: 0.6379310344827587
[2024-12-24 18:22:25] - INFO: Epoch: 6, Batch[221/227], Train loss :1.734, Train acc: 0.6266025641025641
[2024-12-24 18:22:27] - INFO: Epoch: 6, Batch[222/227], Train loss :1.716, Train acc: 0.6209406494960806
[2024-12-24 18:22:29] - INFO: Epoch: 6, Batch[223/227], Train loss :1.693, Train acc: 0.6336254107338445
[2024-12-24 18:22:32] - INFO: Epoch: 6, Batch[224/227], Train loss :1.702, Train acc: 0.6359060402684564
[2024-12-24 18:22:34] - INFO: Epoch: 6, Batch[225/227], Train loss :1.647, Train acc: 0.6409544950055494
[2024-12-24 18:22:35] - INFO: Epoch: 6, Batch[226/227], Train loss :1.637, Train acc: 0.6225296442687747
[2024-12-24 18:22:35] - INFO: Epoch: 6, Train loss: 1.639, Epoch time = 510.829s
[2024-12-24 18:22:37] - INFO: Epoch: 7, Batch[0/227], Train loss :1.583, Train acc: 0.6423666138404649
[2024-12-24 18:22:40] - INFO: Epoch: 7, Batch[1/227], Train loss :1.456, Train acc: 0.6668539325842696
[2024-12-24 18:22:42] - INFO: Epoch: 7, Batch[2/227], Train loss :1.460, Train acc: 0.6716335540838853
[2024-12-24 18:22:44] - INFO: Epoch: 7, Batch[3/227], Train loss :1.501, Train acc: 0.6420423682781097
[2024-12-24 18:22:45] - INFO: Epoch: 7, Batch[4/227], Train loss :1.534, Train acc: 0.6514806378132119
[2024-12-24 18:22:47] - INFO: Epoch: 7, Batch[5/227], Train loss :1.466, Train acc: 0.6583021890016018
[2024-12-24 18:22:49] - INFO: Epoch: 7, Batch[6/227], Train loss :1.474, Train acc: 0.6550912301353737
[2024-12-24 18:22:52] - INFO: Epoch: 7, Batch[7/227], Train loss :1.540, Train acc: 0.6537785588752196
[2024-12-24 18:22:54] - INFO: Epoch: 7, Batch[8/227], Train loss :1.590, Train acc: 0.6334080717488789
[2024-12-24 18:22:56] - INFO: Epoch: 7, Batch[9/227], Train loss :1.570, Train acc: 0.6439185470555862
[2024-12-24 18:22:58] - INFO: Epoch: 7, Batch[10/227], Train loss :1.529, Train acc: 0.6547285954113039
[2024-12-24 18:23:00] - INFO: Epoch: 7, Batch[11/227], Train loss :1.528, Train acc: 0.6655755591925805
[2024-12-24 18:23:03] - INFO: Epoch: 7, Batch[12/227], Train loss :1.421, Train acc: 0.6778336125069794
[2024-12-24 18:23:05] - INFO: Epoch: 7, Batch[13/227], Train loss :1.471, Train acc: 0.6712253829321663
[2024-12-24 18:23:06] - INFO: Epoch: 7, Batch[14/227], Train loss :1.565, Train acc: 0.6418578623391158
[2024-12-24 18:23:09] - INFO: Epoch: 7, Batch[15/227], Train loss :1.466, Train acc: 0.6641137855579868
[2024-12-24 18:23:12] - INFO: Epoch: 7, Batch[16/227], Train loss :1.525, Train acc: 0.6414581066376496
[2024-12-24 18:23:14] - INFO: Epoch: 7, Batch[17/227], Train loss :1.488, Train acc: 0.6633333333333333
[2024-12-24 18:23:16] - INFO: Epoch: 7, Batch[18/227], Train loss :1.501, Train acc: 0.6670378619153675
[2024-12-24 18:23:19] - INFO: Epoch: 7, Batch[19/227], Train loss :1.547, Train acc: 0.6518105849582173
[2024-12-24 18:23:21] - INFO: Epoch: 7, Batch[20/227], Train loss :1.615, Train acc: 0.6216361679224973
[2024-12-24 18:23:23] - INFO: Epoch: 7, Batch[21/227], Train loss :1.455, Train acc: 0.6590016825574874
[2024-12-24 18:23:25] - INFO: Epoch: 7, Batch[22/227], Train loss :1.376, Train acc: 0.6777332570120206
[2024-12-24 18:23:27] - INFO: Epoch: 7, Batch[23/227], Train loss :1.550, Train acc: 0.6527545909849749
[2024-12-24 18:23:29] - INFO: Epoch: 7, Batch[24/227], Train loss :1.537, Train acc: 0.6445164775796867
[2024-12-24 18:23:31] - INFO: Epoch: 7, Batch[25/227], Train loss :1.480, Train acc: 0.6635254988913526
[2024-12-24 18:23:34] - INFO: Epoch: 7, Batch[26/227], Train loss :1.581, Train acc: 0.6553262688232013
[2024-12-24 18:23:36] - INFO: Epoch: 7, Batch[27/227], Train loss :1.498, Train acc: 0.6514065085493657
[2024-12-24 18:23:38] - INFO: Epoch: 7, Batch[28/227], Train loss :1.399, Train acc: 0.6702795208214489
[2024-12-24 18:23:40] - INFO: Epoch: 7, Batch[29/227], Train loss :1.527, Train acc: 0.642504118616145
[2024-12-24 18:23:42] - INFO: Epoch: 7, Batch[30/227], Train loss :1.502, Train acc: 0.658288770053476
[2024-12-24 18:23:44] - INFO: Epoch: 7, Batch[31/227], Train loss :1.421, Train acc: 0.6723842195540308
[2024-12-24 18:23:46] - INFO: Epoch: 7, Batch[32/227], Train loss :1.555, Train acc: 0.6631284916201118
[2024-12-24 18:23:48] - INFO: Epoch: 7, Batch[33/227], Train loss :1.406, Train acc: 0.6672008547008547
[2024-12-24 18:23:51] - INFO: Epoch: 7, Batch[34/227], Train loss :1.498, Train acc: 0.660149511213341
[2024-12-24 18:23:53] - INFO: Epoch: 7, Batch[35/227], Train loss :1.407, Train acc: 0.681044267877412
[2024-12-24 18:23:55] - INFO: Epoch: 7, Batch[36/227], Train loss :1.507, Train acc: 0.6553191489361702
[2024-12-24 18:23:58] - INFO: Epoch: 7, Batch[37/227], Train loss :1.647, Train acc: 0.6455696202531646
[2024-12-24 18:24:00] - INFO: Epoch: 7, Batch[38/227], Train loss :1.510, Train acc: 0.6623081296191018
[2024-12-24 18:24:03] - INFO: Epoch: 7, Batch[39/227], Train loss :1.522, Train acc: 0.6479956068094453
[2024-12-24 18:24:05] - INFO: Epoch: 7, Batch[40/227], Train loss :1.477, Train acc: 0.6685979142526072
[2024-12-24 18:24:07] - INFO: Epoch: 7, Batch[41/227], Train loss :1.565, Train acc: 0.6481681034482759
[2024-12-24 18:24:09] - INFO: Epoch: 7, Batch[42/227], Train loss :1.447, Train acc: 0.6803601575689364
[2024-12-24 18:24:11] - INFO: Epoch: 7, Batch[43/227], Train loss :1.510, Train acc: 0.6553640911617565
[2024-12-24 18:24:13] - INFO: Epoch: 7, Batch[44/227], Train loss :1.517, Train acc: 0.6589912280701754
[2024-12-24 18:24:15] - INFO: Epoch: 7, Batch[45/227], Train loss :1.530, Train acc: 0.6358839050131926
[2024-12-24 18:24:17] - INFO: Epoch: 7, Batch[46/227], Train loss :1.622, Train acc: 0.6364617044228694
[2024-12-24 18:24:19] - INFO: Epoch: 7, Batch[47/227], Train loss :1.515, Train acc: 0.6730987514188422
[2024-12-24 18:24:22] - INFO: Epoch: 7, Batch[48/227], Train loss :1.519, Train acc: 0.6617724174095437
[2024-12-24 18:24:24] - INFO: Epoch: 7, Batch[49/227], Train loss :1.516, Train acc: 0.6689936009307738
[2024-12-24 18:24:26] - INFO: Epoch: 7, Batch[50/227], Train loss :1.606, Train acc: 0.6265193370165746
[2024-12-24 18:24:28] - INFO: Epoch: 7, Batch[51/227], Train loss :1.404, Train acc: 0.6714697406340058
[2024-12-24 18:24:30] - INFO: Epoch: 7, Batch[52/227], Train loss :1.586, Train acc: 0.6361690743713215
[2024-12-24 18:24:33] - INFO: Epoch: 7, Batch[53/227], Train loss :1.517, Train acc: 0.648854961832061
[2024-12-24 18:24:35] - INFO: Epoch: 7, Batch[54/227], Train loss :1.636, Train acc: 0.6342825237297599
[2024-12-24 18:24:38] - INFO: Epoch: 7, Batch[55/227], Train loss :1.524, Train acc: 0.6562841530054645
[2024-12-24 18:24:41] - INFO: Epoch: 7, Batch[56/227], Train loss :1.435, Train acc: 0.6720091585575272
[2024-12-24 18:24:43] - INFO: Epoch: 7, Batch[57/227], Train loss :1.557, Train acc: 0.6359550561797753
[2024-12-24 18:24:44] - INFO: Epoch: 7, Batch[58/227], Train loss :1.438, Train acc: 0.6670500287521565
[2024-12-24 18:24:46] - INFO: Epoch: 7, Batch[59/227], Train loss :1.528, Train acc: 0.651697699890471
[2024-12-24 18:24:48] - INFO: Epoch: 7, Batch[60/227], Train loss :1.462, Train acc: 0.6664769493454752
[2024-12-24 18:24:51] - INFO: Epoch: 7, Batch[61/227], Train loss :1.629, Train acc: 0.6470588235294118
[2024-12-24 18:24:53] - INFO: Epoch: 7, Batch[62/227], Train loss :1.548, Train acc: 0.6526027397260274
[2024-12-24 18:24:56] - INFO: Epoch: 7, Batch[63/227], Train loss :1.549, Train acc: 0.6462472406181016
[2024-12-24 18:24:58] - INFO: Epoch: 7, Batch[64/227], Train loss :1.561, Train acc: 0.6528239202657807
[2024-12-24 18:25:00] - INFO: Epoch: 7, Batch[65/227], Train loss :1.558, Train acc: 0.6338888888888888
[2024-12-24 18:25:02] - INFO: Epoch: 7, Batch[66/227], Train loss :1.483, Train acc: 0.657762938230384
[2024-12-24 18:25:04] - INFO: Epoch: 7, Batch[67/227], Train loss :1.470, Train acc: 0.6797461050201962
[2024-12-24 18:25:06] - INFO: Epoch: 7, Batch[68/227], Train loss :1.471, Train acc: 0.6541057367829022
[2024-12-24 18:25:09] - INFO: Epoch: 7, Batch[69/227], Train loss :1.688, Train acc: 0.6098988823842469
[2024-12-24 18:25:11] - INFO: Epoch: 7, Batch[70/227], Train loss :1.560, Train acc: 0.6523394994559304
[2024-12-24 18:25:14] - INFO: Epoch: 7, Batch[71/227], Train loss :1.680, Train acc: 0.6231050705697857
[2024-12-24 18:25:15] - INFO: Epoch: 7, Batch[72/227], Train loss :1.525, Train acc: 0.6674325100516945
[2024-12-24 18:25:18] - INFO: Epoch: 7, Batch[73/227], Train loss :1.581, Train acc: 0.6348314606741573
[2024-12-24 18:25:20] - INFO: Epoch: 7, Batch[74/227], Train loss :1.496, Train acc: 0.6536585365853659
[2024-12-24 18:25:22] - INFO: Epoch: 7, Batch[75/227], Train loss :1.634, Train acc: 0.6444568868980963
[2024-12-24 18:25:24] - INFO: Epoch: 7, Batch[76/227], Train loss :1.507, Train acc: 0.6657142857142857
[2024-12-24 18:25:26] - INFO: Epoch: 7, Batch[77/227], Train loss :1.548, Train acc: 0.6484679665738161
[2024-12-24 18:25:28] - INFO: Epoch: 7, Batch[78/227], Train loss :1.417, Train acc: 0.6662735849056604
[2024-12-24 18:25:30] - INFO: Epoch: 7, Batch[79/227], Train loss :1.566, Train acc: 0.635048231511254
[2024-12-24 18:25:32] - INFO: Epoch: 7, Batch[80/227], Train loss :1.495, Train acc: 0.6619237336368811
[2024-12-24 18:25:34] - INFO: Epoch: 7, Batch[81/227], Train loss :1.528, Train acc: 0.6629023150762281
[2024-12-24 18:25:38] - INFO: Epoch: 7, Batch[82/227], Train loss :1.584, Train acc: 0.643329658213892
[2024-12-24 18:25:40] - INFO: Epoch: 7, Batch[83/227], Train loss :1.600, Train acc: 0.6350440642820114
[2024-12-24 18:25:41] - INFO: Epoch: 7, Batch[84/227], Train loss :1.503, Train acc: 0.6549413735343383
[2024-12-24 18:25:44] - INFO: Epoch: 7, Batch[85/227], Train loss :1.574, Train acc: 0.6402439024390244
[2024-12-24 18:25:46] - INFO: Epoch: 7, Batch[86/227], Train loss :1.490, Train acc: 0.6644808743169399
[2024-12-24 18:25:48] - INFO: Epoch: 7, Batch[87/227], Train loss :1.644, Train acc: 0.6228315612758814
[2024-12-24 18:25:51] - INFO: Epoch: 7, Batch[88/227], Train loss :1.564, Train acc: 0.6329331046312179
[2024-12-24 18:25:53] - INFO: Epoch: 7, Batch[89/227], Train loss :1.551, Train acc: 0.6520325203252032
[2024-12-24 18:25:55] - INFO: Epoch: 7, Batch[90/227], Train loss :1.566, Train acc: 0.6371774618220116
[2024-12-24 18:25:57] - INFO: Epoch: 7, Batch[91/227], Train loss :1.531, Train acc: 0.651456844420011
[2024-12-24 18:26:00] - INFO: Epoch: 7, Batch[92/227], Train loss :1.670, Train acc: 0.6235616438356164
[2024-12-24 18:26:02] - INFO: Epoch: 7, Batch[93/227], Train loss :1.550, Train acc: 0.6507580011229647
[2024-12-24 18:26:04] - INFO: Epoch: 7, Batch[94/227], Train loss :1.549, Train acc: 0.6446844798180784
[2024-12-24 18:26:06] - INFO: Epoch: 7, Batch[95/227], Train loss :1.541, Train acc: 0.6447811447811448
[2024-12-24 18:26:09] - INFO: Epoch: 7, Batch[96/227], Train loss :1.533, Train acc: 0.6505832449628844
[2024-12-24 18:26:11] - INFO: Epoch: 7, Batch[97/227], Train loss :1.444, Train acc: 0.6643090315560392
[2024-12-24 18:26:14] - INFO: Epoch: 7, Batch[98/227], Train loss :1.586, Train acc: 0.6306111411573824
[2024-12-24 18:26:16] - INFO: Epoch: 7, Batch[99/227], Train loss :1.405, Train acc: 0.6721967963386728
[2024-12-24 18:26:17] - INFO: Epoch: 7, Batch[100/227], Train loss :1.582, Train acc: 0.6273224043715847
[2024-12-24 18:26:21] - INFO: Epoch: 7, Batch[101/227], Train loss :1.658, Train acc: 0.6232508073196986
[2024-12-24 18:26:22] - INFO: Epoch: 7, Batch[102/227], Train loss :1.576, Train acc: 0.6536920435031482
[2024-12-24 18:26:25] - INFO: Epoch: 7, Batch[103/227], Train loss :1.551, Train acc: 0.6494731003882418
[2024-12-24 18:26:27] - INFO: Epoch: 7, Batch[104/227], Train loss :1.568, Train acc: 0.6471226927252985
[2024-12-24 18:26:30] - INFO: Epoch: 7, Batch[105/227], Train loss :1.528, Train acc: 0.6558370044052864
[2024-12-24 18:26:33] - INFO: Epoch: 7, Batch[106/227], Train loss :1.543, Train acc: 0.6576879910213244
[2024-12-24 18:26:35] - INFO: Epoch: 7, Batch[107/227], Train loss :1.511, Train acc: 0.6473565804274466
[2024-12-24 18:26:37] - INFO: Epoch: 7, Batch[108/227], Train loss :1.478, Train acc: 0.6600768808347062
[2024-12-24 18:26:40] - INFO: Epoch: 7, Batch[109/227], Train loss :1.488, Train acc: 0.6517906336088154
[2024-12-24 18:26:42] - INFO: Epoch: 7, Batch[110/227], Train loss :1.608, Train acc: 0.6413456321215409
[2024-12-24 18:26:44] - INFO: Epoch: 7, Batch[111/227], Train loss :1.634, Train acc: 0.6387766878245816
[2024-12-24 18:26:46] - INFO: Epoch: 7, Batch[112/227], Train loss :1.595, Train acc: 0.6502575844304522
[2024-12-24 18:26:48] - INFO: Epoch: 7, Batch[113/227], Train loss :1.654, Train acc: 0.6382271468144044
[2024-12-24 18:26:52] - INFO: Epoch: 7, Batch[114/227], Train loss :1.622, Train acc: 0.625
[2024-12-24 18:26:54] - INFO: Epoch: 7, Batch[115/227], Train loss :1.609, Train acc: 0.640461792193513
[2024-12-24 18:26:56] - INFO: Epoch: 7, Batch[116/227], Train loss :1.649, Train acc: 0.6192584394023243
[2024-12-24 18:26:58] - INFO: Epoch: 7, Batch[117/227], Train loss :1.685, Train acc: 0.6276653909240022
[2024-12-24 18:27:00] - INFO: Epoch: 7, Batch[118/227], Train loss :1.573, Train acc: 0.6426193118756937
[2024-12-24 18:27:02] - INFO: Epoch: 7, Batch[119/227], Train loss :1.609, Train acc: 0.6307519640852974
[2024-12-24 18:27:04] - INFO: Epoch: 7, Batch[120/227], Train loss :1.549, Train acc: 0.6470264317180616
[2024-12-24 18:27:06] - INFO: Epoch: 7, Batch[121/227], Train loss :1.558, Train acc: 0.6361111111111111
[2024-12-24 18:27:08] - INFO: Epoch: 7, Batch[122/227], Train loss :1.604, Train acc: 0.6419821826280624
[2024-12-24 18:27:11] - INFO: Epoch: 7, Batch[123/227], Train loss :1.565, Train acc: 0.6353651839648545
[2024-12-24 18:27:13] - INFO: Epoch: 7, Batch[124/227], Train loss :1.586, Train acc: 0.6391139924365208
[2024-12-24 18:27:16] - INFO: Epoch: 7, Batch[125/227], Train loss :1.592, Train acc: 0.645752359800111
[2024-12-24 18:27:18] - INFO: Epoch: 7, Batch[126/227], Train loss :1.571, Train acc: 0.6545253863134658
[2024-12-24 18:27:21] - INFO: Epoch: 7, Batch[127/227], Train loss :1.614, Train acc: 0.6388109927089175
[2024-12-24 18:27:23] - INFO: Epoch: 7, Batch[128/227], Train loss :1.536, Train acc: 0.6567164179104478
[2024-12-24 18:27:25] - INFO: Epoch: 7, Batch[129/227], Train loss :1.581, Train acc: 0.6534435261707989
[2024-12-24 18:27:27] - INFO: Epoch: 7, Batch[130/227], Train loss :1.543, Train acc: 0.6427369601794728
[2024-12-24 18:27:29] - INFO: Epoch: 7, Batch[131/227], Train loss :1.590, Train acc: 0.6464702612562535
[2024-12-24 18:27:31] - INFO: Epoch: 7, Batch[132/227], Train loss :1.460, Train acc: 0.6570783981951495
[2024-12-24 18:27:33] - INFO: Epoch: 7, Batch[133/227], Train loss :1.580, Train acc: 0.6511497476163769
[2024-12-24 18:27:36] - INFO: Epoch: 7, Batch[134/227], Train loss :1.626, Train acc: 0.6319290465631929
[2024-12-24 18:27:38] - INFO: Epoch: 7, Batch[135/227], Train loss :1.460, Train acc: 0.6638273708120386
[2024-12-24 18:27:40] - INFO: Epoch: 7, Batch[136/227], Train loss :1.677, Train acc: 0.6261261261261262
[2024-12-24 18:27:42] - INFO: Epoch: 7, Batch[137/227], Train loss :1.638, Train acc: 0.6264462809917355
[2024-12-24 18:27:44] - INFO: Epoch: 7, Batch[138/227], Train loss :1.658, Train acc: 0.6134636264929425
[2024-12-24 18:27:46] - INFO: Epoch: 7, Batch[139/227], Train loss :1.525, Train acc: 0.6584958217270195
[2024-12-24 18:27:48] - INFO: Epoch: 7, Batch[140/227], Train loss :1.547, Train acc: 0.649402390438247
[2024-12-24 18:27:51] - INFO: Epoch: 7, Batch[141/227], Train loss :1.708, Train acc: 0.623608017817372
[2024-12-24 18:27:53] - INFO: Epoch: 7, Batch[142/227], Train loss :1.590, Train acc: 0.6194690265486725
[2024-12-24 18:27:55] - INFO: Epoch: 7, Batch[143/227], Train loss :1.622, Train acc: 0.6398446170921198
[2024-12-24 18:27:57] - INFO: Epoch: 7, Batch[144/227], Train loss :1.694, Train acc: 0.6195590729225551
[2024-12-24 18:28:00] - INFO: Epoch: 7, Batch[145/227], Train loss :1.660, Train acc: 0.6384489350081922
[2024-12-24 18:28:01] - INFO: Epoch: 7, Batch[146/227], Train loss :1.652, Train acc: 0.634132738427217
[2024-12-24 18:28:03] - INFO: Epoch: 7, Batch[147/227], Train loss :1.565, Train acc: 0.655328798185941
[2024-12-24 18:28:06] - INFO: Epoch: 7, Batch[148/227], Train loss :1.642, Train acc: 0.6338477913783928
[2024-12-24 18:28:08] - INFO: Epoch: 7, Batch[149/227], Train loss :1.560, Train acc: 0.6401345291479821
[2024-12-24 18:28:10] - INFO: Epoch: 7, Batch[150/227], Train loss :1.630, Train acc: 0.6489542114188808
[2024-12-24 18:28:13] - INFO: Epoch: 7, Batch[151/227], Train loss :1.614, Train acc: 0.6280623608017817
[2024-12-24 18:28:15] - INFO: Epoch: 7, Batch[152/227], Train loss :1.648, Train acc: 0.6236786469344608
[2024-12-24 18:28:17] - INFO: Epoch: 7, Batch[153/227], Train loss :1.569, Train acc: 0.6400220507166483
[2024-12-24 18:28:19] - INFO: Epoch: 7, Batch[154/227], Train loss :1.545, Train acc: 0.6543408360128617
[2024-12-24 18:28:21] - INFO: Epoch: 7, Batch[155/227], Train loss :1.679, Train acc: 0.6318380743982495
[2024-12-24 18:28:23] - INFO: Epoch: 7, Batch[156/227], Train loss :1.609, Train acc: 0.6461020751542345
[2024-12-24 18:28:25] - INFO: Epoch: 7, Batch[157/227], Train loss :1.521, Train acc: 0.6708004509582863
[2024-12-24 18:28:27] - INFO: Epoch: 7, Batch[158/227], Train loss :1.686, Train acc: 0.6329588014981273
[2024-12-24 18:28:30] - INFO: Epoch: 7, Batch[159/227], Train loss :1.722, Train acc: 0.6058242329693188
[2024-12-24 18:28:31] - INFO: Epoch: 7, Batch[160/227], Train loss :1.691, Train acc: 0.6256983240223464
[2024-12-24 18:28:34] - INFO: Epoch: 7, Batch[161/227], Train loss :1.575, Train acc: 0.642656162070906
[2024-12-24 18:28:36] - INFO: Epoch: 7, Batch[162/227], Train loss :1.614, Train acc: 0.6316082359488036
[2024-12-24 18:28:39] - INFO: Epoch: 7, Batch[163/227], Train loss :1.658, Train acc: 0.6221272047033671
[2024-12-24 18:28:41] - INFO: Epoch: 7, Batch[164/227], Train loss :1.636, Train acc: 0.6327212020033389
[2024-12-24 18:28:43] - INFO: Epoch: 7, Batch[165/227], Train loss :1.579, Train acc: 0.6366576819407008
[2024-12-24 18:28:46] - INFO: Epoch: 7, Batch[166/227], Train loss :1.626, Train acc: 0.6459770114942529
[2024-12-24 18:28:48] - INFO: Epoch: 7, Batch[167/227], Train loss :1.632, Train acc: 0.6379310344827587
[2024-12-24 18:28:51] - INFO: Epoch: 7, Batch[168/227], Train loss :1.686, Train acc: 0.6229065370070233
[2024-12-24 18:28:53] - INFO: Epoch: 7, Batch[169/227], Train loss :1.546, Train acc: 0.6376021798365122
[2024-12-24 18:28:55] - INFO: Epoch: 7, Batch[170/227], Train loss :1.580, Train acc: 0.6259541984732825
[2024-12-24 18:28:57] - INFO: Epoch: 7, Batch[171/227], Train loss :1.513, Train acc: 0.6551525618883132
[2024-12-24 18:28:59] - INFO: Epoch: 7, Batch[172/227], Train loss :1.690, Train acc: 0.6236436322101656
[2024-12-24 18:29:01] - INFO: Epoch: 7, Batch[173/227], Train loss :1.696, Train acc: 0.632896983494593
[2024-12-24 18:29:03] - INFO: Epoch: 7, Batch[174/227], Train loss :1.519, Train acc: 0.641954969796815
[2024-12-24 18:29:05] - INFO: Epoch: 7, Batch[175/227], Train loss :1.624, Train acc: 0.6246529705719045
[2024-12-24 18:29:08] - INFO: Epoch: 7, Batch[176/227], Train loss :1.626, Train acc: 0.630879911455451
[2024-12-24 18:29:10] - INFO: Epoch: 7, Batch[177/227], Train loss :1.700, Train acc: 0.6267029972752044
[2024-12-24 18:29:12] - INFO: Epoch: 7, Batch[178/227], Train loss :1.607, Train acc: 0.6487376509330406
[2024-12-24 18:29:14] - INFO: Epoch: 7, Batch[179/227], Train loss :1.516, Train acc: 0.6507413509060955
[2024-12-24 18:29:16] - INFO: Epoch: 7, Batch[180/227], Train loss :1.630, Train acc: 0.642976124375347
[2024-12-24 18:29:19] - INFO: Epoch: 7, Batch[181/227], Train loss :1.636, Train acc: 0.6452173913043479
[2024-12-24 18:29:23] - INFO: Epoch: 7, Batch[182/227], Train loss :1.668, Train acc: 0.6327995582551077
[2024-12-24 18:29:26] - INFO: Epoch: 7, Batch[183/227], Train loss :1.674, Train acc: 0.6275899672846238
[2024-12-24 18:29:28] - INFO: Epoch: 7, Batch[184/227], Train loss :1.529, Train acc: 0.6583285957930642
[2024-12-24 18:29:30] - INFO: Epoch: 7, Batch[185/227], Train loss :1.599, Train acc: 0.6450905101481075
[2024-12-24 18:29:33] - INFO: Epoch: 7, Batch[186/227], Train loss :1.668, Train acc: 0.6275899672846238
[2024-12-24 18:29:34] - INFO: Epoch: 7, Batch[187/227], Train loss :1.687, Train acc: 0.6314943760042849
[2024-12-24 18:29:37] - INFO: Epoch: 7, Batch[188/227], Train loss :1.701, Train acc: 0.6246482836240855
[2024-12-24 18:29:39] - INFO: Epoch: 7, Batch[189/227], Train loss :1.593, Train acc: 0.6340519624101714
[2024-12-24 18:29:41] - INFO: Epoch: 7, Batch[190/227], Train loss :1.592, Train acc: 0.6372809346787042
[2024-12-24 18:29:43] - INFO: Epoch: 7, Batch[191/227], Train loss :1.531, Train acc: 0.6470908102229472
[2024-12-24 18:29:45] - INFO: Epoch: 7, Batch[192/227], Train loss :1.621, Train acc: 0.6283971159179146
[2024-12-24 18:29:48] - INFO: Epoch: 7, Batch[193/227], Train loss :1.620, Train acc: 0.6254763200870985
[2024-12-24 18:29:51] - INFO: Epoch: 7, Batch[194/227], Train loss :1.532, Train acc: 0.6390598768886402
[2024-12-24 18:29:53] - INFO: Epoch: 7, Batch[195/227], Train loss :1.684, Train acc: 0.6226203807390818
[2024-12-24 18:29:56] - INFO: Epoch: 7, Batch[196/227], Train loss :1.640, Train acc: 0.6417112299465241
[2024-12-24 18:29:58] - INFO: Epoch: 7, Batch[197/227], Train loss :1.558, Train acc: 0.6385209713024282
[2024-12-24 18:30:00] - INFO: Epoch: 7, Batch[198/227], Train loss :1.529, Train acc: 0.6500847936687394
[2024-12-24 18:30:02] - INFO: Epoch: 7, Batch[199/227], Train loss :1.612, Train acc: 0.6388434260774686
[2024-12-24 18:30:04] - INFO: Epoch: 7, Batch[200/227], Train loss :1.627, Train acc: 0.6362643364281814
[2024-12-24 18:30:06] - INFO: Epoch: 7, Batch[201/227], Train loss :1.671, Train acc: 0.6249313563975838
[2024-12-24 18:30:08] - INFO: Epoch: 7, Batch[202/227], Train loss :1.677, Train acc: 0.625
[2024-12-24 18:30:10] - INFO: Epoch: 7, Batch[203/227], Train loss :1.604, Train acc: 0.6325369738339022
[2024-12-24 18:30:12] - INFO: Epoch: 7, Batch[204/227], Train loss :1.735, Train acc: 0.6102340772999456
[2024-12-24 18:30:14] - INFO: Epoch: 7, Batch[205/227], Train loss :1.776, Train acc: 0.6090100111234705
[2024-12-24 18:30:17] - INFO: Epoch: 7, Batch[206/227], Train loss :1.641, Train acc: 0.644029428409734
[2024-12-24 18:30:19] - INFO: Epoch: 7, Batch[207/227], Train loss :1.580, Train acc: 0.6637931034482759
[2024-12-24 18:30:22] - INFO: Epoch: 7, Batch[208/227], Train loss :1.568, Train acc: 0.654485049833887
[2024-12-24 18:30:24] - INFO: Epoch: 7, Batch[209/227], Train loss :1.646, Train acc: 0.6286501377410468
[2024-12-24 18:30:27] - INFO: Epoch: 7, Batch[210/227], Train loss :1.752, Train acc: 0.6160667709963484
[2024-12-24 18:30:29] - INFO: Epoch: 7, Batch[211/227], Train loss :1.573, Train acc: 0.6442953020134228
[2024-12-24 18:30:31] - INFO: Epoch: 7, Batch[212/227], Train loss :1.720, Train acc: 0.6304812834224599
[2024-12-24 18:30:34] - INFO: Epoch: 7, Batch[213/227], Train loss :1.642, Train acc: 0.6378041878890776
[2024-12-24 18:30:36] - INFO: Epoch: 7, Batch[214/227], Train loss :1.640, Train acc: 0.6364130434782609
[2024-12-24 18:30:38] - INFO: Epoch: 7, Batch[215/227], Train loss :1.610, Train acc: 0.6443100604727873
[2024-12-24 18:30:40] - INFO: Epoch: 7, Batch[216/227], Train loss :1.550, Train acc: 0.6526374859708193
[2024-12-24 18:30:42] - INFO: Epoch: 7, Batch[217/227], Train loss :1.584, Train acc: 0.6476624857468644
[2024-12-24 18:30:44] - INFO: Epoch: 7, Batch[218/227], Train loss :1.615, Train acc: 0.6422067487948581
[2024-12-24 18:30:46] - INFO: Epoch: 7, Batch[219/227], Train loss :1.589, Train acc: 0.6438828259620908
[2024-12-24 18:30:49] - INFO: Epoch: 7, Batch[220/227], Train loss :1.677, Train acc: 0.6307692307692307
[2024-12-24 18:30:52] - INFO: Epoch: 7, Batch[221/227], Train loss :1.560, Train acc: 0.6467941507311586
[2024-12-24 18:30:54] - INFO: Epoch: 7, Batch[222/227], Train loss :1.764, Train acc: 0.6216216216216216
[2024-12-24 18:30:57] - INFO: Epoch: 7, Batch[223/227], Train loss :1.686, Train acc: 0.6433260393873085
[2024-12-24 18:30:59] - INFO: Epoch: 7, Batch[224/227], Train loss :1.648, Train acc: 0.6430205949656751
[2024-12-24 18:31:01] - INFO: Epoch: 7, Batch[225/227], Train loss :1.627, Train acc: 0.6327092511013216
[2024-12-24 18:31:02] - INFO: Epoch: 7, Batch[226/227], Train loss :1.585, Train acc: 0.6400797607178464
[2024-12-24 18:31:02] - INFO: Epoch: 7, Train loss: 1.573, Epoch time = 506.930s
[2024-12-24 18:31:07] - INFO: Accuracy on validation0.569
[2024-12-24 18:31:10] - INFO: Epoch: 8, Batch[0/227], Train loss :1.326, Train acc: 0.6716583471991125
[2024-12-24 18:31:12] - INFO: Epoch: 8, Batch[1/227], Train loss :1.499, Train acc: 0.6545157780195865
[2024-12-24 18:31:14] - INFO: Epoch: 8, Batch[2/227], Train loss :1.497, Train acc: 0.658669574700109
[2024-12-24 18:31:16] - INFO: Epoch: 8, Batch[3/227], Train loss :1.502, Train acc: 0.6373390557939914
[2024-12-24 18:31:18] - INFO: Epoch: 8, Batch[4/227], Train loss :1.350, Train acc: 0.6745205479452054
[2024-12-24 18:31:21] - INFO: Epoch: 8, Batch[5/227], Train loss :1.601, Train acc: 0.6362672322375398
[2024-12-24 18:31:23] - INFO: Epoch: 8, Batch[6/227], Train loss :1.432, Train acc: 0.6576576576576577
[2024-12-24 18:31:26] - INFO: Epoch: 8, Batch[7/227], Train loss :1.483, Train acc: 0.6593110871905274
[2024-12-24 18:31:28] - INFO: Epoch: 8, Batch[8/227], Train loss :1.333, Train acc: 0.6778523489932886
[2024-12-24 18:31:30] - INFO: Epoch: 8, Batch[9/227], Train loss :1.562, Train acc: 0.6395289298515104
[2024-12-24 18:31:33] - INFO: Epoch: 8, Batch[10/227], Train loss :1.384, Train acc: 0.6718146718146718
[2024-12-24 18:31:34] - INFO: Epoch: 8, Batch[11/227], Train loss :1.405, Train acc: 0.6647662485746865
[2024-12-24 18:31:36] - INFO: Epoch: 8, Batch[12/227], Train loss :1.344, Train acc: 0.6859456333140543
[2024-12-24 18:31:39] - INFO: Epoch: 8, Batch[13/227], Train loss :1.476, Train acc: 0.6616941928609483
[2024-12-24 18:31:41] - INFO: Epoch: 8, Batch[14/227], Train loss :1.402, Train acc: 0.6692477876106194
[2024-12-24 18:31:42] - INFO: Epoch: 8, Batch[15/227], Train loss :1.450, Train acc: 0.6494900697799249
[2024-12-24 18:31:44] - INFO: Epoch: 8, Batch[16/227], Train loss :1.438, Train acc: 0.665258711721225
[2024-12-24 18:31:46] - INFO: Epoch: 8, Batch[17/227], Train loss :1.387, Train acc: 0.6845372460496614
[2024-12-24 18:31:48] - INFO: Epoch: 8, Batch[18/227], Train loss :1.425, Train acc: 0.6645892351274788
[2024-12-24 18:31:51] - INFO: Epoch: 8, Batch[19/227], Train loss :1.446, Train acc: 0.6549375709421112
[2024-12-24 18:31:53] - INFO: Epoch: 8, Batch[20/227], Train loss :1.304, Train acc: 0.7073170731707317
[2024-12-24 18:31:55] - INFO: Epoch: 8, Batch[21/227], Train loss :1.620, Train acc: 0.6351495726495726
[2024-12-24 18:31:57] - INFO: Epoch: 8, Batch[22/227], Train loss :1.411, Train acc: 0.6604938271604939
[2024-12-24 18:32:00] - INFO: Epoch: 8, Batch[23/227], Train loss :1.458, Train acc: 0.6618122977346278
[2024-12-24 18:32:02] - INFO: Epoch: 8, Batch[24/227], Train loss :1.422, Train acc: 0.6639072847682119
[2024-12-24 18:32:05] - INFO: Epoch: 8, Batch[25/227], Train loss :1.494, Train acc: 0.6640226628895184
[2024-12-24 18:32:07] - INFO: Epoch: 8, Batch[26/227], Train loss :1.427, Train acc: 0.6674069961132704
[2024-12-24 18:32:10] - INFO: Epoch: 8, Batch[27/227], Train loss :1.506, Train acc: 0.6653846153846154
[2024-12-24 18:32:13] - INFO: Epoch: 8, Batch[28/227], Train loss :1.390, Train acc: 0.6863181312569522
[2024-12-24 18:32:15] - INFO: Epoch: 8, Batch[29/227], Train loss :1.494, Train acc: 0.6507936507936508
[2024-12-24 18:32:17] - INFO: Epoch: 8, Batch[30/227], Train loss :1.457, Train acc: 0.6483454851374089
[2024-12-24 18:32:20] - INFO: Epoch: 8, Batch[31/227], Train loss :1.528, Train acc: 0.6406080347448425
[2024-12-24 18:32:22] - INFO: Epoch: 8, Batch[32/227], Train loss :1.424, Train acc: 0.6651454649172847
[2024-12-24 18:32:24] - INFO: Epoch: 8, Batch[33/227], Train loss :1.347, Train acc: 0.6852367688022284
[2024-12-24 18:32:27] - INFO: Epoch: 8, Batch[34/227], Train loss :1.507, Train acc: 0.6336796063422635
[2024-12-24 18:32:29] - INFO: Epoch: 8, Batch[35/227], Train loss :1.390, Train acc: 0.6781411359724613
[2024-12-24 18:32:32] - INFO: Epoch: 8, Batch[36/227], Train loss :1.598, Train acc: 0.6261285183218269
[2024-12-24 18:32:34] - INFO: Epoch: 8, Batch[37/227], Train loss :1.459, Train acc: 0.663013698630137
[2024-12-24 18:32:37] - INFO: Epoch: 8, Batch[38/227], Train loss :1.518, Train acc: 0.6439678284182305
[2024-12-24 18:32:39] - INFO: Epoch: 8, Batch[39/227], Train loss :1.463, Train acc: 0.6568848758465011
[2024-12-24 18:32:41] - INFO: Epoch: 8, Batch[40/227], Train loss :1.544, Train acc: 0.6606260296540363
[2024-12-24 18:32:44] - INFO: Epoch: 8, Batch[41/227], Train loss :1.508, Train acc: 0.6433260393873085
[2024-12-24 18:32:46] - INFO: Epoch: 8, Batch[42/227], Train loss :1.494, Train acc: 0.6537805571347356
[2024-12-24 18:32:48] - INFO: Epoch: 8, Batch[43/227], Train loss :1.418, Train acc: 0.6662876634451392
[2024-12-24 18:32:52] - INFO: Epoch: 8, Batch[44/227], Train loss :1.541, Train acc: 0.650319829424307
[2024-12-24 18:32:54] - INFO: Epoch: 8, Batch[45/227], Train loss :1.494, Train acc: 0.6525974025974026
[2024-12-24 18:32:56] - INFO: Epoch: 8, Batch[46/227], Train loss :1.520, Train acc: 0.6302835741037989
[2024-12-24 18:32:58] - INFO: Epoch: 8, Batch[47/227], Train loss :1.446, Train acc: 0.6626984126984127
[2024-12-24 18:33:00] - INFO: Epoch: 8, Batch[48/227], Train loss :1.437, Train acc: 0.6717861205915814
[2024-12-24 18:33:02] - INFO: Epoch: 8, Batch[49/227], Train loss :1.348, Train acc: 0.6832298136645962
[2024-12-24 18:33:05] - INFO: Epoch: 8, Batch[50/227], Train loss :1.474, Train acc: 0.6560088202866593
[2024-12-24 18:33:07] - INFO: Epoch: 8, Batch[51/227], Train loss :1.433, Train acc: 0.6647694934547524
[2024-12-24 18:33:09] - INFO: Epoch: 8, Batch[52/227], Train loss :1.350, Train acc: 0.682648401826484
[2024-12-24 18:33:11] - INFO: Epoch: 8, Batch[53/227], Train loss :1.405, Train acc: 0.6668539325842696
[2024-12-24 18:33:13] - INFO: Epoch: 8, Batch[54/227], Train loss :1.447, Train acc: 0.6586726157278304
[2024-12-24 18:33:15] - INFO: Epoch: 8, Batch[55/227], Train loss :1.465, Train acc: 0.6704545454545454
[2024-12-24 18:33:17] - INFO: Epoch: 8, Batch[56/227], Train loss :1.503, Train acc: 0.6486778197517539
[2024-12-24 18:33:19] - INFO: Epoch: 8, Batch[57/227], Train loss :1.441, Train acc: 0.642692750287687
[2024-12-24 18:33:22] - INFO: Epoch: 8, Batch[58/227], Train loss :1.525, Train acc: 0.6449197860962567
[2024-12-24 18:33:24] - INFO: Epoch: 8, Batch[59/227], Train loss :1.437, Train acc: 0.6676334106728539
[2024-12-24 18:33:27] - INFO: Epoch: 8, Batch[60/227], Train loss :1.472, Train acc: 0.6473537604456825
[2024-12-24 18:33:28] - INFO: Epoch: 8, Batch[61/227], Train loss :1.540, Train acc: 0.6335369239311494
[2024-12-24 18:33:30] - INFO: Epoch: 8, Batch[62/227], Train loss :1.467, Train acc: 0.6770892552586697
[2024-12-24 18:33:32] - INFO: Epoch: 8, Batch[63/227], Train loss :1.492, Train acc: 0.6500559910414334
[2024-12-24 18:33:34] - INFO: Epoch: 8, Batch[64/227], Train loss :1.586, Train acc: 0.6340921710161022
[2024-12-24 18:33:36] - INFO: Epoch: 8, Batch[65/227], Train loss :1.529, Train acc: 0.6556064073226545
[2024-12-24 18:33:38] - INFO: Epoch: 8, Batch[66/227], Train loss :1.477, Train acc: 0.6554347826086957
[2024-12-24 18:33:41] - INFO: Epoch: 8, Batch[67/227], Train loss :1.460, Train acc: 0.6577396893411891
[2024-12-24 18:33:43] - INFO: Epoch: 8, Batch[68/227], Train loss :1.473, Train acc: 0.6526548672566371
[2024-12-24 18:33:46] - INFO: Epoch: 8, Batch[69/227], Train loss :1.477, Train acc: 0.650917176209005
[2024-12-24 18:33:47] - INFO: Epoch: 8, Batch[70/227], Train loss :1.557, Train acc: 0.6338411316648531
[2024-12-24 18:33:50] - INFO: Epoch: 8, Batch[71/227], Train loss :1.530, Train acc: 0.648854961832061
[2024-12-24 18:33:53] - INFO: Epoch: 8, Batch[72/227], Train loss :1.439, Train acc: 0.6627971254836926
[2024-12-24 18:33:55] - INFO: Epoch: 8, Batch[73/227], Train loss :1.540, Train acc: 0.6473265073947668
[2024-12-24 18:33:57] - INFO: Epoch: 8, Batch[74/227], Train loss :1.469, Train acc: 0.6615819209039548
[2024-12-24 18:33:59] - INFO: Epoch: 8, Batch[75/227], Train loss :1.530, Train acc: 0.6358059118795315
[2024-12-24 18:34:02] - INFO: Epoch: 8, Batch[76/227], Train loss :1.489, Train acc: 0.6602491506228766
[2024-12-24 18:34:04] - INFO: Epoch: 8, Batch[77/227], Train loss :1.459, Train acc: 0.6573617952928298
[2024-12-24 18:34:06] - INFO: Epoch: 8, Batch[78/227], Train loss :1.583, Train acc: 0.6304585152838428
[2024-12-24 18:34:08] - INFO: Epoch: 8, Batch[79/227], Train loss :1.527, Train acc: 0.6391139924365208
[2024-12-24 18:34:11] - INFO: Epoch: 8, Batch[80/227], Train loss :1.594, Train acc: 0.6298701298701299
[2024-12-24 18:34:12] - INFO: Epoch: 8, Batch[81/227], Train loss :1.505, Train acc: 0.6543010752688172
[2024-12-24 18:34:14] - INFO: Epoch: 8, Batch[82/227], Train loss :1.427, Train acc: 0.6692056583242655
[2024-12-24 18:34:17] - INFO: Epoch: 8, Batch[83/227], Train loss :1.497, Train acc: 0.6684665226781857
[2024-12-24 18:34:19] - INFO: Epoch: 8, Batch[84/227], Train loss :1.532, Train acc: 0.6513859275053305
[2024-12-24 18:34:22] - INFO: Epoch: 8, Batch[85/227], Train loss :1.447, Train acc: 0.6538461538461539
[2024-12-24 18:34:24] - INFO: Epoch: 8, Batch[86/227], Train loss :1.654, Train acc: 0.6290059750135796
[2024-12-24 18:34:26] - INFO: Epoch: 8, Batch[87/227], Train loss :1.338, Train acc: 0.6893982808022923
[2024-12-24 18:34:29] - INFO: Epoch: 8, Batch[88/227], Train loss :1.451, Train acc: 0.6579091406677614
[2024-12-24 18:34:31] - INFO: Epoch: 8, Batch[89/227], Train loss :1.591, Train acc: 0.6422070534698521
[2024-12-24 18:34:32] - INFO: Epoch: 8, Batch[90/227], Train loss :1.367, Train acc: 0.6854060193072118
[2024-12-24 18:34:34] - INFO: Epoch: 8, Batch[91/227], Train loss :1.498, Train acc: 0.6525847693162868
[2024-12-24 18:34:37] - INFO: Epoch: 8, Batch[92/227], Train loss :1.511, Train acc: 0.6546961325966851
[2024-12-24 18:34:39] - INFO: Epoch: 8, Batch[93/227], Train loss :1.552, Train acc: 0.6468062265163714
[2024-12-24 18:34:41] - INFO: Epoch: 8, Batch[94/227], Train loss :1.493, Train acc: 0.6636670416197975
[2024-12-24 18:34:44] - INFO: Epoch: 8, Batch[95/227], Train loss :1.685, Train acc: 0.6354550236717517
[2024-12-24 18:34:46] - INFO: Epoch: 8, Batch[96/227], Train loss :1.476, Train acc: 0.6518987341772152
[2024-12-24 18:34:49] - INFO: Epoch: 8, Batch[97/227], Train loss :1.546, Train acc: 0.6513812154696133
[2024-12-24 18:34:51] - INFO: Epoch: 8, Batch[98/227], Train loss :1.417, Train acc: 0.6564625850340136
[2024-12-24 18:34:54] - INFO: Epoch: 8, Batch[99/227], Train loss :1.536, Train acc: 0.6511752136752137
[2024-12-24 18:34:56] - INFO: Epoch: 8, Batch[100/227], Train loss :1.502, Train acc: 0.6513865308432372
[2024-12-24 18:34:58] - INFO: Epoch: 8, Batch[101/227], Train loss :1.529, Train acc: 0.6507413509060955
[2024-12-24 18:35:00] - INFO: Epoch: 8, Batch[102/227], Train loss :1.565, Train acc: 0.6385674931129477
[2024-12-24 18:35:03] - INFO: Epoch: 8, Batch[103/227], Train loss :1.569, Train acc: 0.6305555555555555
[2024-12-24 18:35:04] - INFO: Epoch: 8, Batch[104/227], Train loss :1.486, Train acc: 0.6575342465753424
[2024-12-24 18:35:07] - INFO: Epoch: 8, Batch[105/227], Train loss :1.549, Train acc: 0.6504592112371691
[2024-12-24 18:35:09] - INFO: Epoch: 8, Batch[106/227], Train loss :1.440, Train acc: 0.6709141274238227
[2024-12-24 18:35:11] - INFO: Epoch: 8, Batch[107/227], Train loss :1.516, Train acc: 0.6493787142085359
[2024-12-24 18:35:14] - INFO: Epoch: 8, Batch[108/227], Train loss :1.581, Train acc: 0.6361702127659574
[2024-12-24 18:35:16] - INFO: Epoch: 8, Batch[109/227], Train loss :1.619, Train acc: 0.6353467561521253
[2024-12-24 18:35:18] - INFO: Epoch: 8, Batch[110/227], Train loss :1.592, Train acc: 0.6447225244831338
[2024-12-24 18:35:20] - INFO: Epoch: 8, Batch[111/227], Train loss :1.557, Train acc: 0.6427758816837316
[2024-12-24 18:35:23] - INFO: Epoch: 8, Batch[112/227], Train loss :1.502, Train acc: 0.6534154535274356
[2024-12-24 18:35:24] - INFO: Epoch: 8, Batch[113/227], Train loss :1.472, Train acc: 0.6494903737259343
[2024-12-24 18:35:26] - INFO: Epoch: 8, Batch[114/227], Train loss :1.589, Train acc: 0.638692098092643
[2024-12-24 18:35:28] - INFO: Epoch: 8, Batch[115/227], Train loss :1.495, Train acc: 0.657617728531856
[2024-12-24 18:35:30] - INFO: Epoch: 8, Batch[116/227], Train loss :1.607, Train acc: 0.6361655773420479
[2024-12-24 18:35:32] - INFO: Epoch: 8, Batch[117/227], Train loss :1.537, Train acc: 0.6470914127423822
[2024-12-24 18:35:34] - INFO: Epoch: 8, Batch[118/227], Train loss :1.673, Train acc: 0.6174282678002125
[2024-12-24 18:35:37] - INFO: Epoch: 8, Batch[119/227], Train loss :1.533, Train acc: 0.6497747747747747
[2024-12-24 18:35:39] - INFO: Epoch: 8, Batch[120/227], Train loss :1.512, Train acc: 0.6391982182628062
[2024-12-24 18:35:41] - INFO: Epoch: 8, Batch[121/227], Train loss :1.549, Train acc: 0.6465614430665163
[2024-12-24 18:35:43] - INFO: Epoch: 8, Batch[122/227], Train loss :1.599, Train acc: 0.633668903803132
[2024-12-24 18:35:45] - INFO: Epoch: 8, Batch[123/227], Train loss :1.477, Train acc: 0.6517036235803136
[2024-12-24 18:35:47] - INFO: Epoch: 8, Batch[124/227], Train loss :1.489, Train acc: 0.6470588235294118
[2024-12-24 18:35:50] - INFO: Epoch: 8, Batch[125/227], Train loss :1.424, Train acc: 0.6670305676855895
[2024-12-24 18:35:53] - INFO: Epoch: 8, Batch[126/227], Train loss :1.471, Train acc: 0.6628415300546449
[2024-12-24 18:35:55] - INFO: Epoch: 8, Batch[127/227], Train loss :1.504, Train acc: 0.663581927054981
[2024-12-24 18:35:57] - INFO: Epoch: 8, Batch[128/227], Train loss :1.457, Train acc: 0.6632208922742111
[2024-12-24 18:35:59] - INFO: Epoch: 8, Batch[129/227], Train loss :1.496, Train acc: 0.6631046119235096
[2024-12-24 18:36:01] - INFO: Epoch: 8, Batch[130/227], Train loss :1.427, Train acc: 0.6561085972850679
[2024-12-24 18:36:03] - INFO: Epoch: 8, Batch[131/227], Train loss :1.494, Train acc: 0.6541151156535772
[2024-12-24 18:36:05] - INFO: Epoch: 8, Batch[132/227], Train loss :1.495, Train acc: 0.6529445397369925
[2024-12-24 18:36:07] - INFO: Epoch: 8, Batch[133/227], Train loss :1.515, Train acc: 0.6419263456090651
[2024-12-24 18:36:10] - INFO: Epoch: 8, Batch[134/227], Train loss :1.381, Train acc: 0.6668466522678186
[2024-12-24 18:36:12] - INFO: Epoch: 8, Batch[135/227], Train loss :1.578, Train acc: 0.6361679224973089
[2024-12-24 18:36:14] - INFO: Epoch: 8, Batch[136/227], Train loss :1.510, Train acc: 0.6597765363128492
[2024-12-24 18:36:17] - INFO: Epoch: 8, Batch[137/227], Train loss :1.609, Train acc: 0.61742006615215
[2024-12-24 18:36:20] - INFO: Epoch: 8, Batch[138/227], Train loss :1.494, Train acc: 0.6540632054176072
[2024-12-24 18:36:22] - INFO: Epoch: 8, Batch[139/227], Train loss :1.347, Train acc: 0.6835443037974683
[2024-12-24 18:36:25] - INFO: Epoch: 8, Batch[140/227], Train loss :1.491, Train acc: 0.6549643444871092
[2024-12-24 18:36:26] - INFO: Epoch: 8, Batch[141/227], Train loss :1.470, Train acc: 0.6638176638176638
[2024-12-24 18:36:29] - INFO: Epoch: 8, Batch[142/227], Train loss :1.562, Train acc: 0.6493297587131367
[2024-12-24 18:36:30] - INFO: Epoch: 8, Batch[143/227], Train loss :1.396, Train acc: 0.6423188405797101
[2024-12-24 18:36:32] - INFO: Epoch: 8, Batch[144/227], Train loss :1.539, Train acc: 0.6428571428571429
[2024-12-24 18:36:34] - INFO: Epoch: 8, Batch[145/227], Train loss :1.467, Train acc: 0.6504802561366062
[2024-12-24 18:36:37] - INFO: Epoch: 8, Batch[146/227], Train loss :1.676, Train acc: 0.6157662624035282
[2024-12-24 18:36:40] - INFO: Epoch: 8, Batch[147/227], Train loss :1.508, Train acc: 0.6625421822272216
[2024-12-24 18:36:42] - INFO: Epoch: 8, Batch[148/227], Train loss :1.509, Train acc: 0.6520562770562771
[2024-12-24 18:36:43] - INFO: Epoch: 8, Batch[149/227], Train loss :1.350, Train acc: 0.6803601575689364
[2024-12-24 18:36:45] - INFO: Epoch: 8, Batch[150/227], Train loss :1.427, Train acc: 0.6737385321100917
[2024-12-24 18:36:47] - INFO: Epoch: 8, Batch[151/227], Train loss :1.426, Train acc: 0.6563029960429622
[2024-12-24 18:36:49] - INFO: Epoch: 8, Batch[152/227], Train loss :1.540, Train acc: 0.6506849315068494
[2024-12-24 18:36:52] - INFO: Epoch: 8, Batch[153/227], Train loss :1.540, Train acc: 0.6449864498644986
[2024-12-24 18:36:54] - INFO: Epoch: 8, Batch[154/227], Train loss :1.345, Train acc: 0.6823529411764706
[2024-12-24 18:36:56] - INFO: Epoch: 8, Batch[155/227], Train loss :1.559, Train acc: 0.6351575456053068
[2024-12-24 18:36:58] - INFO: Epoch: 8, Batch[156/227], Train loss :1.603, Train acc: 0.639261744966443
[2024-12-24 18:37:01] - INFO: Epoch: 8, Batch[157/227], Train loss :1.539, Train acc: 0.6337402885682575
[2024-12-24 18:37:03] - INFO: Epoch: 8, Batch[158/227], Train loss :1.573, Train acc: 0.6411667583929554
[2024-12-24 18:37:05] - INFO: Epoch: 8, Batch[159/227], Train loss :1.571, Train acc: 0.6448
[2024-12-24 18:37:08] - INFO: Epoch: 8, Batch[160/227], Train loss :1.589, Train acc: 0.6346681294569391
[2024-12-24 18:37:11] - INFO: Epoch: 8, Batch[161/227], Train loss :1.575, Train acc: 0.6450567260940032
[2024-12-24 18:37:13] - INFO: Epoch: 8, Batch[162/227], Train loss :1.627, Train acc: 0.6414674819344081
[2024-12-24 18:37:15] - INFO: Epoch: 8, Batch[163/227], Train loss :1.555, Train acc: 0.6316657504123144
[2024-12-24 18:37:17] - INFO: Epoch: 8, Batch[164/227], Train loss :1.536, Train acc: 0.6513614522156967
[2024-12-24 18:37:19] - INFO: Epoch: 8, Batch[165/227], Train loss :1.461, Train acc: 0.6677985285795133
[2024-12-24 18:37:22] - INFO: Epoch: 8, Batch[166/227], Train loss :1.623, Train acc: 0.6437673130193906
[2024-12-24 18:37:24] - INFO: Epoch: 8, Batch[167/227], Train loss :1.457, Train acc: 0.6523197316936836
[2024-12-24 18:37:26] - INFO: Epoch: 8, Batch[168/227], Train loss :1.474, Train acc: 0.6558033161806747
[2024-12-24 18:37:27] - INFO: Epoch: 8, Batch[169/227], Train loss :1.606, Train acc: 0.6383454443823365
[2024-12-24 18:37:30] - INFO: Epoch: 8, Batch[170/227], Train loss :1.561, Train acc: 0.6443327749860414
[2024-12-24 18:37:33] - INFO: Epoch: 8, Batch[171/227], Train loss :1.508, Train acc: 0.6584564860426929
[2024-12-24 18:37:35] - INFO: Epoch: 8, Batch[172/227], Train loss :1.391, Train acc: 0.6712172923777019
[2024-12-24 18:37:37] - INFO: Epoch: 8, Batch[173/227], Train loss :1.443, Train acc: 0.6653761061946902
[2024-12-24 18:37:40] - INFO: Epoch: 8, Batch[174/227], Train loss :1.573, Train acc: 0.6439645625692137
[2024-12-24 18:37:41] - INFO: Epoch: 8, Batch[175/227], Train loss :1.495, Train acc: 0.6639741518578353
[2024-12-24 18:37:43] - INFO: Epoch: 8, Batch[176/227], Train loss :1.601, Train acc: 0.6329113924050633
[2024-12-24 18:37:46] - INFO: Epoch: 8, Batch[177/227], Train loss :1.555, Train acc: 0.6375570776255708
[2024-12-24 18:37:47] - INFO: Epoch: 8, Batch[178/227], Train loss :1.595, Train acc: 0.6392045454545454
[2024-12-24 18:37:51] - INFO: Epoch: 8, Batch[179/227], Train loss :1.577, Train acc: 0.6545768566493955
[2024-12-24 18:37:54] - INFO: Epoch: 8, Batch[180/227], Train loss :1.481, Train acc: 0.6440860215053763
[2024-12-24 18:37:56] - INFO: Epoch: 8, Batch[181/227], Train loss :1.566, Train acc: 0.6509803921568628
[2024-12-24 18:37:58] - INFO: Epoch: 8, Batch[182/227], Train loss :1.620, Train acc: 0.6382978723404256
[2024-12-24 18:38:01] - INFO: Epoch: 8, Batch[183/227], Train loss :1.606, Train acc: 0.6401741970604246
[2024-12-24 18:38:02] - INFO: Epoch: 8, Batch[184/227], Train loss :1.545, Train acc: 0.6387096774193548
[2024-12-24 18:38:05] - INFO: Epoch: 8, Batch[185/227], Train loss :1.415, Train acc: 0.677382319173364
[2024-12-24 18:38:07] - INFO: Epoch: 8, Batch[186/227], Train loss :1.655, Train acc: 0.610657966286025
[2024-12-24 18:38:10] - INFO: Epoch: 8, Batch[187/227], Train loss :1.453, Train acc: 0.6561620709060214
[2024-12-24 18:38:12] - INFO: Epoch: 8, Batch[188/227], Train loss :1.591, Train acc: 0.6365149833518313
[2024-12-24 18:38:14] - INFO: Epoch: 8, Batch[189/227], Train loss :1.547, Train acc: 0.6623164763458401
[2024-12-24 18:38:16] - INFO: Epoch: 8, Batch[190/227], Train loss :1.656, Train acc: 0.6184782608695653
[2024-12-24 18:38:18] - INFO: Epoch: 8, Batch[191/227], Train loss :1.499, Train acc: 0.6534492123845737
[2024-12-24 18:38:20] - INFO: Epoch: 8, Batch[192/227], Train loss :1.524, Train acc: 0.6467236467236467
[2024-12-24 18:38:23] - INFO: Epoch: 8, Batch[193/227], Train loss :1.687, Train acc: 0.6259863229879011
[2024-12-24 18:38:25] - INFO: Epoch: 8, Batch[194/227], Train loss :1.566, Train acc: 0.644870349492672
[2024-12-24 18:38:27] - INFO: Epoch: 8, Batch[195/227], Train loss :1.415, Train acc: 0.6645496535796767
[2024-12-24 18:38:29] - INFO: Epoch: 8, Batch[196/227], Train loss :1.622, Train acc: 0.6338192419825073
[2024-12-24 18:38:31] - INFO: Epoch: 8, Batch[197/227], Train loss :1.472, Train acc: 0.6539074960127592
[2024-12-24 18:38:33] - INFO: Epoch: 8, Batch[198/227], Train loss :1.580, Train acc: 0.6308615049073064
[2024-12-24 18:38:35] - INFO: Epoch: 8, Batch[199/227], Train loss :1.472, Train acc: 0.6486796785304249
[2024-12-24 18:38:37] - INFO: Epoch: 8, Batch[200/227], Train loss :1.473, Train acc: 0.6622554660529344
[2024-12-24 18:38:40] - INFO: Epoch: 8, Batch[201/227], Train loss :1.579, Train acc: 0.6382978723404256
[2024-12-24 18:38:41] - INFO: Epoch: 8, Batch[202/227], Train loss :1.439, Train acc: 0.6714200831847891
[2024-12-24 18:38:43] - INFO: Epoch: 8, Batch[203/227], Train loss :1.600, Train acc: 0.6419284940411701
[2024-12-24 18:38:46] - INFO: Epoch: 8, Batch[204/227], Train loss :1.585, Train acc: 0.6416666666666667
[2024-12-24 18:38:48] - INFO: Epoch: 8, Batch[205/227], Train loss :1.572, Train acc: 0.6418400876232202
[2024-12-24 18:38:51] - INFO: Epoch: 8, Batch[206/227], Train loss :1.578, Train acc: 0.6556179775280899
[2024-12-24 18:38:53] - INFO: Epoch: 8, Batch[207/227], Train loss :1.605, Train acc: 0.6504018369690011
[2024-12-24 18:38:56] - INFO: Epoch: 8, Batch[208/227], Train loss :1.460, Train acc: 0.6722689075630253
[2024-12-24 18:38:58] - INFO: Epoch: 8, Batch[209/227], Train loss :1.507, Train acc: 0.6515406162464986
[2024-12-24 18:39:00] - INFO: Epoch: 8, Batch[210/227], Train loss :1.527, Train acc: 0.6464120370370371
[2024-12-24 18:39:02] - INFO: Epoch: 8, Batch[211/227], Train loss :1.563, Train acc: 0.6320224719101124
[2024-12-24 18:39:04] - INFO: Epoch: 8, Batch[212/227], Train loss :1.527, Train acc: 0.6412868632707774
[2024-12-24 18:39:07] - INFO: Epoch: 8, Batch[213/227], Train loss :1.616, Train acc: 0.6363636363636364
[2024-12-24 18:39:09] - INFO: Epoch: 8, Batch[214/227], Train loss :1.624, Train acc: 0.6314926189174412
[2024-12-24 18:39:11] - INFO: Epoch: 8, Batch[215/227], Train loss :1.557, Train acc: 0.6434448710916072
[2024-12-24 18:39:14] - INFO: Epoch: 8, Batch[216/227], Train loss :1.629, Train acc: 0.6357333333333334
[2024-12-24 18:39:16] - INFO: Epoch: 8, Batch[217/227], Train loss :1.439, Train acc: 0.6793416572077186
[2024-12-24 18:39:18] - INFO: Epoch: 8, Batch[218/227], Train loss :1.673, Train acc: 0.6276595744680851
[2024-12-24 18:39:21] - INFO: Epoch: 8, Batch[219/227], Train loss :1.532, Train acc: 0.6416849015317286
[2024-12-24 18:39:23] - INFO: Epoch: 8, Batch[220/227], Train loss :1.745, Train acc: 0.6022604951560818
[2024-12-24 18:39:25] - INFO: Epoch: 8, Batch[221/227], Train loss :1.567, Train acc: 0.6398876404494382
[2024-12-24 18:39:27] - INFO: Epoch: 8, Batch[222/227], Train loss :1.628, Train acc: 0.6337950138504155
[2024-12-24 18:39:29] - INFO: Epoch: 8, Batch[223/227], Train loss :1.584, Train acc: 0.6459412780656304
[2024-12-24 18:39:31] - INFO: Epoch: 8, Batch[224/227], Train loss :1.727, Train acc: 0.6091205211726385
[2024-12-24 18:39:34] - INFO: Epoch: 8, Batch[225/227], Train loss :1.656, Train acc: 0.6304347826086957
[2024-12-24 18:39:34] - INFO: Epoch: 8, Batch[226/227], Train loss :1.537, Train acc: 0.6588735387885228
[2024-12-24 18:39:34] - INFO: Epoch: 8, Train loss: 1.509, Epoch time = 506.736s
[2024-12-24 18:39:37] - INFO: Epoch: 9, Batch[0/227], Train loss :1.349, Train acc: 0.6786922209695603
[2024-12-24 18:39:39] - INFO: Epoch: 9, Batch[1/227], Train loss :1.324, Train acc: 0.6742857142857143
[2024-12-24 18:39:41] - INFO: Epoch: 9, Batch[2/227], Train loss :1.319, Train acc: 0.6755379388448471
[2024-12-24 18:39:43] - INFO: Epoch: 9, Batch[3/227], Train loss :1.308, Train acc: 0.6906198573779484
[2024-12-24 18:39:45] - INFO: Epoch: 9, Batch[4/227], Train loss :1.346, Train acc: 0.676552881925014
[2024-12-24 18:39:47] - INFO: Epoch: 9, Batch[5/227], Train loss :1.421, Train acc: 0.6625202812330989
[2024-12-24 18:39:50] - INFO: Epoch: 9, Batch[6/227], Train loss :1.465, Train acc: 0.6609418282548476
[2024-12-24 18:39:52] - INFO: Epoch: 9, Batch[7/227], Train loss :1.478, Train acc: 0.6611702127659574
[2024-12-24 18:39:54] - INFO: Epoch: 9, Batch[8/227], Train loss :1.369, Train acc: 0.6753752084491385
[2024-12-24 18:39:56] - INFO: Epoch: 9, Batch[9/227], Train loss :1.353, Train acc: 0.6757206208425721
[2024-12-24 18:39:58] - INFO: Epoch: 9, Batch[10/227], Train loss :1.318, Train acc: 0.6811434854315558
[2024-12-24 18:40:01] - INFO: Epoch: 9, Batch[11/227], Train loss :1.432, Train acc: 0.677759290072102
[2024-12-24 18:40:03] - INFO: Epoch: 9, Batch[12/227], Train loss :1.281, Train acc: 0.6851540616246499
[2024-12-24 18:40:05] - INFO: Epoch: 9, Batch[13/227], Train loss :1.417, Train acc: 0.6622734761120264
[2024-12-24 18:40:07] - INFO: Epoch: 9, Batch[14/227], Train loss :1.428, Train acc: 0.6534216335540839
[2024-12-24 18:40:10] - INFO: Epoch: 9, Batch[15/227], Train loss :1.279, Train acc: 0.6861642294713161
[2024-12-24 18:40:12] - INFO: Epoch: 9, Batch[16/227], Train loss :1.348, Train acc: 0.6769759450171822
[2024-12-24 18:40:14] - INFO: Epoch: 9, Batch[17/227], Train loss :1.365, Train acc: 0.6688888888888889
[2024-12-24 18:40:17] - INFO: Epoch: 9, Batch[18/227], Train loss :1.366, Train acc: 0.6737193763919822
[2024-12-24 18:40:19] - INFO: Epoch: 9, Batch[19/227], Train loss :1.404, Train acc: 0.6432876712328767
[2024-12-24 18:40:22] - INFO: Epoch: 9, Batch[20/227], Train loss :1.378, Train acc: 0.6783216783216783
[2024-12-24 18:40:24] - INFO: Epoch: 9, Batch[21/227], Train loss :1.364, Train acc: 0.6786885245901639
[2024-12-24 18:40:26] - INFO: Epoch: 9, Batch[22/227], Train loss :1.217, Train acc: 0.7048710601719198
[2024-12-24 18:40:28] - INFO: Epoch: 9, Batch[23/227], Train loss :1.356, Train acc: 0.6738636363636363
[2024-12-24 18:40:30] - INFO: Epoch: 9, Batch[24/227], Train loss :1.363, Train acc: 0.6708004509582863
[2024-12-24 18:40:32] - INFO: Epoch: 9, Batch[25/227], Train loss :1.373, Train acc: 0.6779279279279279
[2024-12-24 18:40:34] - INFO: Epoch: 9, Batch[26/227], Train loss :1.389, Train acc: 0.6777408637873754
[2024-12-24 18:40:36] - INFO: Epoch: 9, Batch[27/227], Train loss :1.331, Train acc: 0.6863611264494754
[2024-12-24 18:40:39] - INFO: Epoch: 9, Batch[28/227], Train loss :1.349, Train acc: 0.6844420010995053
[2024-12-24 18:40:41] - INFO: Epoch: 9, Batch[29/227], Train loss :1.393, Train acc: 0.6600877192982456
[2024-12-24 18:40:43] - INFO: Epoch: 9, Batch[30/227], Train loss :1.343, Train acc: 0.6769558275678552
[2024-12-24 18:40:45] - INFO: Epoch: 9, Batch[31/227], Train loss :1.304, Train acc: 0.6849757673667205
[2024-12-24 18:40:47] - INFO: Epoch: 9, Batch[32/227], Train loss :1.320, Train acc: 0.6798882681564246
[2024-12-24 18:40:49] - INFO: Epoch: 9, Batch[33/227], Train loss :1.435, Train acc: 0.6597995545657016
[2024-12-24 18:40:53] - INFO: Epoch: 9, Batch[34/227], Train loss :1.418, Train acc: 0.6580396475770925
[2024-12-24 18:40:55] - INFO: Epoch: 9, Batch[35/227], Train loss :1.485, Train acc: 0.6552811350499211
[2024-12-24 18:40:56] - INFO: Epoch: 9, Batch[36/227], Train loss :1.409, Train acc: 0.65744920993228
[2024-12-24 18:40:58] - INFO: Epoch: 9, Batch[37/227], Train loss :1.409, Train acc: 0.6537411250682688
[2024-12-24 18:41:00] - INFO: Epoch: 9, Batch[38/227], Train loss :1.450, Train acc: 0.6428969359331477
[2024-12-24 18:41:03] - INFO: Epoch: 9, Batch[39/227], Train loss :1.401, Train acc: 0.6692392502756339
[2024-12-24 18:41:04] - INFO: Epoch: 9, Batch[40/227], Train loss :1.433, Train acc: 0.6503378378378378
[2024-12-24 18:41:06] - INFO: Epoch: 9, Batch[41/227], Train loss :1.325, Train acc: 0.6829823562891292
[2024-12-24 18:41:09] - INFO: Epoch: 9, Batch[42/227], Train loss :1.353, Train acc: 0.6870748299319728
[2024-12-24 18:41:11] - INFO: Epoch: 9, Batch[43/227], Train loss :1.372, Train acc: 0.6603009259259259
[2024-12-24 18:41:13] - INFO: Epoch: 9, Batch[44/227], Train loss :1.389, Train acc: 0.6682926829268293
[2024-12-24 18:41:15] - INFO: Epoch: 9, Batch[45/227], Train loss :1.391, Train acc: 0.663576881134133
[2024-12-24 18:41:17] - INFO: Epoch: 9, Batch[46/227], Train loss :1.447, Train acc: 0.6632096069868996
[2024-12-24 18:41:20] - INFO: Epoch: 9, Batch[47/227], Train loss :1.379, Train acc: 0.6659505907626209
[2024-12-24 18:41:22] - INFO: Epoch: 9, Batch[48/227], Train loss :1.359, Train acc: 0.6785090273733255
[2024-12-24 18:41:25] - INFO: Epoch: 9, Batch[49/227], Train loss :1.467, Train acc: 0.6568993074054342
[2024-12-24 18:41:27] - INFO: Epoch: 9, Batch[50/227], Train loss :1.394, Train acc: 0.6694677871148459
[2024-12-24 18:41:30] - INFO: Epoch: 9, Batch[51/227], Train loss :1.466, Train acc: 0.6633499170812603
[2024-12-24 18:41:31] - INFO: Epoch: 9, Batch[52/227], Train loss :1.436, Train acc: 0.6634285714285715
[2024-12-24 18:41:34] - INFO: Epoch: 9, Batch[53/227], Train loss :1.430, Train acc: 0.6557103064066853
[2024-12-24 18:41:36] - INFO: Epoch: 9, Batch[54/227], Train loss :1.291, Train acc: 0.6867605633802817
[2024-12-24 18:41:38] - INFO: Epoch: 9, Batch[55/227], Train loss :1.428, Train acc: 0.6485260770975056
[2024-12-24 18:41:40] - INFO: Epoch: 9, Batch[56/227], Train loss :1.398, Train acc: 0.6751488900920412
[2024-12-24 18:41:42] - INFO: Epoch: 9, Batch[57/227], Train loss :1.490, Train acc: 0.6473799126637555
[2024-12-24 18:41:44] - INFO: Epoch: 9, Batch[58/227], Train loss :1.467, Train acc: 0.6617179215270413
[2024-12-24 18:41:47] - INFO: Epoch: 9, Batch[59/227], Train loss :1.400, Train acc: 0.6659142212189616
[2024-12-24 18:41:49] - INFO: Epoch: 9, Batch[60/227], Train loss :1.475, Train acc: 0.664218258132214
[2024-12-24 18:41:51] - INFO: Epoch: 9, Batch[61/227], Train loss :1.547, Train acc: 0.6422413793103449
[2024-12-24 18:41:53] - INFO: Epoch: 9, Batch[62/227], Train loss :1.359, Train acc: 0.6755043227665706
[2024-12-24 18:41:56] - INFO: Epoch: 9, Batch[63/227], Train loss :1.450, Train acc: 0.6542409508373852
[2024-12-24 18:41:58] - INFO: Epoch: 9, Batch[64/227], Train loss :1.347, Train acc: 0.6700622524052066
[2024-12-24 18:42:00] - INFO: Epoch: 9, Batch[65/227], Train loss :1.410, Train acc: 0.6662977310459325
[2024-12-24 18:42:03] - INFO: Epoch: 9, Batch[66/227], Train loss :1.517, Train acc: 0.6507936507936508
[2024-12-24 18:42:06] - INFO: Epoch: 9, Batch[67/227], Train loss :1.446, Train acc: 0.6529605263157895
[2024-12-24 18:42:08] - INFO: Epoch: 9, Batch[68/227], Train loss :1.471, Train acc: 0.6623777663407102
[2024-12-24 18:42:11] - INFO: Epoch: 9, Batch[69/227], Train loss :1.481, Train acc: 0.6632816675574559
[2024-12-24 18:42:13] - INFO: Epoch: 9, Batch[70/227], Train loss :1.416, Train acc: 0.6686583378598587
[2024-12-24 18:42:15] - INFO: Epoch: 9, Batch[71/227], Train loss :1.432, Train acc: 0.668903803131991
[2024-12-24 18:42:17] - INFO: Epoch: 9, Batch[72/227], Train loss :1.311, Train acc: 0.6805472932778108
[2024-12-24 18:42:19] - INFO: Epoch: 9, Batch[73/227], Train loss :1.410, Train acc: 0.6560182544209926
[2024-12-24 18:42:22] - INFO: Epoch: 9, Batch[74/227], Train loss :1.422, Train acc: 0.6732505643340858
[2024-12-24 18:42:24] - INFO: Epoch: 9, Batch[75/227], Train loss :1.399, Train acc: 0.6604834176503653
[2024-12-24 18:42:26] - INFO: Epoch: 9, Batch[76/227], Train loss :1.472, Train acc: 0.6566230812961911
[2024-12-24 18:42:27] - INFO: Epoch: 9, Batch[77/227], Train loss :1.404, Train acc: 0.6792134831460674
[2024-12-24 18:42:30] - INFO: Epoch: 9, Batch[78/227], Train loss :1.304, Train acc: 0.6919617762788083
[2024-12-24 18:42:32] - INFO: Epoch: 9, Batch[79/227], Train loss :1.398, Train acc: 0.6719908727895038
[2024-12-24 18:42:34] - INFO: Epoch: 9, Batch[80/227], Train loss :1.401, Train acc: 0.6644736842105263
[2024-12-24 18:42:36] - INFO: Epoch: 9, Batch[81/227], Train loss :1.396, Train acc: 0.6690102758247701
[2024-12-24 18:42:38] - INFO: Epoch: 9, Batch[82/227], Train loss :1.412, Train acc: 0.671021377672209
[2024-12-24 18:42:41] - INFO: Epoch: 9, Batch[83/227], Train loss :1.329, Train acc: 0.684969495285635
[2024-12-24 18:42:42] - INFO: Epoch: 9, Batch[84/227], Train loss :1.532, Train acc: 0.6419423692636073
[2024-12-24 18:42:44] - INFO: Epoch: 9, Batch[85/227], Train loss :1.389, Train acc: 0.6792022792022792
[2024-12-24 18:42:46] - INFO: Epoch: 9, Batch[86/227], Train loss :1.491, Train acc: 0.6440306681270537
[2024-12-24 18:42:49] - INFO: Epoch: 9, Batch[87/227], Train loss :1.439, Train acc: 0.6569506726457399
[2024-12-24 18:42:51] - INFO: Epoch: 9, Batch[88/227], Train loss :1.433, Train acc: 0.6571271326362136
[2024-12-24 18:42:54] - INFO: Epoch: 9, Batch[89/227], Train loss :1.480, Train acc: 0.6526548672566371
[2024-12-24 18:42:56] - INFO: Epoch: 9, Batch[90/227], Train loss :1.366, Train acc: 0.6830786644029428
[2024-12-24 18:42:58] - INFO: Epoch: 9, Batch[91/227], Train loss :1.471, Train acc: 0.6549104720564297
[2024-12-24 18:43:00] - INFO: Epoch: 9, Batch[92/227], Train loss :1.423, Train acc: 0.6760089686098655
[2024-12-24 18:43:02] - INFO: Epoch: 9, Batch[93/227], Train loss :1.541, Train acc: 0.6466205428419372
[2024-12-24 18:43:04] - INFO: Epoch: 9, Batch[94/227], Train loss :1.461, Train acc: 0.6452665941240479
[2024-12-24 18:43:06] - INFO: Epoch: 9, Batch[95/227], Train loss :1.382, Train acc: 0.668517490283176
[2024-12-24 18:43:09] - INFO: Epoch: 9, Batch[96/227], Train loss :1.379, Train acc: 0.673162583518931
[2024-12-24 18:43:11] - INFO: Epoch: 9, Batch[97/227], Train loss :1.421, Train acc: 0.6693136698808848
[2024-12-24 18:43:13] - INFO: Epoch: 9, Batch[98/227], Train loss :1.506, Train acc: 0.6491329479768786
[2024-12-24 18:43:15] - INFO: Epoch: 9, Batch[99/227], Train loss :1.471, Train acc: 0.6523835029459025
[2024-12-24 18:43:18] - INFO: Epoch: 9, Batch[100/227], Train loss :1.373, Train acc: 0.6696478479597541
[2024-12-24 18:43:21] - INFO: Epoch: 9, Batch[101/227], Train loss :1.457, Train acc: 0.6593095642331636
[2024-12-24 18:43:23] - INFO: Epoch: 9, Batch[102/227], Train loss :1.523, Train acc: 0.6489473684210526
[2024-12-24 18:43:25] - INFO: Epoch: 9, Batch[103/227], Train loss :1.408, Train acc: 0.6710601719197707
[2024-12-24 18:43:27] - INFO: Epoch: 9, Batch[104/227], Train loss :1.496, Train acc: 0.6547945205479452
[2024-12-24 18:43:30] - INFO: Epoch: 9, Batch[105/227], Train loss :1.349, Train acc: 0.6782902137232846
[2024-12-24 18:43:31] - INFO: Epoch: 9, Batch[106/227], Train loss :1.441, Train acc: 0.6556934700485699
[2024-12-24 18:43:34] - INFO: Epoch: 9, Batch[107/227], Train loss :1.451, Train acc: 0.6619237336368811
[2024-12-24 18:43:36] - INFO: Epoch: 9, Batch[108/227], Train loss :1.320, Train acc: 0.6789041095890411
[2024-12-24 18:43:38] - INFO: Epoch: 9, Batch[109/227], Train loss :1.420, Train acc: 0.667779632721202
[2024-12-24 18:43:40] - INFO: Epoch: 9, Batch[110/227], Train loss :1.397, Train acc: 0.6670461013090495
[2024-12-24 18:43:42] - INFO: Epoch: 9, Batch[111/227], Train loss :1.429, Train acc: 0.6622734761120264
[2024-12-24 18:43:43] - INFO: Epoch: 9, Batch[112/227], Train loss :1.388, Train acc: 0.671272308578008
[2024-12-24 18:43:46] - INFO: Epoch: 9, Batch[113/227], Train loss :1.396, Train acc: 0.6695890410958905
[2024-12-24 18:43:47] - INFO: Epoch: 9, Batch[114/227], Train loss :1.438, Train acc: 0.6539110861001688
[2024-12-24 18:43:50] - INFO: Epoch: 9, Batch[115/227], Train loss :1.430, Train acc: 0.6613698630136986
[2024-12-24 18:43:52] - INFO: Epoch: 9, Batch[116/227], Train loss :1.528, Train acc: 0.6472191930207197
[2024-12-24 18:43:55] - INFO: Epoch: 9, Batch[117/227], Train loss :1.441, Train acc: 0.6581342434584755
[2024-12-24 18:43:57] - INFO: Epoch: 9, Batch[118/227], Train loss :1.451, Train acc: 0.665938864628821
[2024-12-24 18:44:00] - INFO: Epoch: 9, Batch[119/227], Train loss :1.449, Train acc: 0.652991452991453
[2024-12-24 18:44:02] - INFO: Epoch: 9, Batch[120/227], Train loss :1.319, Train acc: 0.6734581497797357
[2024-12-24 18:44:05] - INFO: Epoch: 9, Batch[121/227], Train loss :1.458, Train acc: 0.6668508287292818
[2024-12-24 18:44:07] - INFO: Epoch: 9, Batch[122/227], Train loss :1.368, Train acc: 0.672316384180791
[2024-12-24 18:44:09] - INFO: Epoch: 9, Batch[123/227], Train loss :1.343, Train acc: 0.6889016676250719
[2024-12-24 18:44:11] - INFO: Epoch: 9, Batch[124/227], Train loss :1.393, Train acc: 0.6696181516325401
[2024-12-24 18:44:13] - INFO: Epoch: 9, Batch[125/227], Train loss :1.448, Train acc: 0.6438730853391685
[2024-12-24 18:44:16] - INFO: Epoch: 9, Batch[126/227], Train loss :1.452, Train acc: 0.6587125416204217
[2024-12-24 18:44:18] - INFO: Epoch: 9, Batch[127/227], Train loss :1.423, Train acc: 0.6566666666666666
[2024-12-24 18:44:20] - INFO: Epoch: 9, Batch[128/227], Train loss :1.440, Train acc: 0.6613272311212814
[2024-12-24 18:44:22] - INFO: Epoch: 9, Batch[129/227], Train loss :1.395, Train acc: 0.6738056013179572
[2024-12-24 18:44:24] - INFO: Epoch: 9, Batch[130/227], Train loss :1.530, Train acc: 0.641390205371248
[2024-12-24 18:44:27] - INFO: Epoch: 9, Batch[131/227], Train loss :1.464, Train acc: 0.6540913921360255
[2024-12-24 18:44:29] - INFO: Epoch: 9, Batch[132/227], Train loss :1.395, Train acc: 0.6660968660968661
[2024-12-24 18:44:31] - INFO: Epoch: 9, Batch[133/227], Train loss :1.465, Train acc: 0.6506287588846364
[2024-12-24 18:44:33] - INFO: Epoch: 9, Batch[134/227], Train loss :1.485, Train acc: 0.65
[2024-12-24 18:44:36] - INFO: Epoch: 9, Batch[135/227], Train loss :1.469, Train acc: 0.6569222283507998
[2024-12-24 18:44:38] - INFO: Epoch: 9, Batch[136/227], Train loss :1.312, Train acc: 0.6874651810584959
[2024-12-24 18:44:41] - INFO: Epoch: 9, Batch[137/227], Train loss :1.351, Train acc: 0.6807538549400343
[2024-12-24 18:44:42] - INFO: Epoch: 9, Batch[138/227], Train loss :1.495, Train acc: 0.6483762597984323
[2024-12-24 18:44:44] - INFO: Epoch: 9, Batch[139/227], Train loss :1.483, Train acc: 0.635663181067694
[2024-12-24 18:44:47] - INFO: Epoch: 9, Batch[140/227], Train loss :1.608, Train acc: 0.6362618914381645
[2024-12-24 18:44:49] - INFO: Epoch: 9, Batch[141/227], Train loss :1.443, Train acc: 0.6609977324263039
[2024-12-24 18:44:51] - INFO: Epoch: 9, Batch[142/227], Train loss :1.423, Train acc: 0.6636556854410202
[2024-12-24 18:44:53] - INFO: Epoch: 9, Batch[143/227], Train loss :1.524, Train acc: 0.6449399656946827
[2024-12-24 18:44:55] - INFO: Epoch: 9, Batch[144/227], Train loss :1.594, Train acc: 0.6259298618490967
[2024-12-24 18:44:57] - INFO: Epoch: 9, Batch[145/227], Train loss :1.499, Train acc: 0.6673960612691466
[2024-12-24 18:45:00] - INFO: Epoch: 9, Batch[146/227], Train loss :1.459, Train acc: 0.6605555555555556
[2024-12-24 18:45:02] - INFO: Epoch: 9, Batch[147/227], Train loss :1.531, Train acc: 0.6349384098544233
[2024-12-24 18:45:04] - INFO: Epoch: 9, Batch[148/227], Train loss :1.439, Train acc: 0.6565315315315315
[2024-12-24 18:45:06] - INFO: Epoch: 9, Batch[149/227], Train loss :1.418, Train acc: 0.6664813785436353
[2024-12-24 18:45:09] - INFO: Epoch: 9, Batch[150/227], Train loss :1.483, Train acc: 0.6549486208761492
[2024-12-24 18:45:11] - INFO: Epoch: 9, Batch[151/227], Train loss :1.501, Train acc: 0.6540389972144847
[2024-12-24 18:45:14] - INFO: Epoch: 9, Batch[152/227], Train loss :1.427, Train acc: 0.6575571667596207
[2024-12-24 18:45:15] - INFO: Epoch: 9, Batch[153/227], Train loss :1.363, Train acc: 0.6782658300057045
[2024-12-24 18:45:17] - INFO: Epoch: 9, Batch[154/227], Train loss :1.499, Train acc: 0.6407551360355358
[2024-12-24 18:45:20] - INFO: Epoch: 9, Batch[155/227], Train loss :1.513, Train acc: 0.6384213580963436
[2024-12-24 18:45:22] - INFO: Epoch: 9, Batch[156/227], Train loss :1.429, Train acc: 0.6557911908646004
[2024-12-24 18:45:24] - INFO: Epoch: 9, Batch[157/227], Train loss :1.442, Train acc: 0.6677833612506979
[2024-12-24 18:45:26] - INFO: Epoch: 9, Batch[158/227], Train loss :1.550, Train acc: 0.6391352549889135
[2024-12-24 18:45:28] - INFO: Epoch: 9, Batch[159/227], Train loss :1.497, Train acc: 0.6626848691695109
[2024-12-24 18:45:30] - INFO: Epoch: 9, Batch[160/227], Train loss :1.562, Train acc: 0.6453257790368272
[2024-12-24 18:45:32] - INFO: Epoch: 9, Batch[161/227], Train loss :1.482, Train acc: 0.6617727025557368
[2024-12-24 18:45:34] - INFO: Epoch: 9, Batch[162/227], Train loss :1.516, Train acc: 0.6470258136924804
[2024-12-24 18:45:36] - INFO: Epoch: 9, Batch[163/227], Train loss :1.517, Train acc: 0.6337562051847766
[2024-12-24 18:45:39] - INFO: Epoch: 9, Batch[164/227], Train loss :1.553, Train acc: 0.6263128800442234
[2024-12-24 18:45:41] - INFO: Epoch: 9, Batch[165/227], Train loss :1.479, Train acc: 0.656072644721907
[2024-12-24 18:45:42] - INFO: Epoch: 9, Batch[166/227], Train loss :1.449, Train acc: 0.6465222348916762
[2024-12-24 18:45:45] - INFO: Epoch: 9, Batch[167/227], Train loss :1.486, Train acc: 0.6704480998298356
[2024-12-24 18:45:47] - INFO: Epoch: 9, Batch[168/227], Train loss :1.476, Train acc: 0.6558011049723756
[2024-12-24 18:45:50] - INFO: Epoch: 9, Batch[169/227], Train loss :1.511, Train acc: 0.6453149814716781
[2024-12-24 18:45:53] - INFO: Epoch: 9, Batch[170/227], Train loss :1.523, Train acc: 0.644880174291939
[2024-12-24 18:45:55] - INFO: Epoch: 9, Batch[171/227], Train loss :1.430, Train acc: 0.6718662952646239
[2024-12-24 18:45:58] - INFO: Epoch: 9, Batch[172/227], Train loss :1.582, Train acc: 0.6249301285634432
[2024-12-24 18:46:00] - INFO: Epoch: 9, Batch[173/227], Train loss :1.445, Train acc: 0.660460021905805
[2024-12-24 18:46:02] - INFO: Epoch: 9, Batch[174/227], Train loss :1.449, Train acc: 0.6649543378995434
[2024-12-24 18:46:05] - INFO: Epoch: 9, Batch[175/227], Train loss :1.546, Train acc: 0.6488423373759648
[2024-12-24 18:46:07] - INFO: Epoch: 9, Batch[176/227], Train loss :1.552, Train acc: 0.6439153439153439
[2024-12-24 18:46:09] - INFO: Epoch: 9, Batch[177/227], Train loss :1.472, Train acc: 0.6477151965993624
[2024-12-24 18:46:12] - INFO: Epoch: 9, Batch[178/227], Train loss :1.515, Train acc: 0.6420605416887945
[2024-12-24 18:46:14] - INFO: Epoch: 9, Batch[179/227], Train loss :1.538, Train acc: 0.6392978482446207
[2024-12-24 18:46:16] - INFO: Epoch: 9, Batch[180/227], Train loss :1.479, Train acc: 0.6646673936750273
[2024-12-24 18:46:19] - INFO: Epoch: 9, Batch[181/227], Train loss :1.523, Train acc: 0.65625
[2024-12-24 18:46:21] - INFO: Epoch: 9, Batch[182/227], Train loss :1.502, Train acc: 0.6520562770562771
[2024-12-24 18:46:24] - INFO: Epoch: 9, Batch[183/227], Train loss :1.565, Train acc: 0.6484290357529794
[2024-12-24 18:46:26] - INFO: Epoch: 9, Batch[184/227], Train loss :1.528, Train acc: 0.6338329764453962
[2024-12-24 18:46:28] - INFO: Epoch: 9, Batch[185/227], Train loss :1.316, Train acc: 0.681994459833795
[2024-12-24 18:46:30] - INFO: Epoch: 9, Batch[186/227], Train loss :1.439, Train acc: 0.6621401412275937
[2024-12-24 18:46:32] - INFO: Epoch: 9, Batch[187/227], Train loss :1.476, Train acc: 0.6549145299145299
[2024-12-24 18:46:34] - INFO: Epoch: 9, Batch[188/227], Train loss :1.451, Train acc: 0.6630057803468208
[2024-12-24 18:46:36] - INFO: Epoch: 9, Batch[189/227], Train loss :1.422, Train acc: 0.6636568848758465
[2024-12-24 18:46:39] - INFO: Epoch: 9, Batch[190/227], Train loss :1.490, Train acc: 0.6597487711632988
[2024-12-24 18:46:41] - INFO: Epoch: 9, Batch[191/227], Train loss :1.557, Train acc: 0.6335303600214938
[2024-12-24 18:46:43] - INFO: Epoch: 9, Batch[192/227], Train loss :1.634, Train acc: 0.6270270270270271
[2024-12-24 18:46:45] - INFO: Epoch: 9, Batch[193/227], Train loss :1.320, Train acc: 0.6837507280139778
[2024-12-24 18:46:48] - INFO: Epoch: 9, Batch[194/227], Train loss :1.440, Train acc: 0.6634341129492299
[2024-12-24 18:46:50] - INFO: Epoch: 9, Batch[195/227], Train loss :1.296, Train acc: 0.688533941814033
[2024-12-24 18:46:52] - INFO: Epoch: 9, Batch[196/227], Train loss :1.463, Train acc: 0.656781987918726
[2024-12-24 18:46:55] - INFO: Epoch: 9, Batch[197/227], Train loss :1.431, Train acc: 0.6584684181106764
[2024-12-24 18:46:57] - INFO: Epoch: 9, Batch[198/227], Train loss :1.516, Train acc: 0.6457750419697817
[2024-12-24 18:46:59] - INFO: Epoch: 9, Batch[199/227], Train loss :1.532, Train acc: 0.6475455046883618
[2024-12-24 18:47:02] - INFO: Epoch: 9, Batch[200/227], Train loss :1.476, Train acc: 0.6465141612200436
[2024-12-24 18:47:04] - INFO: Epoch: 9, Batch[201/227], Train loss :1.510, Train acc: 0.646442360728075
[2024-12-24 18:47:07] - INFO: Epoch: 9, Batch[202/227], Train loss :1.461, Train acc: 0.6569736133548735
[2024-12-24 18:47:10] - INFO: Epoch: 9, Batch[203/227], Train loss :1.596, Train acc: 0.6294085729788389
[2024-12-24 18:47:12] - INFO: Epoch: 9, Batch[204/227], Train loss :1.509, Train acc: 0.6561822125813449
[2024-12-24 18:47:14] - INFO: Epoch: 9, Batch[205/227], Train loss :1.577, Train acc: 0.6322511974454497
[2024-12-24 18:47:16] - INFO: Epoch: 9, Batch[206/227], Train loss :1.494, Train acc: 0.6444695259593679
[2024-12-24 18:47:19] - INFO: Epoch: 9, Batch[207/227], Train loss :1.539, Train acc: 0.6507150715071507
[2024-12-24 18:47:21] - INFO: Epoch: 9, Batch[208/227], Train loss :1.453, Train acc: 0.6526195899772209
[2024-12-24 18:47:24] - INFO: Epoch: 9, Batch[209/227], Train loss :1.456, Train acc: 0.6608026388125343
[2024-12-24 18:47:26] - INFO: Epoch: 9, Batch[210/227], Train loss :1.565, Train acc: 0.6470261256253474
[2024-12-24 18:47:28] - INFO: Epoch: 9, Batch[211/227], Train loss :1.536, Train acc: 0.6430921052631579
[2024-12-24 18:47:30] - INFO: Epoch: 9, Batch[212/227], Train loss :1.450, Train acc: 0.6631165919282511
[2024-12-24 18:47:33] - INFO: Epoch: 9, Batch[213/227], Train loss :1.547, Train acc: 0.6535733769776323
[2024-12-24 18:47:35] - INFO: Epoch: 9, Batch[214/227], Train loss :1.438, Train acc: 0.6532951289398281
[2024-12-24 18:47:37] - INFO: Epoch: 9, Batch[215/227], Train loss :1.661, Train acc: 0.6138040042149632
[2024-12-24 18:47:40] - INFO: Epoch: 9, Batch[216/227], Train loss :1.446, Train acc: 0.6672131147540984
[2024-12-24 18:47:43] - INFO: Epoch: 9, Batch[217/227], Train loss :1.650, Train acc: 0.6318918918918919
[2024-12-24 18:47:45] - INFO: Epoch: 9, Batch[218/227], Train loss :1.567, Train acc: 0.6352941176470588
[2024-12-24 18:47:47] - INFO: Epoch: 9, Batch[219/227], Train loss :1.514, Train acc: 0.6549375709421112
[2024-12-24 18:47:49] - INFO: Epoch: 9, Batch[220/227], Train loss :1.425, Train acc: 0.6742209631728046
[2024-12-24 18:47:51] - INFO: Epoch: 9, Batch[221/227], Train loss :1.431, Train acc: 0.6606359649122807
[2024-12-24 18:47:53] - INFO: Epoch: 9, Batch[222/227], Train loss :1.604, Train acc: 0.6328690807799443
[2024-12-24 18:47:55] - INFO: Epoch: 9, Batch[223/227], Train loss :1.452, Train acc: 0.665725578769057
[2024-12-24 18:47:57] - INFO: Epoch: 9, Batch[224/227], Train loss :1.629, Train acc: 0.6472527472527473
[2024-12-24 18:48:00] - INFO: Epoch: 9, Batch[225/227], Train loss :1.643, Train acc: 0.627955859169732
[2024-12-24 18:48:01] - INFO: Epoch: 9, Batch[226/227], Train loss :1.747, Train acc: 0.6054298642533936
[2024-12-24 18:48:01] - INFO: Epoch: 9, Train loss: 1.444, Epoch time = 506.720s
[2024-12-24 18:48:06] - INFO: Accuracy on validation0.560
```

训练10个epoch之后，在测试集上可以达到在最大准确率为`0.874`。

### 5.4 预测（inference）

直接运行如下命令即可：

```shell
python translate.py
```

示例结果：

云平台结果

```shell
德语：Eine Gruppe von Menschen steht vor einem Iglu.
翻译： A people outside in an wilderness 
英语：A group of people are facing an igloo.


德语：Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.
翻译： A man a blue cleaning machine outside a window . 
英语：A man in a blue shirt is standing on a ladder cleaning a window.
```

本地运行结果：

```bash
德语：Eine Gruppe von Menschen steht vor einem Iglu.
翻译： A group outside .
英语：A group of people are facing an igloo.


德语：Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.
翻译： A man in a building is standing next to a blue shirt .
英语：A man in a blue shirt is standing on a ladder cleaning a window.
```

