'''
Detailed bilingual comments for the code.
对下面的代码进行详细的中英文注释，帮助读者更好地理解代码的功能和流程。
'''

import numpy as np  # Import numpy for numerical computations.
                     # 导入numpy进行数值计算。
import pandas as pd  # Import pandas for data manipulation and analysis.
                     # 导入pandas进行数据处理和分析。
import torch  # Import PyTorch for deep learning.
              # 导入PyTorch用于深度学习任务。
import torch.nn as nn  # Import the neural network module from PyTorch.
                       # 导入PyTorch中的神经网络模块。
from torch.utils.data import Dataset, DataLoader  # Import Dataset and DataLoader for handling data in batches.
                                                   # 导入Dataset和DataLoader以便批量处理数据。
from scipy.stats import spearmanr, pearsonr  # Import correlation functions for evaluation metrics.
                                             # 导入计算Spearman和Pearson相关系数的函数。
from model import PrimeNet  # Import the PrimeNet model from an external module.
                           # 从model模块中导入PrimeNet模型。
import ast  # Import ast to safely parse string expressions.
           # 导入ast模块以安全地将字符串解析为Python表达式。

# --------------------------------------------------------------------------------
# Function: generate_synthetic_image
# 功能：生成8通道伪图像，用于深度学习模型的输入。
# --------------------------------------------------------------------------------
def generate_synthetic_image(dna_seq, dnase_seq, methylation_seq, protospacerlocation=None, RT_initial_location=None, PBSlocation=None, RT_mutated_location=None, target_length=128):
    """
    Generate an 8-channel synthetic image for deep learning model input.
    生成8通道伪图像，用于深度学习模型输入。
    
    Parameters/参数:
        dna_seq: DNA sequence string (A/G/T/C).
                 DNA序列字符串（由A、G、T、C组成）。
        dnase_seq: DNase accessibility states ('Y' for yes, 'N' for no).
                   DNase可及性状态（'Y'表示可及，'N'表示不可及）。
        methylation_seq: Methylation states ('Y' for methylated, 'N' for unmethylated).
                         甲基化状态（'Y'表示甲基化，'N'表示未甲基化）。
        protospacerlocation: String representing protospacer position range, e.g., "[start, end]".
                             表示protospacer位置范围的字符串，例如"[起始位置, 结束位置]"。
        RT_initial_location: String representing the initial location range for reverse transcriptase.
                             表示逆转录酶初始位置范围的字符串。
        PBSlocation: String representing the PBS location range.
                     表示PBS位置范围的字符串。
        RT_mutated_location: String representing the mutated location range for reverse transcriptase.
                             表示逆转录酶突变位置范围的字符串。
        target_length: Integer for the target sequence length (default is 128).
                       整数，表示目标序列长度（默认128）。
                       
    Returns/返回:
        synthetic_image: A numpy array of shape (target_length, 8, 1) representing the synthetic image.
                         一个形状为(target_length, 8, 1)的numpy数组，表示生成的伪图像。
    """
    synthetic_image = np.zeros((target_length, 8, 1))  # Initialize an array with zeros for 8 channels.
                                                    # 初始化一个8通道、全零的数组。

    # Encode DNA sequence into channels 0-3.
    # 将DNA序列按碱基类型编码到通道0到3中。
    for i, base in enumerate(dna_seq):
        if base == 'A':
            synthetic_image[i, 0, 0] = 1  # Channel 0 for Adenine (A).
                                        # 通道0表示腺嘌呤（A）。
        elif base == 'G':
            synthetic_image[i, 1, 0] = 1  # Channel 1 for Guanine (G).
                                        # 通道1表示鸟嘌呤（G）。
        elif base == 'T':
            synthetic_image[i, 2, 0] = 1  # Channel 2 for Thymine (T).
                                        # 通道2表示胸腺嘧啶（T）。
        elif base == 'C':
            synthetic_image[i, 3, 0] = 1  # Channel 3 for Cytosine (C).
                                        # 通道3表示胞嘧啶（C）。

    # Encode DNase accessibility into channel 4.
    # 将DNase可及性状态编码到通道4中：'Y'记为1，否则记为0。
    for i, state in enumerate(dnase_seq):
        synthetic_image[i, 4, 0] = 1 if state == 'Y' else 0

    # Encode methylation state into channel 5.
    # 将甲基化状态编码到通道5中：'Y'记为1，否则记为0。
    for i, state in enumerate(methylation_seq):
        synthetic_image[i, 5, 0] = 1 if state == 'Y' else 0

    # Parse and annotate interval data into channels 6 and 7.
    # 解析并将区间数据标注到通道6和7中。

    # Process protospacer location and mark on channel 6.
    # 处理protospacer位置数据，并在通道6中标注。
    if protospacerlocation:
        protospacerlocation = ast.literal_eval(protospacerlocation)  # Safely parse string into list.
                                                                       # 安全地将字符串解析为列表。
        # Mark positions within the protospacer range.
        # 标记protospacer范围内的位置（注意索引从0开始，因此i-1）。
        for i in range(protospacerlocation[0], protospacerlocation[1] + 1):
            if i < target_length:
                synthetic_image[i-1, 6, 0] = 1
        # Extend annotation to 3 positions following the protospacer.
        # 在protospacer结束后延伸3个位置进行标注。
        for i in range(protospacerlocation[1]+1, protospacerlocation[1] + 4):
            if i < target_length:
                synthetic_image[i-1, 6, 0] = 1

    # Process reverse transcriptase initial location and mark on channel 7.
    # 处理逆转录酶初始位置数据，并在通道7中标注。
    if RT_initial_location:
        RT_initial_location = ast.literal_eval(RT_initial_location)
        for i in range(RT_initial_location[0], RT_initial_location[1] + 1):
            if i < target_length:
                synthetic_image[i-1, 7, 0] = 1

    # Process PBS location and mark on channel 6.
    # 处理PBS位置数据，并在通道6中标注。
    if PBSlocation:
        PBSlocation = ast.literal_eval(PBSlocation)
        for i in range(PBSlocation[0], PBSlocation[1] + 1):
            if i < target_length:
                synthetic_image[i-1, 6, 0] = 1

    # Process mutated reverse transcriptase location and mark on channel 7.
    # 处理逆转录酶突变位置数据，并在通道7中标注。
    if RT_mutated_location:
        RT_mutated_location = ast.literal_eval(RT_mutated_location)
        for i in range(RT_mutated_location[0], RT_mutated_location[1] + 1):
            if i < target_length:
                synthetic_image[i-1, 7, 0] = 1
                
    return synthetic_image  # Return the constructed synthetic image.
                           # 返回生成的伪图像数组。

# --------------------------------------------------------------------------------
# Custom Dataset Class: SequenceDataset
# 自定义数据集类，用于加载序列数据并生成伪图像。
# --------------------------------------------------------------------------------
class SequenceDataset(Dataset):
    """
    Custom dataset class to load sequence data and generate synthetic images.
    自定义数据集类，用于加载序列数据并生成伪图像。
    """
    def __init__(self, data):
        """
        Initialize the dataset with a pandas DataFrame.
        用一个pandas DataFrame初始化数据集。
        
        Parameters/参数:
            data: A pandas DataFrame containing sequence and annotation data.
                  包含序列及注释数据的pandas DataFrame。
        """
        self.data = data  # Store the DataFrame.
                         # 保存数据集。

    def __len__(self):
        """
        Return the total number of samples.
        返回数据集中样本的总数。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the sample at index 'idx', generate synthetic images, and retrieve target labels.
        根据索引获取样本，生成伪图像，并提取目标标签。
        
        Parameters/参数:
            idx: Index of the sample.
                 样本的索引。
                 
        Returns/返回:
            A tuple (combined_image, target) where:
            combined_image: The concatenated synthetic image (initial and mutated parts).
                            初始和突变部分伪图像的合并结果。
            target: A tensor containing the target labels.
                    包含目标标签的张量。
        """
        row = self.data.iloc[idx]  # Get the row corresponding to the sample.
                                  # 从DataFrame中获取对应样本的行数据。
        # Generate synthetic image for the initial sequence.
        # 为初始序列生成伪图像。
        image1 = generate_synthetic_image(
            dna_seq=row['wide_initial_target'],
            dnase_seq=row['initial_dnase'],
            methylation_seq=row['initial_methylation'],
            protospacerlocation=row['protospacerlocation_only_initial'],
            RT_initial_location=row['RT_initial_location']
        )
        # Generate synthetic image for the mutated sequence.
        # 为突变序列生成伪图像。
        image2 = generate_synthetic_image(
            dna_seq=row['wide_mutated_target'],
            dnase_seq=row['mutated_dnase'],
            methylation_seq=row['mutated_methylation'],
            PBSlocation=row['PBSlocation'],
            RT_mutated_location=row['RT_mutated_location']
        )

        # Concatenate the two images along the depth dimension.
        # 将初始与突变伪图像在深度（通道）维度上合并。
        combined_image = np.concatenate((image1, image2), axis=2)
        # Convert the combined numpy array to a PyTorch tensor and adjust dimensions.
        # 将合并后的numpy数组转换为PyTorch张量，并调整维度以符合模型要求。
        combined_image = torch.tensor(combined_image, dtype=torch.float32).permute(1, 0, 2)

        # Create the target tensor from the labels.
        # 根据目标标签构造目标张量。
        target = torch.tensor([row['Validly_Edited'], row['Unedited'], row['Erroneously_Edited']], dtype=torch.float32)
        return combined_image, target

# --------------------------------------------------------------------------------
# Load test data and create DataLoader.
# 加载测试数据并创建DataLoader以便批量处理。
# --------------------------------------------------------------------------------
test_data = pd.read_csv("PrimeNet/data/test_data.csv")  # Load test CSV data.
                                                       # 使用pandas加载测试数据CSV文件。
test_dataset = SequenceDataset(test_data)  # Create a SequenceDataset instance.
                                          # 创建SequenceDataset实例。
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)  # Create DataLoader with batch size 128.
                                                                       # 使用批量大小128创建DataLoader，不进行乱序。

# --------------------------------------------------------------------------------
# Set device for computation.
# 设置计算设备（如果有GPU则使用GPU，否则使用CPU）。
# --------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device.
                                                                       # 根据环境设置设备。

# --------------------------------------------------------------------------------
# Initialize the model and load pre-trained weights.
# 初始化模型并加载预训练权重。
# --------------------------------------------------------------------------------
model = PrimeNet().to(device)  # Initialize the PrimeNet model and move it to the device.
                              # 初始化PrimeNet模型，并将其移动到计算设备上。
# Load the state dictionary containing pre-trained weights.
# 加载保存的预训练权重（state_dict）。
state_dict = torch.load('PrimeNet/PrimeNet.pth', map_location=device, weights_only=True)

# If the model was saved using multiple GPUs, remove the "module." prefix.
# 如果权重保存时使用了多GPU模式，则去除参数名中的“module.”前缀。
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

model.load_state_dict(state_dict)  # Load weights into the model.
                                   # 将权重加载到模型中。

# --------------------------------------------------------------------------------
# Define evaluation metrics function.
# 定义评估指标函数，计算Pearson和Spearman相关系数。
# --------------------------------------------------------------------------------
def evaluate_metrics(true_labels, pred_labels):
    """
    Compute Pearson and Spearman correlation coefficients for each target label.
    计算每个目标标签对应的Pearson和Spearman相关系数。
    
    Parameters/参数:
        true_labels: Array-like of true label values.
                     真实标签数组。
        pred_labels: Array-like of predicted label values.
                     预测标签数组。
                     
    Returns/返回:
        metrics: A dictionary mapping each label to its Pearson and Spearman correlations.
                 字典，其中键为标签名称，值为包含'Pearson'和'Spearman'相关系数的子字典。
    """
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    metrics = {}

    # Calculate metrics for each label.
    # 对每个标签计算相关系数指标。
    for i, label in enumerate(["Validly_Edited", "Unedited", "Erroneously_Edited"]):
        # Check if data exists for calculation.
        # 检查对应标签是否有数据进行计算。
        if true_labels[:, i].size == 0 or pred_labels[:, i].size == 0:
            raise ValueError(f"No data for metric calculation for {label}.")
        
        # Calculate Pearson correlation coefficient.
        # 计算Pearson相关系数。
        pearson_corr, _ = pearsonr(true_labels[:, i], pred_labels[:, i])
        # Calculate Spearman correlation coefficient.
        # 计算Spearman相关系数。
        spearman_corr = spearmanr(true_labels[:, i], pred_labels[:, i]).correlation
        # Store the computed metrics.
        # 将计算结果存入字典中。
        metrics[label] = {'Pearson': pearson_corr, 'Spearman': spearman_corr}
    return metrics

# --------------------------------------------------------------------------------
# Model evaluation on test data.
# 在测试数据上评估模型性能，包括计算损失和相关系数。
# --------------------------------------------------------------------------------
model.eval()  # Set the model to evaluation mode (disables dropout, etc.).
              # 将模型设置为评估模式（禁用dropout等训练时的机制）。
true_labels, pred_labels = [], []  # Initialize lists to collect true and predicted labels.
test_loss = 0.0  # Initialize cumulative test loss.
criterion = nn.MSELoss()  # Define Mean Squared Error loss for regression.
                           # 定义均方误差作为回归任务的损失函数。

# Disable gradient calculation for evaluation to save memory.
# 评估时禁用梯度计算，节省内存。
with torch.no_grad():
    for image, target in test_loader:
        image, target = image.to(device), target.to(device)  # Move batch data to the device.
                                                            # 将数据批次移动到计算设备上。
        output = model(image).squeeze()  # Forward pass: compute model output and remove extra dimensions.
                                       # 前向传播：计算模型输出并去除多余的维度。
        loss = criterion(output, target)  # Compute the loss for the current batch.
                                         # 计算当前批次的损失。
        test_loss += loss.item()  # Accumulate the loss.
                                 # 累加损失值。
        true_labels.extend(target.cpu().numpy())  # Collect ground truth labels.
                                                  # 收集真实标签。
        pred_labels.extend(output.cpu().numpy())  # Collect predicted labels.
                                                   # 收集预测标签。

# Calculate the average test loss.
# 计算平均测试损失。
avg_test_loss = test_loss / len(test_loader)
# Compute correlation metrics for the test set.
# 计算测试集上的相关性指标（Pearson和Spearman）。
test_metrics = evaluate_metrics(true_labels, pred_labels)

# --------------------------------------------------------------------------------
# Output test results.
# 输出测试结果：平均损失及各目标标签的相关系数。
# --------------------------------------------------------------------------------
print(f"Test Loss: {avg_test_loss:.4f}")  # Print average loss.
                                         # 打印平均测试损失。
for key, metric in test_metrics.items():
    # Print the Pearson and Spearman correlations for each label.
    # 打印每个目标标签的Pearson和Spearman相关系数。
    print(f"{key} - Pearson: {metric['Pearson']:.4f}, Spearman: {metric['Spearman']:.4f}")