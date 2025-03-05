'''
This code uses bilingual comments, with English comments followed by their Chinese counterparts. 
The meaning expressed in both languages is identical, ensuring clarity for a wider audience.
本代码采用双语注释，英文注释在前，中文注释在后。
两种语言表达的意思完全一致，以确保更广泛的读者能够理解。
'''

# Import required libraries
# 导入必要的库
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from model import PrimeNet
import ast


# Define function to generate 8-channel synthetic image
# 定义生成8通道伪图像的函数
def generate_synthetic_image(dna_seq, dnase_seq, methylation_seq, protospacerlocation=None, RT_initial_location=None, PBSlocation=None, RT_mutated_location=None, target_length=128):
    """
    Generate 8-channel synthetic image for deep learning model input
    生成8通道伪图像，用于深度学习模型输入
    
    Args/参数:
        dna_seq: DNA sequence (A/G/T/C) / DNA序列 (A/G/T/C)
        dnase_seq: DNase accessibility states (Y/N) / DNase可及性状态 (Y/N)
        methylation_seq: Methylation states (Y/N) / 甲基化状态 (Y/N)
        protospacerlocation: Protospacer location range / Protospacer位置范围
        RT_initial_location: Reverse transcriptase initial location range / 逆转录酶初始位置范围
        PBSlocation: PBS location range / PBS位置范围
        RT_mutated_location: Reverse transcriptase mutated location range / 逆转录酶突变位置范围
        target_length: Target sequence length (default 128) / 目标序列长度 (默认128)
    """
    synthetic_image = np.zeros((target_length, 8, 1))  # 8 channels for different features/8个通道表示不同特征

    # Encode DNA sequence (Channels 0-3)
    # 编码DNA序列（通道0-3）
    for i, base in enumerate(dna_seq):
        if base == 'A':
            synthetic_image[i, 0, 0] = 1
        elif base == 'G':
            synthetic_image[i, 1, 0] = 1
        elif base == 'T':
            synthetic_image[i, 2, 0] = 1
        elif base == 'C':
            synthetic_image[i, 3, 0] = 1

    # Encode DNase accessibility (Channel 4)
    # 编码DNase可及性（通道4）
    for i, state in enumerate(dnase_seq):
        synthetic_image[i, 4, 0] = 1 if state == 'Y' else 0

    # Encode methylation status (Channel 5)
    # 编码甲基化状态（通道5）
    for i, state in enumerate(methylation_seq):
        synthetic_image[i, 5, 0] = 1 if state == 'Y' else 0

    # Encode location ranges (Channels 6-7)
    # 编码位置区间信息（通道6-7）
    if protospacerlocation:
        protospacerlocation = ast.literal_eval(protospacerlocation)
        for i in range(protospacerlocation[0], protospacerlocation[1] + 1):
            if i < target_length:
                synthetic_image[i-1, 6, 0] = 1
        for i in range(protospacerlocation[1]+1, protospacerlocation[1] + 4):
            if i < target_length:
                synthetic_image[i-1, 6, 0] = 1
    
    if RT_initial_location:
        RT_initial_location = ast.literal_eval(RT_initial_location)
        for i in range(RT_initial_location[0], RT_initial_location[1] + 1):
            if i < target_length:
                synthetic_image[i-1, 7, 0] = 1

    if PBSlocation:
        PBSlocation = ast.literal_eval(PBSlocation)
        for i in range(PBSlocation[0], PBSlocation[1] + 1):
            if i < target_length:
                synthetic_image[i-1, 6, 0] = 1

    if RT_mutated_location:
        RT_mutated_location = ast.literal_eval(RT_mutated_location)
        for i in range(RT_mutated_location[0], RT_mutated_location[1] + 1):
            if i < target_length:
                synthetic_image[i-1, 7, 0] = 1
                
    return synthetic_image


# Custom Dataset class for sequence data
# 自定义序列数据集类
class SequenceDataset(Dataset):
    """
    Custom dataset for loading and processing sequence data
    自定义数据集类，用于加载和处理序列数据
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Generate synthetic images for initial and mutated sequences
        # 生成初始和突变序列的合成图像
        image1 = generate_synthetic_image(
            dna_seq=row['wide_initial_target'],
            dnase_seq=row['initial_dnase'],
            methylation_seq=row['initial_methylation'],
            protospacerlocation=row['protospacerlocation_only_initial'],
            RT_initial_location=row['RT_initial_location']
        )
        image2 = generate_synthetic_image(
            dna_seq=row['wide_mutated_target'],
            dnase_seq=row['mutated_dnase'],
            methylation_seq=row['mutated_methylation'],
            PBSlocation=row['PBSlocation'],
            RT_mutated_location=row['RT_mutated_location']
        )

        # Combine images and convert to tensor
        # 合并图像并转换为张量
        combined_image = np.concatenate((image1, image2), axis=2)
        combined_image = torch.tensor(combined_image, dtype=torch.float32).permute(1, 0, 2)

        # Target values (three editing metrics)
        # 目标值（三个编辑指标）
        target = torch.tensor([row['Validly_Edited'], row['Unedited'], row['Erroneously_Edited']], dtype=torch.float32)
        return combined_image, target


# Load datasets
# 加载数据集
train_data = pd.read_csv("PrimeNet/data/train_data.csv")
val_data = pd.read_csv("PrimeNet/data/val_data.csv")

# Create data loaders
# 创建数据加载器
train_dataset = SequenceDataset(train_data)
val_dataset = SequenceDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)


# Initialize model with orthogonal initialization
# 使用正交初始化初始化模型
def orthogonal_init(layer):
    """Apply orthogonal initialization to linear/conv layers/对线性/卷积层应用正交初始化"""
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        torch.nn.init.orthogonal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)

# Detect available device
# 检测可用设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize model and apply initialization
# 初始化模型并应用初始化
model = PrimeNet().to(device)
model.apply(orthogonal_init)


# Lookahead optimizer implementation
# Lookahead优化器实现
class Lookahead(torch.optim.Optimizer):
    """
Implements Lookahead Optimizer Algorithm with dual-weight update mechanism
实现具有双权重更新机制的Lookahead优化器算法

Key Characteristics/核心特性:
1. Maintains both fast weights (base optimizer) and slow weights (lookahead)
   同时维护快速权重（基础优化器）和慢速权重（lookahead）
2. Performs periodic synchronization between two weight sets
   定期执行两个权重集的同步
3. Reduces optimization variance and stabilizes training
   降低优化方差并稳定训练过程

Args/参数:
    base_optimizer: Inner optimizer for fast updates (e.g. Adam, SGD)
                    基础优化器，负责快速权重更新（如Adam, SGD）
    k (int): Number of fast steps between slow updates 
             慢速权重更新间隔步数（默认5步）
    alpha (float): Linear interpolation coefficient (slow_weights = slow_weights + alpha * (fast_weights - slow_weights))
                  慢速权重更新时的线性插值系数（默认0.5）
    """
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not k >= 1:
            raise ValueError("k must be at least 1")

        self.optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.slow_weights = [[p.clone().detach() for p in group['params']] for group in self.param_groups]
        for w in self.slow_weights:
            for p in w:
                p.requires_grad = False
        self.counter = 0

        # Initialize base optimizer
        # 初始化基础优化器
        defaults = {"k": k, "alpha": alpha}
        super().__init__(self.param_groups, defaults)

    def step(self, closure=None):
        """Perform optimization step/执行优化步骤"""
        loss = self.optimizer.step(closure)
        self.counter += 1

        if self.counter % self.k == 0:
            # Save base optimizer state
            # 保存基础优化器状态
            base_state = {
                p: (self.optimizer.state[p]['exp_avg'], 
                    self.optimizer.state[p]['exp_avg_sq'])
                for p in self.optimizer.param_groups[0]['params']
            }

            # Update slow weights
            # 更新慢权重
            for group, slow_weights in zip(self.param_groups, self.slow_weights):
                for p, q in zip(group['params'], slow_weights):
                    q.data.add_(p.data - q.data, alpha=self.alpha)
                    p.data.copy_(q.data)
            
            # Restore base optimizer state
            # 恢复基础优化器状态
            for p in self.optimizer.param_groups[0]['params']:
                self.optimizer.state[p]['exp_avg'], self.optimizer.state[p]['exp_avg_sq'] = base_state[p]
        
        return loss

    def zero_grad(self, set_to_none=False):
        """Clear gradients/清空梯度"""
        self.optimizer.zero_grad(set_to_none=set_to_none)


# Initialize optimizer and loss function
# 初始化优化器和损失函数
base_optimizer = optim.Adam(model.parameters(), lr=0.0008)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.666)
criterion = nn.MSELoss()


# Define evaluation metrics function
# 定义评估指标函数
def evaluate_metrics(true_labels, pred_labels):
    """
    Calculate evaluation metrics for each target
    计算每个预测目标的评估指标
    
    Returns/返回:
        Dictionary containing metrics for each target
        包含每个目标指标的结果字典
    """
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    metrics = {}
    for i, label in enumerate(["Validly_Edited", "Unedited", "Erroneously_Edited"]):
        mse = mean_squared_error(true_labels[:, i], pred_labels[:, i])
        mae = mean_absolute_error(true_labels[:, i], pred_labels[:, i])
        r2 = r2_score(true_labels[:, i], pred_labels[:, i])
        spearman_corr = spearmanr(true_labels[:, i], pred_labels[:, i]).correlation
        
        metrics[label] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'Spearman': spearman_corr
        }
    
    return metrics


# Training loop
# 训练循环
epochs = 200
results = []  # Store training results/存储训练结果

for epoch in range(epochs):
    # Training phase
    # 训练阶段
    model.train()
    running_loss = 0.0
    for image, target in train_loader:
        image, target = image.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Validation phase
    # 验证阶段
    model.eval()
    val_loss = 0.0
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for image, target in val_loader:
            image, target = image.to(device), target.to(device)
            output = model(image).squeeze()
            loss = criterion(output, target)
            val_loss += loss.item()
            true_labels.extend(target.cpu().numpy())
            pred_labels.extend(output.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_metrics = evaluate_metrics(true_labels, pred_labels)
    
    # Print validation metrics
    # 打印验证指标
    print(f"Validation Metrics for Epoch {epoch + 1}:")
    for key, metric in val_metrics.items():
        print(f"{key} - MSE: {metric['MSE']:.4f}, MAE: {metric['MAE']:.4f}, R^2: {metric['R2']:.4f}, Spearman: {metric['Spearman']:.4f}")

    # Save results
    # 保存结果
    results.append({
        'Epoch': epoch + 1,
        'avg_loss': avg_loss,
        'avg_val_loss': avg_val_loss,
        'val_Validly_Edited_MSE': val_metrics["Validly_Edited"]['MSE'],
        'val_Validly_Edited_MAE': val_metrics["Validly_Edited"]['MAE'],
        'val_Validly_Edited_R2': val_metrics["Validly_Edited"]['R2'],
        'val_Validly_Edited_Spearman': val_metrics["Validly_Edited"]['Spearman'],
        'val_Unedited_MSE': val_metrics["Unedited"]['MSE'],
        'val_Unedited_MAE': val_metrics["Unedited"]['MAE'],
        'val_Unedited_R2': val_metrics["Unedited"]['R2'],
        'val_Unedited_Spearman': val_metrics["Unedited"]['Spearman'],
        'val_Erroneously_Edited_MSE': val_metrics["Erroneously_Edited"]['MSE'],
        'val_Erroneously_Edited_MAE': val_metrics["Erroneously_Edited"]['MAE'],
        'val_Erroneously_Edited_R2': val_metrics["Erroneously_Edited"]['R2'],
        'val_Erroneously_Edited_Spearman': val_metrics["Erroneously_Edited"]['Spearman'],
    })

# Save results to CSV
# 保存结果到CSV文件
results_df = pd.DataFrame(results)
results_df.to_csv("PrimeNet/PrimeNet.csv", index=False)

# Save trained model
# 保存训练好的模型
torch.save(model.state_dict(), "PrimeNet/PrimeNet1.pth")
print(f"Model saved to PrimeNet/PrimeNet1.pth")