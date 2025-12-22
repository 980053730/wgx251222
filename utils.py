import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 解决在无GUI服务器上绘图的问题
import matplotlib.pyplot as plt
import random
import os

from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.transforms import functional as TF  # 导入 functional
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn as nn

# PRIORITY 3: 导入新评估所需的库
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# --- 新增：导入混淆矩阵和 Seaborn ---
from sklearn.metrics import confusion_matrix
import seaborn as sns


def self_supervised_augmentation(images):
    """
    应用自监督增强：
    1. 随机旋转 (0°, 90°, 180°, 270°)
    2. 随机遮挡
    3. 颜色抖动
    """
    batch_size = images.size(0)
    device = images.device

    # 1. 随机旋转 (UPDATE: 使用 TF.rotate 替代手动操作)
    rotations = torch.randint(0, 4, (batch_size,))
    rotated_images = torch.zeros_like(images)
    rotation_labels = rotations.clone()

    for i, rot in enumerate(rotations):
        angle = int(rot * 90)
        if angle > 0:
            rotated_images[i] = TF.rotate(images[i], float(angle))
        else:
            rotated_images[i] = images[i]

    # 2. 随机遮挡
    masked_images = rotated_images.clone()
    masks = torch.zeros((batch_size, 1, images.size(2), images.size(3)), device=device)

    for i in range(batch_size):
        # 随机遮挡位置和大小
        mask_size = random.randint(8, 16)
        x = random.randint(0, images.size(2) - mask_size)
        y = random.randint(0, images.size(3) - mask_size)

        # 应用遮挡
        masked_images[i, :, x:x + mask_size, y:y + mask_size] = 0
        masks[i, 0, x:x + mask_size, y:y + mask_size] = 1

    # 3. 轻微颜色抖动 (对 MNIST 作用不大, 但保留)
    color_jitter = torch.rand(batch_size, 1, 1, 1, device=device) * 0.1 - 0.05
    augmented_images = torch.clamp(rotated_images + color_jitter, -1, 1)

    return {
        'original': images,
        'augmented': augmented_images,
        'masked': masked_images,
        'rotation_labels': rotation_labels.to(device),
        'masks': masks
    }


class SelfSupervisedLoss(nn.Module):
    """自监督损失函数 (未修改)"""

    def __init__(self, lambda_rot=1.0, lambda_cont=0.5, lambda_inp=0.7):
        super(SelfSupervisedLoss, self).__init__()
        self.lambda_rot = lambda_rot
        self.lambda_cont = lambda_cont
        self.lambda_inp = lambda_inp
        self.rotation_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = nn.CosineEmbeddingLoss()
        self.inpainting_loss = nn.BCELoss()  # 使用BCELoss

    def forward(self, outputs, targets):
        loss = 0

        # 旋转预测损失
        if 'rotation_pred' in outputs and 'rotation_label' in outputs:
            rot_loss = self.rotation_loss(
                outputs['rotation_pred'],
                outputs['rotation_label']
            )
            loss += self.lambda_rot * rot_loss

        # 对比学习损失
        if 'projection_orig' in outputs and 'projection_aug' in outputs:
            # 目标标签全为1（表示相似）
            labels = torch.ones(outputs['projection_orig'].size(0),
                                device=outputs['projection_orig'].device)
            cont_loss = self.contrastive_loss(
                outputs['projection_orig'],
                outputs['projection_aug'],
                labels
            )
            loss += self.lambda_cont * cont_loss

        # 图像修复损失
        if 'inpainting_pred' in outputs and 'mask' in outputs:
            # 只计算被遮挡区域的损失
            masked_pred = outputs['inpainting_pred'] * outputs['mask']

            # 目标是原始图像 (在 [-1, 1] 范围), InpaintingHead 输出是 [0, 1]
            # 我们需要将目标也映射到 [0, 1]
            normalized_target = (targets['original'] + 1) / 2.0
            masked_target = normalized_target * outputs['mask']

            inp_loss = self.inpainting_loss(masked_pred, masked_target)
            loss += self.lambda_inp * inp_loss

        return loss


def load_mnist_data(batch_size=128, data_dir='./mnist'):
    """加载MNIST数据集 (未修改)"""
    # 数据处理
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
    ])

    train_set = MNIST(data_dir, download=True, train=True, transform=im_tfs)
    test_set = MNIST(data_dir, download=True, train=False, transform=im_tfs)

    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_data = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_data, test_data


def evaluate_classification(model, data_loader, device):
    """评估分类准确率 (用于 Discriminator) (未修改)"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 获取分类结果
            _, logits, _ = model(images)
            _, predicted = torch.max(logits.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# --- 新增：获取所有预测结果 ---
def get_all_predictions(model, data_loader, device):
    """获取模型在给定数据加载器上的所有预测和标签"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 获取分类结果
            _, logits, _ = model(images)
            _, predicted = torch.max(logits.data, 1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 将列表展平为单个数组
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_preds, all_labels


# --- PRIORITY 3: 新增评估函数 ---

def linear_probe_test(encoder, train_loader, test_loader, device):
    """
    线性可分性测试 (Linear Probing)
    评估 Encoder 学到的特征 z 的质量
    """
    encoder.eval()

    # 1. 提取训练集特征
    z_train_list, y_train_list = [], []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            z, _, _ = encoder(x)  # Encoder 现在接收 4D 张量
            z_train_list.append(z.cpu())
            y_train_list.append(y.cpu())

    z_train = torch.cat(z_train_list, dim=0).numpy()
    y_train = torch.cat(y_train_list, dim=0).numpy()

    # 2. 提取测试集特征
    z_test_list, y_test_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            z, _, _ = encoder(x)
            z_test_list.append(z.cpu())
            y_test_list.append(y.cpu())

    z_test = torch.cat(z_test_list, dim=0).numpy()
    y_test = torch.cat(y_test_list, dim=0).numpy()

    # 3. 特征标准化
    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_train)
    z_test = scaler.transform(z_test)

    # 4. 训练线性分类器
    try:
        # 增加迭代次数和容忍度, 解决 'lbfgs failed to converge'
        clf = LogisticRegression(solver='lbfgs', max_iter=2000, multi_class='multinomial', tol=0.01)
        clf.fit(z_train, y_train)
    except Exception as e:
        print(f"Warning: Logistic Regression (lbfgs) failed ({e}). Using simpler solver.")
        clf = LogisticRegression(solver='liblinear', max_iter=500, multi_class='auto')  # liblinear 通常更稳定
        clf.fit(z_train, y_train)

    # 5. 评估
    y_pred = clf.predict(z_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy * 100


def visualize_latent_space(encoder, data_loader, device, save_path):
    """
    使用 t-SNE 可视化潜在空间 z
    """
    encoder.eval()
    z_list, y_list = [], []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            z, _, _ = encoder(x)  # Encoder 现在接收 4D 张量
            z_list.append(z.cpu())
            y_list.append(y.cpu())

    z_all = torch.cat(z_list, dim=0).numpy()
    y_all = torch.cat(y_list, dim=0).numpy()

    # 限制样本数量, t-SNE 对大数据量很慢
    if len(z_all) > 2000:
        indices = np.random.choice(len(z_all), 2000, replace=False)
        z_all = z_all[indices]
        y_all = y_all[indices]

    try:
        # 使用 PCA 初始化, 更快更稳定
        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', n_iter=1000)
        z_2d = tsne.fit_transform(z_all)

        plt.figure(figsize=(12, 10))
        # 使用 10 个离散的颜色
        cmap = plt.cm.get_cmap("tab10", 10)
        scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_all, cmap=cmap, s=10)
        plt.colorbar(scatter, ticks=range(10))
        plt.clim(-0.5, 9.5)
        plt.title("t-SNE visualization of latent space (z)")
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not generate t-SNE plot: {e}")
        plt.close()


# --- 新增：绘制混淆矩阵 ---
def plot_confusion_matrix(y_true, y_pred, save_path, class_names):
    """绘制并保存混淆矩阵"""
    try:
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix (Discriminator)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"绘制混淆矩阵失败: {e}")
        plt.close()


# --- 绘图函数 (已修改) ---

def show_images(images, title=None, save_path=None):
    """可视化图像网格 (修改: 调整归一化范围)"""
    images = images.detach().cpu()
    # 归一化范围从 [-1, 1] 映射到 [0, 1]
    grid = make_grid(images, nrow=8, normalize=True, value_range=(-1, 1), pad_value=0.5)
    np_grid = grid.numpy().transpose((1, 2, 0))

    plt.figure(figsize=(10, 10))
    plt.imshow(np_grid, interpolation='nearest')
    plt.axis('off')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_losses(losses_dict, save_path=None):
    """绘制损失曲线 (未修改)"""
    plt.figure(figsize=(12, 8))
    for name, values in losses_dict.items():
        plt.plot(values, label=name)
    plt.title("Training Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_accuracy(train_acc, test_acc, linear_probe_acc=None, save_path=None):
    """
    绘制准确率曲线
    PRIORITY 3 UPDATE: 增加 linear_probe_acc
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Train Accuracy (Discriminator)')
    plt.plot(test_acc, label='Test Accuracy (Discriminator)')

    if linear_probe_acc and len(linear_probe_acc) > 0:
        plt.plot(linear_probe_acc, label='Linear Probe Accuracy (Encoder)', linestyle='--', marker='o')

    plt.title("Classification Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

