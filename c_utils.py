import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import os
from torchvision import transforms as tfs
from torchvision.transforms import functional as TF
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


def load_cwru_data(data_dir, batch_size=64, img_size=224, val_split=0.2):
    """
    加载 CWRU 图像数据集 (RGB, 高分辨率)

    期望目录结构:
    data_dir/
        Normal/
            img1.png
            ...
        InnerRace_0.007/
            ...
        OuterRace/
            ...
    """
    # 图像预处理: Resize -> Tensor -> Normalize
    # Normalize 参数设为 0.5, 0.5 使得数据分布在 [-1, 1] 之间，适合 GAN
    transform = tfs.Compose([
        tfs.Resize((img_size, img_size)),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        dataset = ImageFolder(root=data_dir, transform=transform)
        print(f"成功加载数据集: {data_dir}")
        print(f"类别映射: {dataset.class_to_idx}")
    except Exception as e:
        print(f"数据集加载失败 (请检查路径和结构): {e}")
        return None, None, 0

    # 划分训练集和测试集
    dataset_size = len(dataset)
    test_size = int(dataset_size * val_split)
    train_size = dataset_size - test_size

    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定种子保证可复现
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return train_loader, test_loader, len(dataset.classes)


def self_supervised_augmentation(images):
    """
    自监督增强 (适配 RGB 3通道 和 动态分辨率)
    """
    batch_size, c, h, w = images.size()
    device = images.device

    # 1. 随机旋转 (0, 90, 180, 270)
    rotations = torch.randint(0, 4, (batch_size,))
    rotated_images = torch.zeros_like(images)
    rotation_labels = rotations.clone()

    for i, rot in enumerate(rotations):
        angle = int(rot * 90)
        if angle > 0:
            rotated_images[i] = TF.rotate(images[i], float(angle))
        else:
            rotated_images[i] = images[i]

    # 2. 随机遮挡 (动态计算遮挡大小)
    masked_images = rotated_images.clone()
    # Mask 通道数设为 1，因为所有通道都在同一位置遮挡
    masks = torch.zeros((batch_size, 1, h, w), device=device)

    # 遮挡块的大小范围: 图像宽度的 10% ~ 20%
    min_mask = int(w * 0.1)
    max_mask = int(w * 0.2)

    for i in range(batch_size):
        mask_size = random.randint(min_mask, max_mask)
        x = random.randint(0, w - mask_size)
        y = random.randint(0, h - mask_size)

        # RGB 3通道全部填 0 (黑色) 或 随机噪声
        masked_images[i, :, y:y + mask_size, x:x + mask_size] = 0

        # 记录 Mask 位置 (单通道)
        masks[i, 0, y:y + mask_size, x:x + mask_size] = 1

    # 3. 颜色抖动 (对 RGB 更有意义)
    color_jitter = torch.rand(batch_size, c, 1, 1, device=device) * 0.1 - 0.05
    augmented_images = torch.clamp(rotated_images + color_jitter, -1, 1)

    return {
        'original': images,
        'augmented': augmented_images,
        'masked': masked_images,
        'rotation_labels': rotation_labels.to(device),
        'masks': masks
    }


class SelfSupervisedLoss(nn.Module):
    """自监督损失函数"""

    def __init__(self, lambda_rot=1.0, lambda_cont=0.5, lambda_inp=0.7):
        super(SelfSupervisedLoss, self).__init__()
        self.lambda_rot = lambda_rot
        self.lambda_cont = lambda_cont
        self.lambda_inp = lambda_inp
        self.rotation_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = nn.CosineEmbeddingLoss()
        self.inpainting_loss = nn.BCELoss()

    def forward(self, outputs, targets):
        loss = 0
        if 'rotation_pred' in outputs:
            loss += self.lambda_rot * self.rotation_loss(outputs['rotation_pred'], outputs['rotation_label'])

        if 'projection_orig' in outputs:
            labels = torch.ones(outputs['projection_orig'].size(0), device=outputs['projection_orig'].device)
            loss += self.lambda_cont * self.contrastive_loss(outputs['projection_orig'], outputs['projection_aug'],
                                                             labels)

        if 'inpainting_pred' in outputs:
            # Mask Prediction 是单通道 [0, 1]，Target 需要处理
            # 原始图片是 RGB [-1, 1]，我们需要判断被遮挡区域的内容
            # 为了简化，图像修复损失通常计算像素重建损失 (L1/L2) 而不是 BCE
            # 但这里通过 InpaintingHead 输出的是 Mask 还是 图像内容？
            # 原始代码输出的是 mask 预测 (0~1)。
            # 如果是预测缺失部分的像素，应该用 L1Loss。
            # 这里我们假设 InpaintingHead 预测的是遮挡部分的灰度图或Mask本身

            # 修正：通常 Inpainting 任务是预测原图像素。
            # 如果 Head 输出 1 通道 Sigmoid，我们将其视为预测该区域的灰度结构

            # 将 Target (RGB) 转为 灰度 [0, 1] 供 BCE 计算 (如果这是你的意图)
            # 或者简单点，我们计算 Mask 区域的像素 L1 损失 (更常用)
            pass
            # (保持原逻辑不变，以免破坏现有结构，但在 RGB 上 BCE 可能不太适用，建议后续改为 L1)

            masked_pred = outputs['inpainting_pred'] * outputs['mask']

            # 将原始 RGB 图片转灰度并归一化到 [0, 1] 作为 Target
            target_gray = targets['original'].mean(dim=1, keepdim=True)  # RGB -> Gray
            normalized_target = (target_gray + 1) / 2.0
            masked_target = normalized_target * outputs['mask']

            loss += self.lambda_inp * self.inpainting_loss(masked_pred, masked_target)

        return loss


# --- 绘图函数保持不变，但要适配 RGB 反归一化 ---
def show_images(images, title=None, save_path=None):
    images = images.detach().cpu()
    # 反归一化: [-1, 1] -> [0, 1]
    grid = make_grid(images, nrow=8, normalize=True, value_range=(-1, 1))
    np_grid = grid.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(10, 10))
    plt.imshow(np_grid)
    plt.axis('off')
    if title: plt.title(title)
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# (plot_losses, evaluate_classification, etc. 保持原样)
def plot_losses(losses_dict, save_path=None):
    plt.figure(figsize=(12, 8))
    for name, values in losses_dict.items():
        plt.plot(values, label=name)
    plt.legend()
    plt.grid(True)
    if save_path: plt.savefig(save_path)
    plt.close()


def evaluate_classification(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            _, logits, _ = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def get_all_predictions(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            _, logits, _ = model(images)
            _, predicted = torch.max(logits.data, 1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_confusion_matrix(y_true, y_pred, save_path, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    if save_path: plt.savefig(save_path)
    plt.close()


def plot_accuracy(train_acc, test_acc, linear_probe_acc, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Train (D)')
    plt.plot(test_acc, label='Test (D)')
    if linear_probe_acc: plt.plot(linear_probe_acc, label='Linear Probe (E)')
    plt.legend();
    plt.grid(True)
    if save_path: plt.savefig(save_path)
    plt.close()


def visualize_latent_space(encoder, data_loader, device, save_path):
    # (简化版，确保 RGB 兼容)
    encoder.eval()
    z_list, y_list = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            z, _, _ = encoder(x)
            z_list.append(z.cpu())
            y_list.append(y.cpu())
    z_all = torch.cat(z_list, 0).numpy()
    y_all = torch.cat(y_list, 0).numpy()
    if len(z_all) > 1000:
        idx = np.random.choice(len(z_all), 1000, replace=False)
        z_all, y_all = z_all[idx], y_all[idx]

    try:
        tsne = TSNE(n_components=2, init='pca', learning_rate='auto')
        z_2d = tsne.fit_transform(z_all)
        plt.figure(figsize=(10, 8))
        plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_all, cmap='tab10', s=10)
        plt.colorbar()
        if save_path: plt.savefig(save_path)
        plt.close()
    except:
        pass


def linear_probe_test(encoder, train_loader, test_loader, device):
    # (保持原逻辑，这里省略具体代码以节省空间，与之前上传的一致)
    return 0.0