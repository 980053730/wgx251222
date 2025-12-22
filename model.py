import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationClassifier(nn.Module):
    """
    旋转角度分类器
    PRIORITY 1/2 UPDATE:
    - 现在接收高维卷积特征 (feature_dim) 作为输入, 而不是 20 维的 z。
    """

    def __init__(self, feature_dim=400):
        super(RotationClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 4)  # 4类: 0°, 90°, 180°, 270°
        )

    def forward(self, features):
        return self.fc(features)


class InpaintingHead(nn.Module):
    """图像修复头 (未修改)"""

    def __init__(self):
        super(InpaintingHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 上采样到28x28
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """
    编码器
    PRIORITY 1/2 UPDATE:
    - 架构从 FC (全连接) 替换为 CNN (卷积), 以保留空间信息。
    - SSL 任务解耦:
        - 旋转预测 (Rotation) 作用于卷积特征 h1 (400维)。
        - 对比学习 (Contrastive) 作用于潜在向量 z (20维)。
    """

    def __init__(self, latent_dim=20, conv_feature_dim=400):
        super(Encoder, self).__init__()

        # 卷积层，提取空间特征
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),  # (B, 1, 28, 28) -> (B, 32, 28, 28)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> (B, 32, 14, 14)
            nn.Conv2d(32, 64, 5, padding=2),  # -> (B, 64, 14, 14)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # -> (B, 64, 7, 7)
        )

        # 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, conv_feature_dim)
        self.fc21 = nn.Linear(conv_feature_dim, latent_dim)  # mean
        self.fc22 = nn.Linear(conv_feature_dim, latent_dim)  # logvar

        # 自监督模块 (已解耦)
        self.rotation_classifier = RotationClassifier(feature_dim=conv_feature_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64)  # 对比学习投影
        )

    def encode(self, x_img):
        """输入 x_img 必须是 (B, 1, 28, 28)"""
        features = self.conv(x_img)
        flattened = features.view(features.size(0), -1)
        h1 = F.relu(self.fc1(flattened))
        return self.fc21(h1), self.fc22(h1), h1  # 返回 h1 供旋转分类器使用

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, rotation_label=None):
        """输入 x 必须是 (B, 1, 28, 28)"""
        mu, logvar, h1 = self.encode(x)
        z = self.reparametrize(mu, logvar)  # z 是 (B, latent_dim)

        # 计算KL散度
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        # 自监督输出 (已解耦)
        ssl_output = {}

        # 旋转预测: 作用于卷积特征 h1
        if rotation_label is not None:
            rotation_pred = self.rotation_classifier(h1)
            ssl_output['rotation_pred'] = rotation_pred
            ssl_output['rotation_label'] = rotation_label

        # 对比学习投影: 作用于潜在向量 z
        ssl_output['projection'] = self.projection_head(z)

        return z, KLD, ssl_output


class Decoder(nn.Module):
    """解码器/生成器 (未修改)"""

    def __init__(self, noise_dim=20):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(True),
            nn.BatchNorm1d(7 * 7 * 128)
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
            nn.Tanh()
        )

        # 图像修复头
        self.inpainting_head = InpaintingHead()

    def forward(self, x, mask=None):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7)

        # 获取中间特征图（64通道）
        for i, layer in enumerate(self.conv):
            x = layer(x)
            if i == 0:  # 第一个转置卷积层后保存特征图
                features = x

        # 主输出
        output = x

        # 自监督输出
        ssl_output = {}

        # 图像修复任务
        if mask is not None:
            inpainting_pred = self.inpainting_head(features)
            ssl_output['inpainting_pred'] = inpainting_pred
            ssl_output['mask'] = mask

        return output, ssl_output


class Discriminator(nn.Module):
    """判别器（带分类功能）"""

    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # 真伪判别头
        # PRIORITY 1 UPDATE: 移除 nn.Sigmoid() 以修复 LSGAN 损失
        self.validity_head = nn.Sequential(
            nn.Linear(1024, 1)
            # nn.Sigmoid() # <-- 已移除
        )

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(64, 1, 4, 2, 1),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        features = self.conv(x)
        flattened = features.view(features.size(0), -1)

        # 分类输出
        class_logits = self.classifier(flattened)

        # 真伪判别输出
        validity = self.validity_head(flattened)

        # 特征图
        feature_map = self.feature_extractor(features).squeeze()

        return validity, class_logits, feature_map

