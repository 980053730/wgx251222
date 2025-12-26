import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationClassifier(nn.Module):
    """
    旋转角度分类器
    接收 Encoder 提取的深层卷积特征
    """

    def __init__(self, feature_dim=512):  # 特征维度增加到 512
        super(RotationClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 4)  # 0°, 90°, 180°, 270°
        )

    def forward(self, features):
        return self.fc(features)


class InpaintingHead(nn.Module):
    """
    图像修复头
    适配高分辨率: 接收解码器倒数第二层的特征 (例如 112x112 或 224x224 的特征图)
    输出单通道掩码预测 (Mask Prediction)
    """

    def __init__(self, input_channels=64):
        super(InpaintingHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """
    高分辨率 RGB 编码器
    输入: (B, 3, 224, 224)
    """

    def __init__(self, latent_dim=20, input_channels=3):
        super(Encoder, self).__init__()

        # 下采样模块: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.conv = nn.Sequential(
            # Layer 1: 224 -> 112
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 112 -> 56
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 56 -> 28
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 28 -> 14
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: 14 -> 7
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 自适应池化: 无论前面卷积输出多大，这里强制变成 1x1 或 4x4
        # 这里我们使用 Global Average Pooling (1x1) 得到 512 维特征
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        feature_dim = 512

        # 潜在变量均值和方差
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)

        # 自监督模块
        self.rotation_classifier = RotationClassifier(feature_dim=feature_dim)

        # 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64)
        )

    def encode(self, x):
        # 卷积提取特征
        conv_feat = self.conv(x)  # (B, 512, 7, 7)

        # 全局池化并展平
        pooled = self.global_pool(conv_feat)  # (B, 512, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)  # (B, 512)

        return self.fc_mu(flattened), self.fc_logvar(flattened), flattened

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, rotation_label=None):
        mu, logvar, h1 = self.encode(x)
        z = self.reparametrize(mu, logvar)

        # KL散度
        batch_size = x.size(0)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD / batch_size

        ssl_output = {}

        # 旋转预测 (使用卷积后的深层特征 h1)
        if rotation_label is not None:
            rotation_pred = self.rotation_classifier(h1)
            ssl_output['rotation_pred'] = rotation_pred
            ssl_output['rotation_label'] = rotation_label

        # 对比学习投影 (使用潜在变量 z)
        ssl_output['projection'] = self.projection_head(z)

        return z, KLD, ssl_output


class Decoder(nn.Module):
    """
    高分辨率 RGB 解码器
    将 latent_dim (z) 还原为 (B, 3, 224, 224)
    """

    def __init__(self, latent_dim=20, output_channels=3):
        super(Decoder, self).__init__()

        # 将 z 映射回初始特征图大小: 512通道 * 7 * 7
        self.fc_init_size = 7
        self.fc_channels = 512
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.fc_channels * self.fc_init_size * self.fc_init_size),
            nn.ReLU(True)
        )

        # 上采样模块: 7 -> 14 -> 28 -> 56 -> 112 -> 224
        self.upsample_blocks = nn.ModuleList([
            # Block 1: 7 -> 14
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            ),
            # Block 2: 14 -> 28
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            ),
            # Block 3: 28 -> 56
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ),
            # Block 4: 56 -> 112
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(True)
            ),
            # Block 5: 112 -> 224 (Output Layer)
            nn.Sequential(
                nn.ConvTranspose2d(32, output_channels, 4, 2, 1),
                nn.Tanh()  # 输出范围 [-1, 1]
            )
        ])

        # 图像修复头: 接收 Block 4 的输出 (32通道, 112x112)
        # 上采样一次到 224x224
        self.inpainting_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            InpaintingHead(input_channels=32)
        )

    def forward(self, z, mask=None):
        # FC -> Reshape
        out = self.fc(z)
        out = out.view(out.size(0), self.fc_channels, self.fc_init_size, self.fc_init_size)

        features_for_inpainting = None

        # 逐层上采样
        for i, block in enumerate(self.upsample_blocks):
            out = block(out)
            # 在倒数第二层 (Block 4) 后保存特征用于 Inpainting
            if i == 3:  # Index 3 is the 4th block (56->112)
                features_for_inpainting = out

        # 主输出 (图像)
        img_output = out

        ssl_output = {}
        if mask is not None and features_for_inpainting is not None:
            inp_pred = self.inpainting_head(features_for_inpainting)
            ssl_output['inpainting_pred'] = inp_pred
            ssl_output['mask'] = mask

        return img_output, ssl_output


class Discriminator(nn.Module):
    """
    高分辨率 RGB 判别器
    输入: (B, 3, 224, 224)
    """

    def __init__(self, num_classes=10, input_channels=3):
        super(Discriminator, self).__init__()

        # 特征提取
        self.conv = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 112 -> 56
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 56 -> 28
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 28 -> 14
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 14 -> 7
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 自适应池化 -> 变成 (B, 512, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat_dim = 512

        # 辅助分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # 真伪判别头 (LSGAN 不需要 Sigmoid)
        self.validity_head = nn.Sequential(
            nn.Linear(self.flat_dim, 1)
        )

    def forward(self, x):
        conv_out = self.conv(x)

        # 池化并展平
        pooled = self.global_pool(conv_out)
        flattened = pooled.view(pooled.size(0), -1)

        validity = self.validity_head(flattened)
        class_logits = self.classifier(flattened)

        # 返回最后一个卷积层的输出作为特征匹配 (Feature Matching) 用
        # 我们用池化后的特征作为 feature map 的简化版
        return validity, class_logits, flattened