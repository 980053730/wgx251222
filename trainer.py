import torch
import torch.optim as optim
from torch import nn
import time
import os
# 假设 model.py 和 utils.py 在同一个目录或在 python 路径上
# --- 修改：移除相对导入 (.), 确保 main.py 可以运行 ---
from model import Encoder, Decoder, Discriminator
from utils import (self_supervised_augmentation, SelfSupervisedLoss,
                   evaluate_classification, show_images, plot_losses,
                   plot_accuracy, visualize_latent_space, linear_probe_test,
                   get_all_predictions, plot_confusion_matrix)  # <-- 导入新函数


class VAEGANTrainer:
    def __init__(self, device='cuda', latent_dim=20, num_classes=10,
                 lr_e=1e-3, lr_g=3e-4, lr_d=3e-4,
                 lambda_kld=0.1, kld_anneal_max_epochs=20,
                 lambda_recon=10.0, lambda_feat=0.5,
                 ssl_lambda_rot=1.0, ssl_lambda_cont=0.5, ssl_lambda_inp=0.7,
                 save_dir="results"):  # <-- 接受 save_dir 和超参数

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 初始化模型
        # PRIORITY 1: Encoder() 现在会创建新的卷积 Encoder
        self.encoder = Encoder(latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim).to(self.device)
        # PRIORITY 1: Discriminator() 现在创建不带 Sigmoid 的版本
        self.discriminator = Discriminator(num_classes).to(self.device)

        # 优化器
        self.E_optim = optim.Adam(self.encoder.parameters(), lr=lr_e)
        self.G_optim = optim.Adam(self.decoder.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.D_optim = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

        # 损失函数
        self.ssl_criterion = SelfSupervisedLoss(
            lambda_rot=ssl_lambda_rot,
            lambda_cont=ssl_lambda_cont,
            lambda_inp=ssl_lambda_inp
        )
        self.cls_criterion = nn.CrossEntropyLoss()

        # PRIORITY 2: KLD 退火参数
        self.current_epoch = 0
        self.kld_anneal_max_epochs = kld_anneal_max_epochs

        # 损失权重 (模块化, 易于调整)
        self.lambda_kld = lambda_kld
        self.lambda_recon = lambda_recon
        self.lambda_feat = lambda_feat

        # 训练状态
        self.losses = {
            'D_loss': [], 'G_loss': [], 'KLD': [],
            'SSL_loss': [], 'Recon_loss': [], 'Cls_loss': []
        }
        self.train_accuracies = []
        self.test_accuracies = []
        # PRIORITY 3: 新的评估列表
        self.linear_probe_accuracies = []
        # --- 新增：用于保存分类结果txt ---
        self.classification_results = []

        # 固定噪声用于生成样本
        self.fixed_noise = (torch.rand(64, latent_dim) - 0.5) / 0.5
        self.fixed_noise = self.fixed_noise.to(self.device)

        # --- 修改：使用传入的 save_dir ---
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)  # (添加 exist_ok=True 确保安全)

    def train_step(self, x_real, labels):
        """执行单批次训练"""
        batch_size = x_real.size(0)
        # x_real 已经是 (B, 1, 28, 28)

        # 应用自监督增强
        aug_data = self_supervised_augmentation(x_real)
        x_aug = aug_data['augmented']
        x_masked = aug_data['masked']
        rotation_labels = aug_data['rotation_labels']
        masks = aug_data['masks']

        # ===== 训练判别器 =====
        self.D_optim.zero_grad()

        # 真实图像
        real_validity, real_logits, _ = self.discriminator(x_real)

        # 重建图像
        # PRIORITY 1 UPDATE: 不再 .view(), 直接传入 4D 张量
        z_real, _, _ = self.encoder(x_real)
        x_recon, _ = self.decoder(z_real)
        recon_validity, recon_logits, _ = self.discriminator(x_recon.detach())

        # 随机生成图像
        noise = (torch.rand(batch_size, 20) - 0.5) / 0.5
        noise = noise.to(self.device)
        x_rand, _ = self.decoder(noise)
        rand_validity, rand_logits, _ = self.discriminator(x_rand.detach())

        # 判别器损失（LSGAN 损失）
        D_loss_validity = 0.5 * ((real_validity - 1) ** 2).mean() + \
                          0.5 * (recon_validity ** 2).mean() + \
                          0.5 * (rand_validity ** 2).mean()

        # 分类损失（仅真实图像）
        D_loss_cls = self.cls_criterion(real_logits, labels)
        D_loss = D_loss_validity + D_loss_cls

        D_loss.backward()
        self.D_optim.step()

        # ===== 训练生成器和编码器 =====
        self.E_optim.zero_grad()
        self.G_optim.zero_grad()

        # PRIORITY 1/2 UPDATE:
        # - 传入 4D 张量 (x_real)
        # - 传入 rotation_labels 以便在 h1 上进行旋转预测
        z_real, KLD, ssl_output_orig = self.encoder(
            x_real,
            rotation_labels
        )
        x_recon, _ = self.decoder(z_real)

        # 增强图像路径 (4D)
        z_aug, _, ssl_output_aug = self.encoder(x_aug)

        # 图像修复路径 (4D)
        z_masked, _, _ = self.encoder(x_masked)
        x_repaired, ssl_output_rep = self.decoder(z_masked, masks)

        # 随机生成路径
        x_rand, _ = self.decoder(noise)
        rand_validity, _, rand_features = self.discriminator(x_rand)

        # 合并自监督输出
        ssl_output = {
            **ssl_output_orig,
            'projection_orig': ssl_output_orig['projection'],
            'projection_aug': ssl_output_aug['projection'],
            **ssl_output_rep
        }

        # 计算自监督损失
        ssl_loss = self.ssl_criterion(ssl_output, {'original': x_real})

        # 重建损失 (L1)
        recon_loss = nn.L1Loss()(x_recon, x_real)

        # 特征匹配损失
        _, _, real_features = self.discriminator(x_real)
        feature_loss = nn.MSELoss()(rand_features, real_features.detach())

        # GAN损失
        G_loss = 0.5 * ((rand_validity - 1) ** 2).mean()

        # PRIORITY 2 UPDATE: KLD 退火
        kld_weight_annealed = min(1.0, self.current_epoch / self.kld_anneal_max_epochs) * self.lambda_kld

        # 组合损失 (使用权重属性)
        total_loss = (
                kld_weight_annealed * KLD +
                self.lambda_recon * recon_loss +
                self.lambda_feat * feature_loss +
                G_loss +
                ssl_loss
        )

        total_loss.backward()
        self.E_optim.step()
        self.G_optim.step()

        # 返回损失值
        return {
            'D_loss': D_loss.item(),
            'G_loss': G_loss.item(),
            'KLD': KLD.item() * kld_weight_annealed,  # 记录加权后的KLD
            'SSL_loss': ssl_loss.item(),
            'Recon_loss': recon_loss.item(),
            'Cls_loss': D_loss_cls.item()
        }

    def train_epoch(self, data_loader, test_loader=None):
        """训练一个epoch"""
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        epoch_losses = {k: 0.0 for k in self.losses.keys()}
        num_batches = len(data_loader)

        start_time = time.time()
        for batch_idx, (x_real, labels) in enumerate(data_loader):
            x_real = x_real.to(self.device)  # 已经是 (B, 1, 28, 28)
            labels = labels.to(self.device)

            losses = self.train_step(x_real, labels)

            # 累计损失
            for k in epoch_losses:
                epoch_losses[k] += losses[k]

        # 计算平均损失
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            self.losses[k].append(epoch_losses[k])

        # 评估分类性能 (Discriminator)
        train_acc = evaluate_classification(self.discriminator, data_loader, self.device)
        self.train_accuracies.append(train_acc)

        if test_loader:
            test_acc = evaluate_classification(self.discriminator, test_loader, self.device)
            self.test_accuracies.append(test_acc)

            # PRIORITY 3: 执行新的评估
            # 1. Linear Probe
            linear_acc = linear_probe_test(self.encoder, data_loader, test_loader, self.device)
            self.linear_probe_accuracies.append(linear_acc)

            # 2. t-SNE 可视化
            tsne_save_path = os.path.join(self.save_dir, f"tsne_epoch_{len(self.losses['D_loss'])}.png")
            visualize_latent_space(self.encoder, test_loader, self.device, tsne_save_path)

        else:
            test_acc = 0.0
            linear_acc = 0.0

        # 生成样本
        with torch.no_grad():
            samples, _ = self.decoder(self.fixed_noise)
            show_images(samples, title=f"Generated Samples - Epoch {len(self.losses['D_loss'])}",
                        save_path=os.path.join(self.save_dir, f"gen_epoch_{len(self.losses['D_loss'])}.png"))

            # 重建样本
            test_iter = iter(test_loader)
            test_batch, _ = next(test_iter)
            test_batch = test_batch[:8].to(self.device)
            # PRIORITY 1: 传入 4D 张量
            z_test, _, _ = self.encoder(test_batch)
            recon_test, _ = self.decoder(z_test)

            # 可视化重建结果
            comparison = torch.cat([test_batch, recon_test])
            show_images(comparison, title=f"Reconstructions - Epoch {len(self.losses['D_loss'])}",
                        save_path=os.path.join(self.save_dir, f"recon_epoch_{len(self.losses['D_loss'])}.png"))

        epoch_time = time.time() - start_time

        # --- 修改：捕获打印的日志 ---
        epoch_summary_line = (
            f"Epoch {len(self.losses['D_loss'])} | Time: {epoch_time:.2f}s | "
            f"D Loss: {epoch_losses['D_loss']:.4f} | G Loss: {epoch_losses['G_loss']:.4f} | "
            f"KLD: {epoch_losses['KLD']:.4f} | SSL Loss: {epoch_losses['SSL_loss']:.4f} | "
            f"Recon: {epoch_losses['Recon_loss']:.4f} | Cls Loss: {epoch_losses['Cls_loss']:.4f} | "
            f"Train Acc (D): {train_acc:.2f}% | Test Acc (D): {test_acc:.2f}% | "
            f"Linear Probe (E): {linear_acc:.2f}%"
        )
        print(epoch_summary_line)  # 仍然打印到控制台
        self.classification_results.append(epoch_summary_line)  # 保存到列表

        # PRIORITY 2: KLD 退火
        self.current_epoch += 1

        return epoch_losses

    def save_model(self, path, epoch=None):
        """保存模型状态"""
        state = {
            'epoch': epoch if epoch else len(self.losses['D_loss']),
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'E_optimizer_state_dict': self.E_optim.state_dict(),
            'G_optimizer_state_dict': self.G_optim.state_dict(),
            'D_optimizer_state_dict': self.D_optim.state_dict(),
            'losses': self.losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'linear_probe_accuracies': self.linear_probe_accuracies,  # PRIORITY 3
            'classification_results': self.classification_results  # --- 新增 ---
        }
        torch.save(state, path)

    def load_model(self, path):
        """加载模型状态"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.E_optim.load_state_dict(checkpoint['E_optimizer_state_dict'])
        self.G_optim.load_state_dict(checkpoint['G_optimizer_state_dict'])
        self.D_optim.load_state_dict(checkpoint['D_optimizer_state_dict'])
        self.losses = checkpoint['losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.test_accuracies = checkpoint['test_accuracies']
        # PRIORITY 3
        self.linear_probe_accuracies = checkpoint.get('linear_probe_accuracies', [])
        # --- 新增：加载分类结果 (如果存在) ---
        self.classification_results = checkpoint.get('classification_results', [])
        self.current_epoch = checkpoint.get('epoch', 0)

    def visualize_training(self, test_loader=None):  # <-- 修改：接收 test_loader
        """可视化训练结果"""
        # 绘制损失曲线
        loss_path = os.path.join(self.save_dir, 'losses.png')
        plot_losses(self.losses, save_path=loss_path)

        # 绘制准确率曲线
        if self.test_accuracies:
            acc_path = os.path.join(self.save_dir, 'accuracy.png')
            # PRIORITY 3: 传入新指标
            plot_accuracy(
                self.train_accuracies,
                self.test_accuracies,
                self.linear_probe_accuracies,
                save_path=acc_path
            )

        # 生成最终样本
        samples, _ = self.decoder(self.fixed_noise)
        sample_path = os.path.join(self.save_dir, 'final_samples.png')
        show_images(samples, title="Final Generated Samples", save_path=sample_path)

        # --- 新增：保存分类结果 TXT ---
        results_save_path = os.path.join(self.save_dir, 'classification_results.txt')
        print(f"保存分类日志到: {results_save_path}...")
        try:
            with open(results_save_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.classification_results))
        except Exception as e:
            print(f"保存分类日志失败: {e}")

        # --- 新增：生成并保存混淆矩阵 ---
        if test_loader:
            print("生成 (Discriminator) 混淆矩阵...")
            try:
                # 确保模型在评估模式
                self.discriminator.eval()
                y_pred, y_true = get_all_predictions(self.discriminator, test_loader, self.device)

                cm_save_path = os.path.join(self.save_dir, 'confusion_matrix_discriminator.png')
                class_names = [str(i) for i in range(10)]  # MNIST 类别 0-9

                plot_confusion_matrix(y_true, y_pred, cm_save_path, class_names)
                print(f"混淆矩阵已保存到: {cm_save_path}")

            except Exception as e:
                print(f"生成混淆矩阵失败: {e}")

