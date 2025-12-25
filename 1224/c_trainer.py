import torch
import torch.optim as optim
from torch import nn
import time
import os
import csv
from c_model import Encoder, Decoder, Discriminator
from c_utils import (self_supervised_augmentation, SelfSupervisedLoss,
                   evaluate_classification, show_images, plot_losses,
                   plot_accuracy, visualize_latent_space, get_all_predictions,
                   plot_confusion_matrix)


class VAEGANTrainer:
    def __init__(self, device='cuda', latent_dim=20, num_classes=10,
                 lr_e=1e-3, lr_g=3e-4, lr_d=3e-4,
                 lambda_kld=0.1, kld_anneal_max_epochs=20,
                 lambda_recon=10.0, lambda_feat=0.5,
                 ssl_lambda_rot=1.0, ssl_lambda_cont=0.5, ssl_lambda_inp=0.7,
                 save_dir="results",
                 # 新增参数
                 img_size=224, input_channels=3):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.input_channels = input_channels
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        print(f"初始化 VAE-GAN: 图像尺寸={img_size}x{img_size}, 通道数={input_channels}")

        # 初始化模型 (传递 input_channels)
        self.encoder = Encoder(latent_dim, input_channels).to(self.device)
        self.decoder = Decoder(latent_dim, output_channels=input_channels).to(self.device)
        self.discriminator = Discriminator(num_classes, input_channels).to(self.device)

        # 优化器
        self.E_optim = optim.Adam(self.encoder.parameters(), lr=lr_e)
        self.G_optim = optim.Adam(self.decoder.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.D_optim = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

        # 损失函数
        self.ssl_criterion = SelfSupervisedLoss(ssl_lambda_rot, ssl_lambda_cont, ssl_lambda_inp)
        self.cls_criterion = nn.CrossEntropyLoss()

        # 参数记录
        self.lambda_kld = lambda_kld
        self.kld_anneal_max_epochs = kld_anneal_max_epochs
        self.lambda_recon = lambda_recon
        self.lambda_feat = lambda_feat
        self.current_epoch = 0

        # 记录
        self.losses = {'D_loss': [], 'G_loss': [], 'KLD': [], 'SSL_loss': [], 'Recon_loss': [], 'Cls_loss': []}
        self.train_accuracies = []
        self.test_accuracies = []
        self.classification_results = []

        # 固定噪声 (用于生成可视化)
        self.fixed_noise = torch.randn(64, latent_dim).to(self.device)

        def log_to_csv(self, epoch_losses, train_acc, test_acc):
            """
            将当前 Epoch 的数据追加写入 CSV 文件
            """
            csv_path = os.path.join(self.save_dir, 'training_log.csv')

            # 定义表头
            headers = ['Epoch', 'D_loss', 'G_loss', 'KLD', 'SSL_loss', 'Recon_loss', 'Cls_loss', 'Train_Acc','Test_Acc']
            # 准备数据行
            row = [
                self.current_epoch,
                f"{epoch_losses['D_loss']:.4f}",
                f"{epoch_losses['G_loss']:.4f}",
                f"{epoch_losses['KLD']:.4f}",
                f"{epoch_losses['SSL_loss']:.4f}",
                f"{epoch_losses['Recon_loss']:.4f}",
                f"{epoch_losses['Cls_loss']:.4f}",
                f"{train_acc:.2f}",
                f"{test_acc:.2f}"
            ]

            # 写入文件
            file_exists = os.path.isfile(csv_path)
            try:
                with open(csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(headers)  # 如果文件不存在，先写表头
                    writer.writerow(row)
            except Exception as e:
                print(f"写入 CSV 失败: {e}")

    def train_step(self, x_real, labels):
        batch_size = x_real.size(0)

        # 自监督增强
        aug_data = self_supervised_augmentation(x_real)
        x_aug = aug_data['augmented']
        x_masked = aug_data['masked']
        rotation_labels = aug_data['rotation_labels']
        masks = aug_data['masks']

        # === 1. 判别器训练 ===
        self.D_optim.zero_grad()

        real_validity, real_logits, _ = self.discriminator(x_real)

        z_real, _, _ = self.encoder(x_real)
        x_recon, _ = self.decoder(z_real)
        recon_validity, _, _ = self.discriminator(x_recon.detach())

        noise = torch.randn(batch_size, z_real.size(1)).to(self.device)
        x_rand, _ = self.decoder(noise)
        rand_validity, _, _ = self.discriminator(x_rand.detach())

        # LSGAN Loss
        d_loss_val = 0.5 * ((real_validity - 1) ** 2).mean() + \
                     0.5 * (recon_validity ** 2).mean() + \
                     0.5 * (rand_validity ** 2).mean()
        d_loss_cls = self.cls_criterion(real_logits, labels)
        d_loss = d_loss_val + d_loss_cls
        d_loss.backward()
        self.D_optim.step()

        # === 2. 生成器与编码器训练 ===
        self.E_optim.zero_grad()
        self.G_optim.zero_grad()

        # 原始路径
        z_real, KLD, ssl_out_orig = self.encoder(x_real, rotation_labels)
        x_recon, _ = self.decoder(z_real)

        # 增强路径 (对比学习)
        z_aug, _, ssl_out_aug = self.encoder(x_aug)

        # 修复路径
        z_masked, _, _ = self.encoder(x_masked)
        x_repaired, ssl_out_rep = self.decoder(z_masked, masks)

        # 随机生成 (GAN欺骗)
        x_rand, _ = self.decoder(noise)
        rand_validity, _, rand_features = self.discriminator(x_rand)
        _, _, real_features = self.discriminator(x_real)

        # 汇总 SSL 输出
        ssl_output = {
            **ssl_out_orig,
            'projection_orig': ssl_out_orig['projection'],
            'projection_aug': ssl_out_aug['projection'],
            **ssl_out_rep
        }

        # 计算各种损失
        ssl_loss = self.ssl_criterion(ssl_output, {'original': x_real})
        recon_loss = nn.L1Loss()(x_recon, x_real)
        feature_loss = nn.MSELoss()(rand_features, real_features.detach())
        g_loss_adv = 0.5 * ((rand_validity - 1) ** 2).mean()

        # KLD 退火
        kld_weight = min(1.0, self.current_epoch / self.kld_anneal_max_epochs) * self.lambda_kld

        total_loss = kld_weight * KLD + \
                     self.lambda_recon * recon_loss + \
                     self.lambda_feat * feature_loss + \
                     g_loss_adv + ssl_loss

        total_loss.backward()
        self.E_optim.step()
        self.G_optim.step()

        return {
            'D_loss': d_loss.item(), 'G_loss': g_loss_adv.item(),
            'KLD': KLD.item() * kld_weight, 'SSL_loss': ssl_loss.item(),
            'Recon_loss': recon_loss.item(), 'Cls_loss': d_loss_cls.item()
        }

    def train_epoch(self, data_loader, test_loader=None):
        self.encoder.train();
        self.decoder.train();
        self.discriminator.train()
        epoch_losses = {k: 0.0 for k in self.losses.keys()}

        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            loss_dict = self.train_step(x, y)
            for k, v in loss_dict.items(): epoch_losses[k] += v

        for k in epoch_losses:
            epoch_losses[k] /= len(data_loader)
            self.losses[k].append(epoch_losses[k])

        train_acc = evaluate_classification(self.discriminator, data_loader, self.device)
        self.train_accuracies.append(train_acc)

        test_acc = 0.0
        if test_loader:
            test_acc = evaluate_classification(self.discriminator, test_loader, self.device)
            self.test_accuracies.append(test_acc)

        # 打印进度
        print(f"Epoch {len(self.losses['D_loss'])} | D Loss: {epoch_losses['D_loss']:.4f} | "
              f"Recon: {epoch_losses['Recon_loss']:.4f} | SSL: {epoch_losses['SSL_loss']:.4f} | "
              f"Acc: {train_acc:.2f}% / {test_acc:.2f}%")

        self.log_to_csv(epoch_losses, train_acc, test_acc)
        self.current_epoch += 1
        return epoch_losses

    # (save_model, load_model, visualize_training 保持基本不变，但注意 visualize_training 里路径问题)
    def save_model(self, path):
        state = {
            'epoch': self.current_epoch,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'losses': self.losses,
            'train_acc': self.train_accuracies,
            'test_acc': self.test_accuracies
        }
        torch.save(state, path)

    def visualize_training(self, test_loader=None):
        plot_losses(self.losses, os.path.join(self.save_dir, 'losses.png'))
        plot_accuracy(self.train_accuracies, self.test_accuracies, None, os.path.join(self.save_dir, 'acc.png'))

        # 最终生成
        with torch.no_grad():
            samples, _ = self.decoder(self.fixed_noise)
            show_images(samples, title="Final Samples", save_path=os.path.join(self.save_dir, 'samples.png'))

        if test_loader:
            # 混淆矩阵
            y_pred, y_true = get_all_predictions(self.discriminator, test_loader, self.device)
            # 假设类别名就是数字，如果是 ImageFolder 后面可以优化
            names = [str(i) for i in range(10)]
            plot_confusion_matrix(y_true, y_pred, os.path.join(self.save_dir, 'cm.png'), names)
            #t - SNE可视化(Encoder) - --
            print("正在生成 t-SNE 可视化 (这可能需要一点时间)...")
            tsne_path = os.path.join(self.save_dir, f'tsne_epoch_{self.current_epoch}.png')
            try:
                visualize_latent_space(self.encoder, test_loader, self.device, tsne_path)
                print(f"t-SNE 已保存到: {tsne_path}")
            except Exception as e:
                print(f"t-SNE 生成失败 (可能是数据量不足或缺少 sklearn): {e}")