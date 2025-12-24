import torch
import os
from trainer import VAEGANTrainer
from utils import load_cwru_data

# --- 配置 ---
# 请将此路径改为你 ST.py 生成图像的根目录
# 例如: ./images_stft_style
DATA_DIR = './images_stft_style'

IMG_SIZE = 224
BATCH_SIZE = 32  # 224x224 显存占用大，建议减小 batch size
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 检查数据目录是否存在
if not os.path.exists(DATA_DIR):
    print(f"错误: 数据目录 '{DATA_DIR}' 不存在。")
    print("请先运行 ST.py 或 ST_claude.py 生成图像数据。")
    exit()


def main():
    print(f"--- 训练开始: RGB {IMG_SIZE}x{IMG_SIZE} ---")

    # 1. 加载 CWRU 数据
    train_loader, test_loader, num_classes = load_cwru_data(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE
    )

    if train_loader is None:
        return

    print(f"检测到类别数: {num_classes}")

    # 2. 初始化训练器
    save_dir = f"results_rgb_{IMG_SIZE}"

    trainer = VAEGANTrainer(
        device=DEVICE,
        latent_dim=64,  # 图像变大，潜在向量维度建议增加
        num_classes=num_classes,  # 自动适配类别数
        img_size=IMG_SIZE,
        input_channels=3,  # RGB
        save_dir=save_dir
    )

    # 3. 训练
    for epoch in range(EPOCHS):
        trainer.train_epoch(train_loader, test_loader)

        # 每10轮保存一次中间结果
        if (epoch + 1) % 10 == 0:
            trainer.visualize_training(test_loader)
            trainer.save_model(os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))

    print("训练结束。")


if __name__ == "__main__":
    main()