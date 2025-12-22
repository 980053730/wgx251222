import torch
from utils import load_mnist_data
from trainer import VAEGANTrainer
import os
import json  # 导入 json 用于保存超参数
from datetime import datetime

# --- 超参数 ---
# PRIORITY: 调低 EPOCHS 和 BATCH_SIZE 以便快速测试
# 原始值可能是: EPOCHS = 50, BATCH_SIZE = 128
EPOCHS = 10  # 总训练轮数 (设置低一点以便快速看到结果)
BATCH_SIZE = 64  # 批量大小
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
DATA_DIR = './mnist'  # 数据集下载路径

# --- VAEGANTrainer 的超参数 ---
# 将它们定义在这里，方便统一保存
trainer_config = {
    'latent_dim': 20,
    'num_classes': 10,
    'lr_e': 1e-3,
    'lr_g': 3e-4,
    'lr_d': 3e-4,
    'lambda_kld': 0.1,
    'kld_anneal_max_epochs': 20,
    'lambda_recon': 10.0,
    'lambda_feat': 0.5,
    'ssl_lambda_rot': 1.0,
    'ssl_lambda_cont': 0.5,
    'ssl_lambda_inp': 0.7
}


# --- 新增：创建带时间戳的保存目录 ---
def create_save_directory():
    """创建一个基于当前时间的唯一目录来保存所有结果"""
    # 在 'training_runs' 文件夹下创建
    base_dir = "training_runs"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(base_dir, timestamp)

    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    print(f"所有结果将保存到: {save_dir}")
    return save_dir


# --- 新增：保存超参数 ---
def save_hyperparameters(save_dir, main_config, trainer_cfg):
    """将超参数保存为 JSON 文件"""
    all_params = {
        "main_config": main_config,
        "trainer_config": trainer_cfg
    }

    save_path = os.path.join(save_dir, "hyperparameters.json")
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_params, f, indent=4, ensure_ascii=False)
        print(f"超参数已保存到: {save_path}")
    except Exception as e:
        print(f"保存超参数失败: {e}")


def main():
    """主训练函数"""
    print(f"--- VAEGAN 训练开始 ---")
    print(f"设备: {DEVICE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"批量大小: {BATCH_SIZE}")

    # 1. 加载数据
    print("加载 MNIST 数据集...")
    # 注意: utils.py 中的 num_workers=4, 如果在 Windows 上或内存不足, 可能需要调低
    train_loader, test_loader = load_mnist_data(batch_size=BATCH_SIZE, data_dir=DATA_DIR)
    print("数据加载完毕。")

    # --- 修改：创建保存目录并传递给 Trainer ---
    run_save_dir = create_save_directory()

    # --- 新增：保存超参数 ---
    main_hyperparams = {
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "DEVICE": DEVICE,
        "DATA_DIR": DATA_DIR
    }
    save_hyperparameters(run_save_dir, main_hyperparams, trainer_config)  

    # 2. 初始化训练器
    print("初始化 VAEGANTrainer...")
    # 使用 trainer.py 中定义的默认超参数, 并传入新的保存路径
    # 使用 **trainer_config 解包传递超参数
    trainer = VAEGANTrainer(device=DEVICE, save_dir=run_save_dir, **trainer_config)
    print("训练器初始化完毕。")

    # 3. 开始训练循环
    for epoch in range(1, EPOCHS + 1):
        # train_epoch 方法会处理所有逻辑并打印当前轮次的结果
        trainer.train_epoch(train_loader, test_loader)

    print(f"--- 训练完成 ---")

    # 4. 可视化最终结果
    print("生成最终可视化结果 (损失图, 准确率图, 最终样本)...")
    # --- 修改：传入 test_loader 以便生成混淆矩阵 ---
    trainer.visualize_training(test_loader)

    # 5. 保存模型
    # --- 修改：保存到新的 run_save_dir ---
    model_save_path = os.path.join(run_save_dir, "vaegan_final_model.pth")
    print(f"保存最终模型到 {model_save_path}...")
    trainer.save_model(model_save_path)

    print(f"--- 任务结束 ---")


if __name__ == "__main__":
    # 设置随机种子以便复现 (可选)
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)

    main()

  