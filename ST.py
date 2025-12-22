import numpy as np
from matplotlib import cm
from PIL import Image
from scipy.io import loadmat
from scipy.fft import fft, ifft, fftshift
from scipy.signal import butter, filtfilt
import os
from datetime import datetime
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings
import multiprocessing
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm  # <--- 1. 导入 tqdm 库

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """处理配置类"""
    fs: int = 12000
    segment_length: int = 1024
    overlap: float = 0.5
    image_size: Tuple[int, int] = (224, 224)
    filter_order: int = 4
    low_freq_ratio: float = 50 / 6000  # 10Hz / Nyquist
    high_freq_ratio: float = 2000 / 6000
    colormap: str = 'viridis'  # <--- 色图
    num_workers: Optional[int] = None

    # --- 仅保留频率裁剪参数 ---
    freq_min_hz: int = 1000
    freq_max_hz: int = 5000

    keep_aspect_ratio: bool = True  # True=不拉伸，False=填满图像
    background_color: Tuple[int, int, int] = (0, 0, 0)  # 背景色(黑色)


class OptimizedCWRUDataProcessor:
    """
    优化版本的CWRU数据处理器
    """

    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.fs = self.config.fs
        try:
            self.cmap = cm.get_cmap(self.config.colormap)
        except ValueError:
            logger.warning(f"无效的 colormap '{self.config.colormap}', 将回退到 'viridis'")
            self.config.colormap = 'viridis'
            self.cmap = cm.get_cmap(self.config.colormap)

    def fast_s_transform_optimized(self, signal: np.ndarray) -> np.ndarray:
        """
        优化版本的快速S变换实现 (已移除 st_alpha)
        """
        N = len(signal)
        if N == 0:
            raise ValueError("输入信号长度不能为0")

        X = fft(signal)
        S = np.zeros((N, N), dtype=np.complex128)
        S[:, 0] = X[0] / N
        n = np.arange(N)
        n_shifted = n - N // 2

        for k in range(1, N):
            if k == 0:
                continue

            # --- 恢复：使用原始S变换窗口 ---
            denominator = k ** 2
            if denominator < 1e-9:
                denominator = 1e-9

            window = np.exp(-2 * (np.pi ** 2) * (n_shifted ** 2) / denominator)
            window = fftshift(window)
            W = fft(window)
            convolution = ifft(X * W)
            phase_factor = np.exp(1j * 2 * np.pi * k * n / N)
            S[:, k] = convolution * phase_factor

        return S

    def preprocess_signal_robust(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        改进的信号预处理，增强鲁棒性 (未修改)
        """
        if len(signal) == 0:
            raise ValueError("输入信号为空")
        signal = signal - np.mean(signal)
        if np.all(signal == 0):
            logger.warning("信号全为零，跳过滤波处理")
            return [signal]
        try:
            nyquist = self.fs / 2
            low_freq = self.config.low_freq_ratio
            high_freq = self.config.high_freq_ratio
            if low_freq >= high_freq:
                logger.warning(f"无效的频率范围: {low_freq} >= {high_freq}")
                low_freq = 0.01
                high_freq = 0.4
            b, a = butter(self.config.filter_order, [low_freq, high_freq], btype='band')
            signal = filtfilt(b, a, signal)
        except Exception as e:
            logger.warning(f"滤波失败: {e}，使用原始信号")
        segment_length = self.config.segment_length
        overlap = self.config.overlap
        step = int(segment_length * (1 - overlap))
        if step <= 0:
            raise ValueError("步长必须大于0")
        segments = []
        for i in range(0, len(signal) - segment_length + 1, step):
            segment = signal[i:i + segment_length]
            std_val = np.std(segment)
            if std_val > 1e-8:
                segment = (segment - np.mean(segment)) / std_val
            else:
                logger.warning(f"段 {i // step} 标准差过小，跳过归一化")
            segments.append(segment)
        return segments

    def generate_time_freq_image_enhanced(self, S: np.ndarray, save_path: str) -> bool:
        """
        增强版时频图像生成 (仅包含频率裁剪)
        """
        try:
            amplitude = np.abs(S)
            N = amplitude.shape[0]

            # --- 频率裁剪逻辑 ---
            df = self.fs / N
            k_min = max(0, int(self.config.freq_min_hz / df))
            k_max = min(N // 2, int(self.config.freq_max_hz / df))

            if k_min >= k_max:
                logger.warning("无效的频率范围 (min >= max)，跳过图像生成")
                return False

            # --- 更改：转置矩阵，使行=频率, 列=时间 ---
            # positive_freq_amplitude = amplitude[:, k_min:k_max] # 旧代码：行=时间, 列=频率
            positive_freq_amplitude = amplitude[:, k_min:k_max].T
            # ----------------------------------------

            if positive_freq_amplitude.size == 0:
                logger.warning("裁剪后的频率振幅为空，跳过")
                return False

            log_amplitude = np.log1p(positive_freq_amplitude)

            if np.isnan(log_amplitude).any() or np.isinf(log_amplitude).any():
                logger.error("时频数据包含无效值")
                return False

            # --- 恢复：使用标准的最大/最小值归一化 ---
            min_val = np.min(log_amplitude)
            max_val = np.max(log_amplitude)

            if max_val - min_val < 1e-9:
                norm_amplitude = np.zeros_like(log_amplitude)
            else:
                norm_amplitude = (log_amplitude - min_val) / (max_val - min_val)

            colored_array = self.cmap(norm_amplitude)
            rgb_array_uint8 = (colored_array[:, :, :3] * 255).astype(np.uint8)
            img_pil = Image.fromarray(rgb_array_uint8)
            img_resized = img_pil.resize(self.config.image_size, Image.Resampling.LANCZOS)
            img_resized.save(save_path, 'PNG')
            return True

        except Exception as e:
            logger.error(f"生成图像时出错：{e}")
            return False

    def process_single_mat_file_enhanced(self, mat_file_path: str, output_dir: str) -> int:  # <--- 2. 更改返回类型为 int
        """
        增强版单文件处理 (返回成功生成的图像数量)
        """
        if not os.path.exists(mat_file_path):
            logger.error(f"文件不存在：{mat_file_path}")
            return 0  # <--- 返回 0
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"无法创建输出目录：{e}")
            return 0  # <--- 返回 0
        try:
            # logger.info(f"正在加载文件：{mat_file_path}") # 在tqdm循环中这会刷屏，注释掉
            data = loadmat(mat_file_path)
            signal_key = self._find_signal_key(data)
            if signal_key is None:
                logger.error(f"文件 {mat_file_path} 未找到有效的信号数据")
                return 0  # <--- 返回 0
            signal = data[signal_key].flatten()
            segments = self.preprocess_signal_robust(signal)
            base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
            success_count = 0
            for i, segment in enumerate(segments):
                try:
                    S = self.fast_s_transform_optimized(segment)
                    save_path = os.path.join(output_dir, f"{base_name}_seg{i:03d}.png")
                    if self.generate_time_freq_image_enhanced(S, save_path):
                        success_count += 1
                except Exception as e:
                    logger.error(f"文件 {mat_file_path} 处理段 {i} 时出错：{e}")
                    continue
            # logger.info(f"处理完成 {mat_file_path}！成功生成 {success_count}/{len(segments)} 张时频图") # 同样注释掉
            return success_count  # <--- 3. 返回图像数量
        except Exception as e:
            logger.error(f"处理文件 {mat_file_path} 时出错：{e}")
            return 0  # <--- 返回 0

    def _find_signal_key(self, data: Dict[str, Any]) -> Optional[str]:
        """
        智能查找信号数据键 (未修改)
        """
        possible_keys = []
        for key in data.keys():
            if not key.startswith('__'):
                possible_keys.append(key)
                if 'DE_time' in key:
                    # logger.info(f"找到驱动端信号键：{key}")
                    return key
        if possible_keys:
            # logger.info(f"使用数据键：{possible_keys[0]}")
            return possible_keys[0]
        return None

    def auto_detect_signal_key(self, mat_file_path: str) -> List[str]:
        """
        自动检测.mat文件中的信号数据键 (未修改)
        """
        try:
            data = loadmat(mat_file_path)
            keys = [key for key in data.keys() if not key.startswith('__')]
            logger.info(f"文件 {os.path.basename(mat_file_path)} 中的数据键：")
            for i, key in enumerate(keys):
                try:
                    shape = data[key].shape
                    logger.info(f"  {i + 1}. {key}: {shape}")
                except Exception as e:
                    logger.warning(f"  {i + 1}. {key}: 无法获取形状 - {e}")
            return keys
        except Exception as e:
            logger.error(f"读取文件时出错：{e}")
            return []


# --- 用于并行处理的顶层辅助函数 ---
def global_process_wrapper(task_tuple: Tuple[str, str, ProcessingConfig]) -> Tuple[str, int]:  # <--- 4. 更改返回类型
    """
    一个顶层函数，用于 multiprocessing.Pool.map
    (返回: 文件路径, 成功生成的图像数, -1表示文件处理异常)
    """
    file_path, output_root, config = task_tuple
    try:
        processor = OptimizedCWRUDataProcessor(config)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        final_output_dir = os.path.join(output_root, base_name)
        # --- 5. 捕获图像数量 ---
        image_count = processor.process_single_mat_file_enhanced(
            mat_file_path=file_path,
            output_dir=final_output_dir
        )
        return file_path, image_count
    except Exception as e:
        logger.error(f"子进程处理 {file_path} 失败: {e}")
        return file_path, -1  # <--- 6. 返回-1表示异常


# --- 使用示例 ---
if __name__ == "__main__":

    # --- 1. 配置 ---
    config = ProcessingConfig(
        fs=12000,
        segment_length=1024,
        overlap=0.5,
        image_size=(224, 224),
        num_workers=None,
        colormap='viridis',  # 色图
        freq_min_hz=50,  # 开始频率 (Hz)
        freq_max_hz=3000,  # 结束频率 (Hz)
        keep_aspect_ratio=True,  # 不拉伸
        background_color=(0, 0, 0)  # 黑色背景填充
    )

    # --- 2. 定义 *总* 输出根目录 ---
    output_dir_root = r'images'
    os.makedirs(output_dir_root, exist_ok=True)

    # --- 3. 弹出文件选择框 ---
    logger.info("正在打开文件选择对话框...")
    root = tk.Tk()
    root.withdraw()

    file_list = filedialog.askopenfilenames(
        title="请选择一个或多个 .mat 处理文件",
        filetypes=[("MAT files", "*.mat"), ("All files", "*.*")]
    )

    if not file_list:
        logger.info("未选择文件，程序退出。")
        exit()  

    logger.info(f"总共选择了 {len(file_list)} 个 .mat 文件。")

    # --- 4. 准备并行任务 ---
    tasks = [(file_path, output_dir_root, config) for file_path in file_list]

    # --- 5. 执行并行处理 (使用 tqdm) ---
    num_workers = config.num_workers or os.cpu_count()
    logger.info(f"=== 开始并行处理（使用 {num_workers} 个核心） ===")

    total_images_generated = 0
    success_file_count = 0
    failed_files = []

    start_time = datetime.now()
    try:
        # 设置 'spawn' 启动方式，在 Windows 和 macOS 上更稳定
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            # 使用 imap_unordered 以便在任务完成时立即获取结果
            # 使用 tqdm 包装迭代器以显示进度条
            progress_bar = tqdm(
                pool.imap_unordered(global_process_wrapper, tasks),
                total=len(tasks),
                desc="文件处理进度",
                unit="file",
                ncols=100  # 可选：设置进度条宽度
            )

            for file_path, image_count in progress_bar:
                if image_count >= 0:  # 0 或 更多图片表示文件本身处理未出错
                    total_images_generated += image_count
                    if image_count > 0:
                        success_file_count += 1
                    else:
                        # 文件处理成功，但未生成图片 (例如，信号太短或全为0)
                        logger.warning(f"文件 {file_path} 处理完毕，但未生成任何图片。")
                        failed_files.append(file_path)  # 仍将其计为“失败”
                else:  # image_count == -1, 表示处理过程中出现异常
                    logger.error(f"文件 {file_path} 处理时发生严重错误。")
                    failed_files.append(file_path)

                # 动态更新进度条的后缀信息
                progress_bar.set_postfix_str(
                    f"已生成图片: {total_images_generated}, 成功文件: {success_file_count}"
                )

    except Exception as e:
        logger.error(f"并行池出错: {e}")
    end_time = datetime.now()

    # --- 6. 报告总结 ---
    logger.info("=== 并行处理完成 ===")
    logger.info(f"总耗时: {end_time - start_time}")
    logger.info(f"成功处理文件 (至少1张图片): {success_file_count} / {len(tasks)}")
    logger.info(f"总共生成图片: {total_images_generated}")
    logger.info(f"所有图像已保存到 '{output_dir_root}' 目录下的对应子文件夹中。")

    if failed_files:
        logger.warning(f"失败或未生成图片的文件: {len(failed_files)} 个")
        for f in failed_files:
            logger.warning(f"  - {f}")

