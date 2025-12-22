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
import multiprocessing
import tkinter as tk
from tkinter import filedialog

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """处理配置类"""
    fs: int = 12000
    segment_length: int = 2048
    overlap: float = 0.75
    image_size: Tuple[int, int] = (224, 224)
    filter_order: int = 4
    low_freq_ratio: float = 50 / 6000
    high_freq_ratio: float = 2000 / 6000
    colormap: str = 'jet'
    num_workers: Optional[int] = None

    # 频率裁剪参数
    freq_min_hz: int = 0
    freq_max_hz: int = 2000

    # === 新增：S变换优化参数 ===
    st_window_scale: float = 1.0  # 窗口缩放因子（增大=更平滑）
    use_percentile_norm: bool = True  # 使用百分位归一化
    norm_percentile: float = 99.5  # 归一化百分位
    freq_smoothing: int = 5  # 频率维度平滑（减少条纹）


class OptimizedCWRUDataProcessor:
    """优化版本的CWRU数据处理器"""

    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.fs = self.config.fs
        try:
            self.cmap = cm.get_cmap(self.config.colormap)
        except ValueError:
            logger.warning(f"无效的 colormap '{self.config.colormap}', 将回退到 'jet'")
            self.config.colormap = 'jet'
            self.cmap = cm.get_cmap(self.config.colormap)

    def fast_s_transform_optimized(self, signal: np.ndarray) -> np.ndarray:
        """
        优化版本的快速S变换实现
        === 修改：添加窗口缩放，使其更接近STFT ===
        """
        N = len(signal)
        if N == 0:
            raise ValueError("输入信号长度不能为0")

        X = fft(signal)
        S = np.zeros((N, N), dtype=np.complex128)
        S[:, 0] = X[0] / N
        n = np.arange(N)
        n_shifted = n - N // 2

        # 获取窗口缩放因子
        scale = self.config.st_window_scale

        for k in range(1, N):
            if k == 0:
                continue

            # === 修改：应用窗口缩放 ===
            # 原始：denominator = k ** 2
            # 缩放后：窗口变宽，频率分辨率降低，更接近STFT
            denominator = (k / scale) ** 2
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
        """改进的信号预处理，增强鲁棒性"""
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
        增强版时频图像生成
        === 修改：优化归一化和平滑处理 ===
        """
        try:
            amplitude = np.abs(S)
            N = amplitude.shape[0]

            # 频率裁剪逻辑
            df = self.fs / N
            k_min = max(0, int(self.config.freq_min_hz / df))
            k_max = min(N // 2, int(self.config.freq_max_hz / df))

            if k_min >= k_max:
                logger.warning("无效的频率范围 (min >= max)，跳过图像生成")
                return False

            positive_freq_amplitude = amplitude[:, k_min:k_max]

            if positive_freq_amplitude.size == 0:
                logger.warning("裁剪后的频率振幅为空，跳过")
                return False

            # === 修改1：频率维度平滑（减少横向条纹）===
            if self.config.freq_smoothing > 1:
                from scipy.ndimage import uniform_filter
                # 只在频率维度（axis=1）平滑
                positive_freq_amplitude = uniform_filter(
                    positive_freq_amplitude,
                    size=(1, self.config.freq_smoothing)
                )

            # === 修改2：对数变换 ===
            log_amplitude = np.log1p(positive_freq_amplitude)

            if np.isnan(log_amplitude).any() or np.isinf(log_amplitude).any():
                logger.error("时频数据包含无效值")
                return False

            # === 修改3：使用百分位归一化（类似STFT）===
            if self.config.use_percentile_norm:
                # 使用百分位数而不是最大最小值
                vmin = np.percentile(log_amplitude, 100 - self.config.norm_percentile)
                vmax = np.percentile(log_amplitude, self.config.norm_percentile)

                if vmax - vmin < 1e-9:
                    norm_amplitude = np.zeros_like(log_amplitude)
                else:
                    # 裁剪到百分位范围
                    norm_amplitude = np.clip(log_amplitude, vmin, vmax)
                    norm_amplitude = (norm_amplitude - vmin) / (vmax - vmin)
            else:
                # 原始归一化方式
                min_val = np.min(log_amplitude)
                max_val = np.max(log_amplitude)
                if max_val - min_val < 1e-9:
                    norm_amplitude = np.zeros_like(log_amplitude)
                else:
                    norm_amplitude = (log_amplitude - min_val) / (max_val - min_val)

            # 应用色图
            colored_array = self.cmap(norm_amplitude)
            rgb_array_uint8 = (colored_array[:, :, :3] * 255).astype(np.uint8)
            img_pil = Image.fromarray(rgb_array_uint8)
            img_resized = img_pil.resize(self.config.image_size, Image.Resampling.LANCZOS)
            img_resized.save(save_path, 'PNG')

            return True

        except Exception as e:
            logger.error(f"生成图像时出错：{e}")
            return False

    def process_single_mat_file_enhanced(self, mat_file_path: str, output_dir: str) -> bool:
        """增强版单文件处理"""
        if not os.path.exists(mat_file_path):
            logger.error(f"文件不存在：{mat_file_path}")
            return False
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"无法创建输出目录：{e}")
            return False
        try:
            logger.info(f"正在加载文件：{mat_file_path}")
            data = loadmat(mat_file_path)
            signal_key = self._find_signal_key(data)
            if signal_key is None:
                logger.error("未找到有效的信号数据")
                return False
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
                    logger.error(f"处理段 {i} 时出错：{e}")
                    continue
            logger.info(f"处理完成 {mat_file_path}：成功生成 {success_count}/{len(segments)} 张时频图")
            return success_count > 0
        except Exception as e:
            logger.error(f"处理文件 {mat_file_path} 时出错：{e}")
            return False

    def _find_signal_key(self, data: Dict[str, Any]) -> Optional[str]:
        """智能查找信号数据键"""
        possible_keys = []
        for key in data.keys():
            if not key.startswith('__'):
                possible_keys.append(key)
                if 'DE_time' in key:
                    logger.info(f"找到驱动端信号键：{key}")
                    return key
        if possible_keys:
            logger.info(f"使用数据键：{possible_keys[0]}")
            return possible_keys[0]
        return None


def global_process_wrapper(task_tuple: Tuple[str, str, ProcessingConfig]) -> Tuple[str, bool]:
    """用于并行处理的顶层辅助函数"""
    file_path, output_root, config = task_tuple
    try:
        processor = OptimizedCWRUDataProcessor(config)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        final_output_dir = os.path.join(output_root, base_name)
        success = processor.process_single_mat_file_enhanced(
            mat_file_path=file_path,
            output_dir=final_output_dir
        )
        return file_path, success
    except Exception as e:
        logger.error(f"子进程处理 {file_path} 失败: {e}")
        return file_path, False


if __name__ == "__main__":

    # === 推荐配置：使S变换更接近STFT ===
    config = ProcessingConfig(
        fs=12000,
        segment_length=2048,
        overlap=0.75,
        image_size=(224, 224),

        # 频率范围
        freq_min_hz=0,
        freq_max_hz=2000,

        # 滤波参数
        low_freq_ratio=50 / 6000,
        high_freq_ratio=2000 / 6000,

        # 可视化
        colormap='jet',

        # === 关键优化参数 ===
        st_window_scale=2.0,  # 窗口缩放（1.0=标准ST，越大越接近STFT）
        use_percentile_norm=True,  # 使用百分位归一化
        norm_percentile=99.0,  # 归一化百分位（99%）
        freq_smoothing=3,  # 频率平滑（减少横向条纹）

        num_workers=None
    )

    output_dir_root = r'images_stft_style'
    os.makedirs(output_dir_root, exist_ok=True)

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

    tasks = [(file_path, output_dir_root, config) for file_path in file_list]

    num_workers = config.num_workers or os.cpu_count()
    logger.info(f"=== 开始并行处理（使用 {num_workers} 个核心） ===")

    start_time = datetime.now()
    results = []
    try:
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            results = list(pool.map(global_process_wrapper, tasks))
    except Exception as e:
        logger.error(f"并行池出错: {e}")
    end_time = datetime.now()

    success_count = sum(1 for res in results if res[1])
    failed_files = [res[0] for res in results if not res[1]]

    logger.info("=== 并行处理完成 ===")
    logger.info(f"总耗时: {end_time - start_time}")
    logger.info(f"成功处理文件: {success_count} / {len(tasks)}")
    logger.info(f"所有图像已保存到 '{output_dir_root}' 目录下的对应子文件夹中。")

    if failed_files:
        logger.warning(f"失败 {len(failed_files)} 个文件:")
        for f in failed_files:
            logger.warning(f"  - {f}")