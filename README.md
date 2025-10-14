# ComfyUI-Upscale-CUDAspeed

一个高性能的ComfyUI图像放大插件，通过CUDA加速和模型编译优化提供极致的放大速度。
<img width="649" height="438" alt="image" src="https://github.com/user-attachments/assets/37597f8a-f5f2-4932-a3a7-59c6c17fc17e" />

## 特性

- 🚀 **高性能CUDA加速**：利用PyTorch编译技术优化模型推理速度
- 🔧 **智能模型编译**：自动编译模型并缓存，避免重复编译
- 🎯 **尺寸感知缓存**：为不同输入尺寸分别缓存编译结果，避免尺寸变化导致的重新编译
- 💾 **智能内存管理**：动态调整瓦片大小，优化显存使用
- 🛡️ **数值稳定性**：增强的后处理确保编译模型输出质量
- ⚡ **异步处理**：使用多CUDA流并行处理，最大化GPU利用率

## 安装

### 前提条件

- ComfyUI 已安装并运行
- NVIDIA GPU 支持 CUDA
- PyTorch 2.0+ (支持 `torch.compile`)

### 安装步骤

1. 将本仓库克隆到ComfyUI的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-Upscale-CUDAspeed.git
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 重启ComfyUI

## 使用说明

### 节点介绍

#### 🚀 Upscale Image CUDAspeed

主要放大节点，提供高性能的图像放大功能。

**输入参数：**

- `upscale_model`: 放大模型（通过UpscaleModelLoader加载）推荐模型：RealESRGAN_x2plus.pth
- `image`: 输入图像
- `use_autocast`: 自动混合精度（启用/禁用）
- `precision`: 精度模式（自动/fp16/fp32/bf16）推荐：fp16
- `tile_size`: 瓦片大小（0表示自动计算）推荐：视频最长边如1280x720，则输入1280
- `overlap`: 瓦片重叠大小（0表示自动计算）推荐：8（视频最小值即可）
- `enable_compile`: 模型编译（启用/禁用）如果批量处理同尺寸，建议打开。第一次编译有点长，但是加速效果明显。
- `optimization_level`: 优化级别（平衡/速度/内存）推荐：speed
- `batch_size` (可选): 批处理大小 推荐：1

#### UpscaleModelLoader

放大模型加载器，用于加载各种放大模型。

**输入参数：**

- `model_name`: 模型文件名（从 `upscale_models` 目录选择）

### 工作流示例

1. 使用 `UpscaleModelLoader` 节点加载放大模型
2. 连接图像到 `🚀 Upscale Image CUDAspeed` 节点
3. 调整参数以获得最佳性能和质量
4. 执行工作流

### 性能优化建议

#### 优化级别选择

- **平衡模式**：适合大多数场景，在速度和内存间取得平衡
- **速度模式**：追求最高速度，适合大显存显卡
- **内存模式**：节省显存，适合小显存显卡

#### 模型编译

启用 `enable_compile` 可以显著提升推理速度，但首次运行需要编译时间。编译后的模型会自动缓存，后续运行无需重新编译。

#### 自动混合精度

启用 `use_autocast` 可以利用Tensor Core加速计算，在支持的GPU上提供更好的性能。

## 支持的模型

本插件支持所有与Spandrel兼容的放大模型，包括：

- ESRGAN
- Real-ESRGAN
- Real-CUGAN
- SwinIR
- HAT
- 以及其他单图像放大模型

## 技术细节

### 模型编译优化

插件使用PyTorch的 `torch.compile` 功能对模型进行即时编译优化：

- **尺寸感知缓存**：为不同输入尺寸分别缓存编译结果
- **运行时缓存**：在内存中缓存编译模型，避免重复编译
- **持久化记录**：记录模型编译状态，重启后仍有效

### 内存管理

- **动态瓦片调整**：根据可用显存自动调整瓦片大小
- **智能内存评估**：基于实际张量计算显存需求
- **流式处理**：使用多CUDA流并行处理数据

### 数值稳定性

针对编译模型可能出现的数值范围问题，提供了增强的后处理：

- 异常值检测和裁剪
- 分位数归一化
- 统计驱动的范围调整

## 故障排除

### 常见问题

**Q: 编译失败或出现错误**
A: 尝试禁用 `enable_compile` 使用普通模式，或检查PyTorch版本是否支持编译。

**Q: 显存不足**
A: 尝试使用内存优化模式，减小 `tile_size`，或启用 `use_autocast`。

**Q: 输出图像发白或颜色异常**
A: 这是编译模型的常见问题，插件已内置增强后处理。如果问题持续，尝试禁用模型编译。

**Q: 性能没有提升**
A: 确保使用支持的GPU和PyTorch 2.0+版本，首次运行需要编译时间。

### 日志调试

插件会输出详细的调试信息，包括：

- 模型编译状态
- 内存使用情况
- 处理时间统计
- 设备跟踪信息

查看ComfyUI控制台输出可以了解详细的运行状态。

## 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。
