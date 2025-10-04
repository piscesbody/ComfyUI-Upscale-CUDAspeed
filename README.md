# ComfyUI-Upscale-CUDAspeed

## 中文说明

一个高性能的ComfyUI图像放大插件，利用CUDA加速，支持多GPU、混合精度和Tensor Core优化。

### 功能特性

- **多GPU支持**：将放大工作负载分配到多个GPU以获得更快的处理速度
- **混合精度**：自动混合精度(FP16)以提高性能
- **Tensor Core优化**：利用Tensor Cores实现最大吞吐量
- **CUDA流**：重叠计算和内存传输
- **高效内存管理**：优化的内存分配策略
- **瓦片处理**：通过重叠处理高效处理大图像

### 安装

1. 将此仓库克隆到您的ComfyUI自定义节点目录：
```
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-Upscale-CUDAspeed.git
```

2. 安装依赖：
```
cd ComfyUI-Upscale-CUDAspeed
pip install -r requirements.txt
```

3. 重启ComfyUI

### 使用方法

#### 放大模型加载器
- 从ComfyUI models/upscale_models目录加载放大模型
- 兼容大多数放大模型（ESRGAN、Real-ESRGAN等）

#### 放大图像 CUDAspeed
- 将加载的模型和图像连接到节点
- 配置以下参数：

##### 参数说明：
- **use_autocast**：启用/禁用自动混合精度（默认：启用）
  - 混合精度（Autocast）是一种技术，它在计算过程中自动使用不同的数值精度（通常是FP16和FP32）来加速计算并减少内存使用，同时保持模型输出质量。启用后可以显著提高性能，但可能在某些情况下影响输出质量。
- **precision**：选择精度模式（auto, fp16, fp32, bf16）
- **multi_gpu_mode**：GPU使用策略（auto, primary_only, dual_gpu）
- **tile_size**：处理瓦片大小（128-2048，默认：512）
  - 瓦片大小（Tile Size）决定了每次处理的图像块大小。较大的瓦片尺寸通常更高效，但需要更多显存（VRAM）。较小的瓦片尺寸使用较少显存，但处理效率较低。根据您的GPU显存大小调整此参数。
- **overlap**：瓦片重叠大小（8-128，默认：32）
  - 重叠（Overlap）是指相邻瓦片之间的重叠像素数。这用于减少瓦片边界处的接缝和伪影。较大的重叠值可以减少接缝，但会增加处理时间。通常，重叠值应设置为瓦片大小的5-10%。
- **gpu_load_balance**：GPU之间的负载分布（0.0-1.0）

##### 推荐设置：

###### 单GPU模式：
- multi_gpu_mode: primary_only
- use_autocast: enable（如果支持）
- tile_size: 512-1024（基于显存）

###### 多GPU模式：
- multi_gpu_mode: dual_gpu
- gpu_load_balance: 根据GPU规格调整
- use_autocast: enable

###### 最佳质量：
- use_autocast: disable
- precision: fp32
- overlap: 较高值（64-128）

###### 最佳速度：
- use_autocast: enable
- precision: fp16
- multi_gpu_mode: dual_gpu（如果有多GPU）

### 性能提示

1. **显存管理**：如果遇到内存不足错误，请降低瓦片大小
2. **多GPU**：当GPU相同或相似时，性能提升最明显
3. **混合精度**：提供20-30%的速度提升，质量损失极小
4. **瓦片大小**：较大的瓦片更高效，但需要更多显存

### 故障排除

#### 常见问题：
- **黑色输出**：尝试禁用autocast或切换到单GPU模式
- **模型兼容性**：部分模型目前可能存在不兼容的情况，目前还不知道具体原因，如果遇到输出内容为黑色，请关闭多GPU改为单GPU，同时关闭autocast，即可正常出图。遇到不兼容的模型，麻烦提issues我这边测试（如果能解决的话）
- **内存不足错误**：减小瓦片大小或使用单GPU模式
- **性能缓慢**：确保已正确安装CUDA和cuDNN

#### 兼容性：
- 需要支持CUDA的NVIDIA GPU
- 兼容ComfyUI 0.2.0+
- 需要支持CUDA的PyTorch

### 节点

#### UpscaleModelLoader
- 从标准ComfyUI模型目录加载放大模型
- 支持spandrel库兼容的所有模型

#### ImageUpscaleWithModelCUDAspeed
- 具有所有性能优化的主要放大节点
- 支持图像批次和单个图像
- 处理具有各种放大倍数的模型

---

## English Documentation

A high-performance image upscaling plugin for ComfyUI that leverages CUDA acceleration with multi-GPU support, mixed precision, and Tensor Core optimization.

### Features

- **Multi-GPU Support**: Distribute upscaling workload across multiple GPUs for faster processing
- **Mixed Precision**: Automatic mixed precision (FP16) for improved performance
- **Tensor Core Optimization**: Leverages Tensor Cores for maximum throughput
- **CUDA Streams**: Overlapped computation and memory transfers
- **Efficient Memory Management**: Optimized memory allocation strategies
- **Tile-based Processing**: Handles large images efficiently with overlap handling

### Installation

1. Clone this repository to your ComfyUI custom nodes directory:
```
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-Upscale-CUDAspeed.git
```

2. Install dependencies:
```
cd ComfyUI-Upscale-CUDAspeed
pip install -r requirements.txt
```

3. Restart ComfyUI

### Usage

#### Upscale Model Loader
- Load upscale models from your ComfyUI models/upscale_models directory
- Compatible with most upscale models (ESRGAN, Real-ESRGAN, etc.)

#### Upscale Image CUDAspeed
- Connect your loaded model and image to the node
- Configure the following parameters:

##### Parameters:
- **use_autocast**: Enable/disable automatic mixed precision (default: enable)
  - Mixed precision (Autocast) is a technique that automatically uses different numerical precisions (typically FP16 and FP32) during computation to accelerate calculations and reduce memory usage while maintaining model output quality. Enabling it can significantly improve performance, but may affect output quality in some cases.
- **precision**: Choose precision mode (auto, fp16, fp32, bf16)
- **multi_gpu_mode**: GPU usage strategy (auto, primary_only, dual_gpu)
- **tile_size**: Processing tile size (128-2048, default: 512)
  - Tile size determines the size of image chunks processed at once. Larger tile sizes are generally more efficient but require more VRAM. Smaller tile sizes use less VRAM but are less efficient. Adjust this parameter based on your GPU's VRAM capacity.
- **overlap**: Tile overlap size (8-128, default: 32)
  - Overlap refers to the number of overlapping pixels between adjacent tiles. This is used to reduce seams and artifacts at tile boundaries. Larger overlap values can reduce seams but increase processing time. Usually, overlap should be set to 5-10% of the tile size.
- **gpu_load_balance**: Load distribution between GPUs (0.0-1.0)

##### Recommended Settings:

###### For Single GPU:
- multi_gpu_mode: primary_only
- use_autocast: enable (if supported)
- tile_size: 512-1024 based on VRAM

###### For Multi GPU:
- multi_gpu_mode: dual_gpu
- gpu_load_balance: Adjust based on your GPU specs
- use_autocast: enable

###### For Best Quality:
- use_autocast: disable
- precision: fp32
- overlap: higher values (64-128)

###### For Best Speed:
- use_autocast: enable
- precision: fp16
- multi_gpu_mode: dual_gpu (if multiple GPUs available)

### Performance Tips

1. **VRAM Management**: Lower tile_size if you encounter out-of-memory errors
2. **Multi-GPU**: Performance scales best when GPUs are identical or similar
3. **Mixed Precision**: Provides 20-30% speedup with minimal quality loss
4. **Tile Size**: Larger tiles are more efficient but require more VRAM

### Troubleshooting

#### Common Issues:
- **Black Output**: Try disabling autocast or switching to single GPU mode
- **Model Compatibility**: Some models may currently have compatibility issues, the specific reasons are unknown at present. If you encounter black output, please turn off multi-GPU and switch to single GPU, while also turning off autocast, to generate images normally. For incompatible models, please submit issues for testing (if solvable).
- **OOM Errors**: Reduce tile_size or use single GPU mode
- **Slow Performance**: Ensure CUDA and cuDNN are properly installed

#### Compatibility:
- Requires NVIDIA GPU with CUDA support
- Compatible with ComfyUI 0.2.0+
- Requires PyTorch with CUDA support

### Nodes

#### UpscaleModelLoader
- Loads upscale models from the standard ComfyUI model directory
- Supports all models compatible with spandrel library

#### ImageUpscaleWithModelCUDAspeed
- Main upscaling node with all performance optimizations
- Supports image batches and single images
- Handles models with various scale factors

## License

MIT License - see LICENSE file for details.