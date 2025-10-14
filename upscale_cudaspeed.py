"""
ComfyUI Upscale CUDAspeed
优化编译后长时间处理和尺寸变化重新编译的问题
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from spandrel import ModelLoader, ImageModelDescriptor
from comfy import model_management
import comfy.utils
import folder_paths
from typing import Optional, Tuple, List
import math
import time
import gc
import os
import pickle
import hashlib

try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError:
    tqdm_available = False
    print("tqdm not available, using basic progress indicators")

try:
    from spandrel_extra_arches import EXTRA_REGISTRY
    from spandrel import MAIN_REGISTRY
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    logging.info("成功导入spandrel_extra_arches：支持非商业放大模型。")
except:
    pass

class UpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                             }}
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"

    def load_model(self, model_name):
        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
        out = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(out, ImageModelDescriptor):
            raise Exception("放大模型必须是单图像模型。")

        return (out, )


class ImageUpscaleWithModelCUDAspeedFixed:
    """高性能放大节点
    优化编译后长时间处理和尺寸变化重新编译的问题
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "use_autocast": (["enable", "disable"], {"default": "enable"}),
                "precision": (["auto", "fp16", "fp32", "bf16"], {"default": "auto"}),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 128, "step": 8}),
                "enable_compile": (["enable", "disable"], {"default": "enable"}),
                "optimization_level": (["balanced", "speed", "memory"], {"default": "balanced"}),
            },
            "optional": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    
    # 编译后的模型缓存（类变量，在实例间共享）
    _compiled_models = {}
    
    # 编译模型存储目录（用于记录编译状态）
    _compiled_models_dir = None
    
    # 运行时编译缓存（类变量，在实例间共享）
    _runtime_compiled_models = {}
    
    # 尺寸编译缓存 - 关键修复：为不同尺寸缓存编译结果
    _size_compiled_models = {}
    
    def __init__(self):
        """初始化模型存储目录"""
        # 设置编译模型存储目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self._compiled_models_dir = os.path.join(current_dir, "compiled_models")
        os.makedirs(self._compiled_models_dir, exist_ok=True)
        print(f"📁 编译模型存储目录: {self._compiled_models_dir}")
        
        # 初始化运行时缓存（如果是第一次实例化）
        if not hasattr(ImageUpscaleWithModelCUDAspeedFixed, '_runtime_compiled_models'):
            ImageUpscaleWithModelCUDAspeedFixed._runtime_compiled_models = {}
        
        # 初始化尺寸编译缓存
        if not hasattr(ImageUpscaleWithModelCUDAspeedFixed, '_size_compiled_models'):
            ImageUpscaleWithModelCUDAspeedFixed._size_compiled_models = {}
        
        # 加载已编译模型记录
        self._load_compiled_models_info()
        
        # 调试：显示运行时缓存状态
        print(f"🔍 运行时缓存状态: {len(ImageUpscaleWithModelCUDAspeedFixed._runtime_compiled_models)} 个编译模型")
        print(f"🔍 尺寸缓存状态: {len(ImageUpscaleWithModelCUDAspeedFixed._size_compiled_models)} 个尺寸缓存")
    
    def _get_model_hash(self, model_state_dict):
        """生成模型状态字典的哈希值"""
        # 创建一个简化的模型状态用于哈希计算
        simplified_state = {}
        for key, value in model_state_dict.items():
            # 只取部分关键参数计算哈希，避免计算量过大
            if 'weight' in key or 'bias' in key:
                # 取前100个元素计算哈希
                flat_value = value.flatten()
                sample_size = min(100, len(flat_value))
                simplified_state[key] = flat_value[:sample_size].cpu().numpy().tobytes()
        
        # 计算哈希
        hash_obj = hashlib.md5()
        for key in sorted(simplified_state.keys()):
            hash_obj.update(simplified_state[key])
        
        return hash_obj.hexdigest()
    
    def _get_compiled_model_path(self, model_hash):
        """获取编译模型信息文件路径"""
        return os.path.join(self._compiled_models_dir, f"compiled_{model_hash}.pkl")
    
    def _load_compiled_models_info(self):
        """加载已编译模型信息（仅记录，不加载编译函数）"""
        print("🔍 检查已编译的模型信息...")
        loaded_count = 0
        
        for filename in os.listdir(self._compiled_models_dir):
            if filename.startswith("compiled_") and filename.endswith(".pkl"):
                file_path = os.path.join(self._compiled_models_dir, filename)
                try:
                    # 只检查文件是否存在，不实际加载编译函数
                    if os.path.getsize(file_path) > 0:
                        # 只记录模型哈希，不加载编译函数
                        model_hash = filename.replace("compiled_", "").replace(".pkl", "")
                        self._compiled_models[model_hash] = True  # 标记为已编译
                        loaded_count += 1
                        print(f"  ✅ 发现编译模型记录: {filename}")
                        
                except Exception as e:
                    print(f"  ❌ 检查编译模型失败 {filename}: {e}")
        
        print(f"📊 发现 {loaded_count} 个编译模型记录")
    
    def _save_compiled_model_info(self, model_hash):
        """保存编译模型信息到文件（不保存实际的编译函数）"""
        try:
            # 不保存编译函数本身，只保存编译记录
            compiled_data = {
                'model_hash': model_hash,
                'save_time': time.time(),
                'compile_info': '模型已编译，编译函数无法序列化保存'
            }
            
            file_path = self._get_compiled_model_path(model_hash)
            with open(file_path, 'wb') as f:
                pickle.dump(compiled_data, f)
            
            print(f"💾 编译模型信息已保存: {os.path.basename(file_path)}")
            return True
            
        except Exception as e:
            print(f"❌ 保存编译模型信息失败: {e}")
            return False

    def upscale(self, upscale_model, image, use_autocast="enable", precision="auto",
                tile_size=0, overlap=0, enable_compile="enable", optimization_level="balanced",
                batch_size=1):
        
        print(f"🚀 开始图像放大处理")
        print(f"📊 输入图像尺寸: {image.shape}")
        
        # 获取模型信息
        model_name = self._get_model_name(upscale_model)
        print(f"🔧 使用放大模型: {model_name}, 模型缩放比例: {upscale_model.scale}")
        print(f"⚙️ 使用参数 - 自动混合精度: {use_autocast}, 精度: {precision}")
        print(f"🔧 优化级别: {optimization_level}, 模型编译: {enable_compile}")
        
        # 详细性能监控
        total_start_time = time.time()
        phase_start_time = total_start_time
        
        # 确定精度和优化设置
        dtype, autocast_enabled = self._determine_precision(precision, use_autocast)
        phase_end_time = time.time()
        print(f"⏱️ 精度设置完成 - 耗时: {phase_end_time - phase_start_time:.3f}秒")
        phase_start_time = phase_end_time
        
        # 智能参数计算
        tile_size, overlap = self._calculate_optimal_tile_size(
            image.shape, upscale_model.scale, tile_size, overlap, optimization_level
        )
        phase_end_time = time.time()
        print(f"⏱️ 参数计算完成 - 耗时: {phase_end_time - phase_start_time:.3f}秒")
        phase_start_time = phase_end_time
        
        print(f"📐 优化参数 - 瓦片大小: {tile_size}, 重叠: {overlap}")
        
        # 执行放大处理
        result = self._upscale_fixed(
            upscale_model, image, dtype, autocast_enabled,
            tile_size, overlap, enable_compile, batch_size
        )
        
        # 性能统计
        total_end_time = time.time()
        processing_time = total_end_time - total_start_time
        print(f"✅ 图像放大处理完成 - 总耗时: {processing_time:.2f}秒")
        print(f"📊 输出图像尺寸: {result[0].shape}")
        
        return result

    def _get_model_name(self, upscale_model):
        """获取模型名称信息"""
        model_name = getattr(upscale_model, 'name', None)
        if model_name is None:
            model_name = getattr(upscale_model, '__class__', type(upscale_model)).__name__
            if hasattr(upscale_model, 'model'):
                underlying_model = getattr(upscale_model.model, '__class__', None)
                if underlying_model:
                    model_name = f"{model_name}({underlying_model.__name__})"
            else:
                model_name = type(upscale_model).__name__
        return model_name

    def _determine_precision(self, precision, use_autocast):
        """确定精度设置"""
        if precision == "auto":
            if model_management.should_use_fp16():
                precision = "fp16"
            else:
                precision = "fp32"
        
        dtype = torch.float32
        autocast_enabled = False
        
        if use_autocast == "enable":
            if precision == "fp16":
                dtype = torch.float16
                autocast_enabled = True
            elif precision == "bf16":
                dtype = torch.bfloat16
                autocast_enabled = True
        
        return dtype, autocast_enabled

    def _calculate_optimal_tile_size(self, image_shape, scale_factor, tile_size, overlap, optimization_level):
        """智能计算最优瓦片大小和重叠"""
        _, _, height, width = image_shape if len(image_shape) == 4 else (1, *image_shape[1:])
        
        # 如果用户指定了参数，使用用户指定的值
        if tile_size > 0 and overlap > 0:
            return tile_size, overlap
        
        # 根据优化级别计算默认值
        if optimization_level == "speed":
            base_tile = 512  # 优化：减小默认瓦片大小，避免过大瓦片导致性能下降
            base_overlap = 16
        elif optimization_level == "memory":
            base_tile = 256   # 小瓦片，节省内存
            base_overlap = 24
        else:  # balanced
            base_tile = 384
            base_overlap = 32
        
        # 根据图像尺寸智能调整瓦片大小
        max_dim = max(height, width)
        
        # 优化：更智能的瓦片大小计算
        if max_dim <= 512:
            tile_size = min(512, base_tile)
        elif max_dim <= 1024:
            tile_size = min(512, base_tile)  # 对于1080p以下图像，使用512瓦片
        elif max_dim <= 1920:
            tile_size = min(640, base_tile)  # 对于2K图像，使用640瓦片
        else:
            tile_size = base_tile
        
        # 优化：根据实际图像尺寸进一步调整
        # 如果图像尺寸小于瓦片大小，直接使用图像尺寸
        if height < tile_size and width < tile_size:
            tile_size = max(height, width)
        
        # 根据缩放比例调整重叠
        overlap = max(8, base_overlap // max(1, int(scale_factor)))
        
        print(f"🔧 智能瓦片计算 - 图像尺寸: {width}x{height}, 计算瓦片: {tile_size}x{tile_size}, 重叠: {overlap}")
        
        return tile_size, overlap

    def _upscale_fixed(self, upscale_model, image, dtype, autocast_enabled,
                      tile_size, overlap, enable_compile, batch_size):
        """修复性能问题的单GPU放大实现"""
        device = model_management.get_torch_device()
        print(f"💻 使用设备: {device}")
        print(f"🔍 设备跟踪 - _upscale_fixed入口: 输入图像设备={image.device}")
        
        # 先将原始模型移到设备
        upscale_model.to(device)
        
        # 准备编译模型 - 关键修复：使用尺寸感知的编译缓存
        use_compiled_model = False
        compiled_forward = None
        
        # 生成尺寸键用于缓存
        size_key = f"{image.shape[2]}x{image.shape[3]}"  # 高度x宽度
        print(f"📐 当前输入尺寸: {size_key}")
        
        if enable_compile == "enable" and hasattr(torch, 'compile'):
            # 获取模型哈希作为唯一标识
            model_hash = None
            try:
                if hasattr(upscale_model, 'model') and hasattr(upscale_model.model, 'state_dict'):
                    model_state_dict = upscale_model.model.state_dict()
                    model_hash = self._get_model_hash(model_state_dict)
                    print(f"🔑 模型哈希: {model_hash}")
            except Exception as e:
                print(f"⚠️ 获取模型哈希失败: {e}")
                model_hash = None
            
            # 检查是否有编译记录
            has_compile_record = model_hash and model_hash in self._compiled_models
            
            # 使用尺寸感知的缓存键
            model_key = f"{model_hash}_{size_key}" if model_hash else f"{id(upscale_model)}_{size_key}"
            
            # 调试：显示缓存查找状态
            print(f"🔍 缓存查找 - 模型键: {model_key}")
            print(f"🔍 运行时缓存中存在: {model_key in ImageUpscaleWithModelCUDAspeedFixed._runtime_compiled_models}")
            print(f"🔍 尺寸缓存中存在: {model_key in ImageUpscaleWithModelCUDAspeedFixed._size_compiled_models}")
            print(f"🔍 编译记录 - 模型哈希: {model_hash}, 记录存在: {has_compile_record}")
            
            # 关键修复：优先检查尺寸缓存
            if model_key in ImageUpscaleWithModelCUDAspeedFixed._size_compiled_models:
                # 使用尺寸缓存的编译模型
                compiled_forward = ImageUpscaleWithModelCUDAspeedFixed._size_compiled_models[model_key]
                use_compiled_model = True
                print(f"✅ 使用已编译模型 (尺寸缓存: {size_key})")
            elif model_key in ImageUpscaleWithModelCUDAspeedFixed._runtime_compiled_models:
                # 使用运行时缓存的编译模型
                compiled_forward = ImageUpscaleWithModelCUDAspeedFixed._runtime_compiled_models[model_key]
                use_compiled_model = True
                print(f"✅ 使用已编译模型 (运行时缓存: {size_key})")
            else:
                # 需要重新编译
                if has_compile_record:
                    print(f"🔧 重新编译模型 (已有记录，但尺寸 {size_key} 未缓存)...")
                else:
                    print(f"🔧 编译模型以优化性能 (尺寸: {size_key})...")
                
                try:
                    # 尝试编译模型的forward方法
                    if hasattr(upscale_model, 'model') and hasattr(upscale_model.model, 'forward'):
                        # 使用最安全的编译配置，完全避免CUDA图问题
                        import os
                        os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
                        torch._inductor.config.triton.cudagraphs = False
                        torch._inductor.config.triton.cudagraph_trees = False
                        
                        # 简化的编译过程 - 移除复杂的进度条
                        print("🔄 开始模型编译... (这可能需要几秒钟)")
                        compile_start_time = time.time()
                        
                        # 使用最简单的编译模式
                        compiled_forward = torch.compile(
                            upscale_model.model.forward,
                            mode="default",
                            fullgraph=False,
                            dynamic=False  # 固定尺寸编译，性能更好
                        )
                        
                        compile_end_time = time.time()
                        compile_time = compile_end_time - compile_start_time
                        
                        print(f"✅ 编译完成 - 耗时: {compile_time:.2f}秒")
                        
                        # 关键修复：同时保存到运行时缓存和尺寸缓存
                        ImageUpscaleWithModelCUDAspeedFixed._runtime_compiled_models[model_key] = compiled_forward
                        ImageUpscaleWithModelCUDAspeedFixed._size_compiled_models[model_key] = compiled_forward
                        
                        # 保存编译记录（不保存编译函数本身）
                        if model_hash and not has_compile_record:
                            self._compiled_models[model_hash] = True
                            self._save_compiled_model_info(model_hash)
                            print("✅ 模型编译成功并已记录")
                        else:
                            print("✅ 模型编译成功")
                        
                        use_compiled_model = True
                        
                    else:
                        print("⚠️ 模型结构不支持编译，使用普通模式")
                        use_compiled_model = False
                except Exception as e:
                    print(f"⚠️ 模型编译失败，使用普通模式: {e}")
                    use_compiled_model = False
        
        # 启用Tensor Core优化
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # 创建优化的CUDA流
        compute_stream = torch.cuda.Stream(device)
        data_stream = torch.cuda.Stream(device)
        
        # 异步数据预处理：在编译模型的同时准备输入数据
        print("🔄 开始异步数据预处理...")
        data_prep_start = time.time()
        
        # 准备输入图像（异步）
        with torch.cuda.stream(data_stream):
            in_img = image.movedim(-1, -3).to(device, non_blocking=True)
        
        data_prep_end = time.time()
        print(f"⏱️ 数据预处理完成 - 耗时: {data_prep_end - data_prep_start:.2f}秒")
        
        # 内存管理
        print("🔄 开始内存优化...")
        memory_start = time.time()
        self._optimize_memory_usage(upscale_model, in_img, tile_size, device)
        memory_end = time.time()
        print(f"⏱️ 内存优化完成 - 耗时: {memory_end - memory_start:.2f}秒")
        
        # 等待数据预处理完成
        print("🔄 等待数据预处理完成...")
        data_stream.synchronize()
        
        # 执行放大处理
        try:
            result = self._process_tiles_fixed(
                upscale_model, compiled_forward, use_compiled_model, in_img,
                autocast_enabled, dtype, tile_size, overlap, compute_stream,
                data_stream, batch_size, device
            )
            
            # 智能显存管理：根据显存情况决定输出设备
            result = self._smart_memory_management(result, upscale_model, device)
            
        finally:
            # 清理内存
            upscale_model.to("cpu")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        return result

    def _optimize_memory_usage(self, upscale_model, image, tile_size, device):
        """优化内存使用"""
        # 计算内存需求
        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (tile_size * tile_size * 3) * image.element_size() * 384.0
        memory_required += image.nelement() * image.element_size()
        
        # 释放内存
        model_management.free_memory(memory_required, device)
        
        # 预分配GPU内存池（如果可用）
        if hasattr(torch.cuda, 'memory_allocated'):
            current_allocated = torch.cuda.memory_allocated(device)
            print(f"💾 GPU内存使用: {current_allocated / 1024**3:.2f} GB")

    def _process_tiles_fixed(self, upscale_model, compiled_forward, use_compiled_model, in_img,
                           autocast_enabled, dtype, tile_size, overlap, compute_stream,
                           data_stream, batch_size, device):
        """修复的瓦片处理 - 简化流程，移除不必要的预热"""
        print(f"🔍 设备跟踪 - _process_tiles_fixed入口: 输入图像设备={in_img.device}")
        oom = True
        current_tile_size = tile_size
        max_retries = 3
        retry_count = 0
        
        while oom and retry_count < max_retries:
            try:
                # 计算处理步骤
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2],
                    tile_x=current_tile_size, tile_y=current_tile_size,
                    overlap=overlap
                )
                print(f"📈 预计处理步骤数: {steps}, 当前瓦片大小: {current_tile_size}x{current_tile_size}")
                
                # 创建进度条
                pbar = self._create_progress_bar(steps)
                
                # 优化的放大函数 - 支持编译和普通模式
                def upscale_fn(x):
                    with torch.cuda.stream(compute_stream):
                        if use_compiled_model and compiled_forward is not None:
                            # 使用编译后的forward函数
                            if autocast_enabled:
                                with torch.autocast(device_type="cuda", dtype=dtype):
                                    # 编译后的函数已经绑定了模型实例
                                    result = compiled_forward(x)
                            else:
                                result = compiled_forward(x)
                        else:
                            # 使用原始模型
                            if autocast_enabled:
                                with torch.autocast(device_type="cuda", dtype=dtype):
                                    result = upscale_model(x)
                            else:
                                result = upscale_model(x)
                        
                        # 确保输出数据类型正确
                        if autocast_enabled and result.dtype != torch.float32:
                            result = result.float()
                    
                    compute_stream.synchronize()
                    return result
                
                # 使用优化的瓦片缩放
                print("🔄 开始tiled_scale处理...")
                print(f"🔍 设备跟踪 - tiled_scale调用前: 输入设备={in_img.device}")
                tiled_scale_start_time = time.time()
                
                # 执行实际的tiled_scale处理
                with torch.no_grad():
                    s = comfy.utils.tiled_scale(
                        in_img,
                        upscale_fn,
                        tile_x=current_tile_size,
                        tile_y=current_tile_size,
                        overlap=overlap,
                        upscale_amount=upscale_model.scale,
                        output_device=device,  # 关键优化：直接输出到GPU，避免不必要的CPU传输
                        pbar=pbar
                    )
                
                tiled_scale_end_time = time.time()
                print(f"✅ tiled_scale处理完成 - 耗时: {tiled_scale_end_time - tiled_scale_start_time:.3f}秒")
                print(f"🔍 设备跟踪 - tiled_scale调用后: 输出设备={s.device}")
                
                oom = False
                
                # 关闭进度条
                if hasattr(pbar, 'close'):
                    pbar.close()
                    
            except model_management.OOM_EXCEPTION as e:
                retry_count += 1
                current_tile_size = max(128, current_tile_size // 2)
                print(f"⚠️ 内存不足，减小瓦片大小到 {current_tile_size}x{current_tile_size} (重试 {retry_count}/{max_retries})")
                
                if current_tile_size < 128:
                    raise e
        
        if oom:
            raise model_management.OOM_EXCEPTION("无法在可用内存内处理图像")
        
        # 优化：由于tiled_scale已直接输出到GPU，直接使用GPU后处理
        print("🔍 检查输出设备状态...")
        print(f"📊 输出张量设备: {s.device}, 形状: {s.shape}")
        
        # 确保在GPU上进行后处理
        if s.device.type != 'cuda':
            print(f"🔄 将结果移动到GPU进行后处理 (当前设备: {s.device})")
            s = s.to(device, non_blocking=True)
            print(f"✅ 结果已移动到GPU: {s.device}")
        
        # 使用GPU后处理
        s = self._gpu_post_process(s, device)
        
        return (s,)

    def _create_progress_bar(self, steps):
        """创建进度条"""
        if tqdm_available:
            return tqdm(total=steps, desc="单GPU放大处理", unit="tile", leave=False)
        else:
            return comfy.utils.ProgressBar(steps)

    def _post_process_output(self, output_tensor):
        """修复编译模型输出发白问题的后处理"""
        print(f"🔧 开始增强后处理，输入设备: {output_tensor.device}")
        print(f"🔍 设备跟踪 - _post_process_output: 输入设备={output_tensor.device}")
        
        # 调整维度顺序
        s = output_tensor.movedim(-3, -1)
        print(f"🔍 设备跟踪 - movedim后: 设备={s.device}")
        
        # 处理非数值
        s = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
        print(f"🔍 设备跟踪 - nan_to_num后: 设备={s.device}")
        
        # 详细的数值统计分析
        s_min = torch.min(s)
        s_max = torch.max(s)
        s_mean = torch.mean(s)
        s_std = torch.std(s)
        
        print(f"📊 原始输出统计 - 最小值: {s_min:.4f}, 最大值: {s_max:.4f}, 平均值: {s_mean:.4f}, 标准差: {s_std:.4f}")
        
        # 检测编译模型特有的数值范围问题
        if s_max > 10.0 or s_min < -5.0:
            # 严重范围偏移 - 编译模型常见问题
            print("⚠️ 检测到严重数值范围偏移，进行深度归一化")
            
            # 方法1: 基于统计的归一化
            if s_std > 0.01:  # 有合理的分布
                # 使用3-sigma规则裁剪异常值
                lower_bound = s_mean - 3 * s_std
                upper_bound = s_mean + 3 * s_std
                s = torch.clamp(s, min=lower_bound, max=upper_bound)
                
                # 重新计算统计量
                s_min = torch.min(s)
                s_max = torch.max(s)
            
            # 方法2: 分位数归一化（更鲁棒）
            try:
                # 使用分位数避免极端值影响
                q_low = torch.quantile(s, 0.01)
                q_high = torch.quantile(s, 0.99)
                s = torch.clamp(s, min=q_low, max=q_high)
                
                # 重新计算统计量
                s_min = torch.min(s)
                s_max = torch.max(s)
            except:
                pass  # 分位数计算失败时使用原有方法
            
            # 最终归一化到[0,1]
            if s_max - s_min > 1e-6:
                s = (s - s_min) / (s_max - s_min)
            else:
                s = torch.zeros_like(s)  # 全零情况
        
        elif s_max > 1.0 or s_min < 0.0:
            # 轻微范围偏移
            print("⚠️ 检测到轻微数值偏移，进行裁剪归一化")
            
            # 限制到合理范围
            s = torch.clamp(s, min=0.0, max=s_max)
            
            # 如果最大值仍然大于1，进行缩放
            if s_max > 1.0:
                s = s / s_max
        
        else:
            # 正常范围，直接限制
            s = torch.clamp(s, min=0.0, max=1.0)
        
        # 最终确保在[0,1]范围内
        s = torch.clamp(s, min=0.0, max=1.0)
        
        # 最终统计验证
        final_min = torch.min(s)
        final_max = torch.max(s)
        final_mean = torch.mean(s)
        
        print(f"✅ 处理后统计 - 最小值: {final_min:.4f}, 最大值: {final_max:.4f}, 平均值: {final_mean:.4f}")
        print(f"🔧 增强后处理完成，输出设备: {s.device}")
        
        return s

    def _accurate_memory_assessment(self, output_tensor, device):
        """优化的显存评估 - 基于实际张量，使用更宽松的阈值"""
        # 使用实际张量计算显存需求
        output_memory = output_tensor.nelement() * output_tensor.element_size()
        
        # 获取当前显存状态
        if hasattr(torch.cuda, 'get_device_properties'):
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            
            # 计算真正的可用显存：总显存 - 已分配显存
            actual_available_memory = total_memory - allocated
            
            # 优化：根据总显存大小动态调整安全余量
            if total_memory >= 20 * 1024**3:  # 20GB以上大显存显卡
                safety_margin = 2 * 1024**3  # 2GB
            else:
                safety_margin = 4 * 1024**3  # 4GB
                
            available_memory = actual_available_memory - safety_margin
            
            print(f"💾 优化显存评估 - 输出张量形状: {output_tensor.shape}")
            print(f"💾 优化显存评估 - 元素数量: {output_tensor.nelement()}")
            print(f"💾 优化显存评估 - 元素大小: {output_tensor.element_size()} 字节")
            print(f"💾 优化显存评估 - 总显存: {total_memory/1024**3:.2f}GB")
            print(f"💾 优化显存评估 - 已分配: {allocated/1024**3:.2f}GB")
            print(f"💾 优化显存评估 - 实际可用: {actual_available_memory/1024**3:.2f}GB")
            print(f"💾 优化显存评估 - 安全余量后可用: {available_memory/1024**3:.2f}GB")
            print(f"💾 优化显存评估 - 输出需求: {output_memory/1024**3:.2f}GB")
            
            # 优化：使用更宽松的检查条件
            # 条件1：可用显存足够容纳输出张量
            # 条件2：输出张量不超过总显存的60%
            memory_condition = available_memory >= output_memory
            threshold_condition = output_memory <= total_memory * 0.6
            
            result = memory_condition and threshold_condition
            
            if result:
                print("✅ 显存评估通过，可以使用GPU处理")
            else:
                print("❌ 显存评估未通过，使用CPU处理")
                
            return result
            
        return False

    def _ensure_gpu_processing(self, tensor, device):
        """确保张量在GPU上处理"""
        if tensor.device.type != 'cuda':
            print(f"🔄 将张量从 {tensor.device} 移动到 GPU")
            return tensor.to(device, non_blocking=True)
        return tensor

    def _gpu_post_process(self, output_tensor, device):
        """GPU上的后处理"""
        print(f"🔧 开始GPU增强后处理，输入设备: {output_tensor.device}")
        
        # 确保输入在GPU上
        output_tensor = self._ensure_gpu_processing(output_tensor, device)
        
        # 调整维度顺序
        s = output_tensor.movedim(-3, -1)
        print(f"🔍 设备跟踪 - GPU movedim后: 设备={s.device}")
        
        # 处理非数值
        s = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
        print(f"🔍 设备跟踪 - GPU nan_to_num后: 设备={s.device}")
        
        # 详细的数值统计分析
        s_min = torch.min(s)
        s_max = torch.max(s)
        s_mean = torch.mean(s)
        s_std = torch.std(s)
        
        print(f"📊 GPU原始输出统计 - 最小值: {s_min:.4f}, 最大值: {s_max:.4f}, 平均值: {s_mean:.4f}, 标准差: {s_std:.4f}")
        
        # 检测编译模型特有的数值范围问题
        if s_max > 10.0 or s_min < -5.0:
            # 严重范围偏移 - 编译模型常见问题
            print("⚠️ GPU检测到严重数值范围偏移，进行深度归一化")
            
            # 方法1: 基于统计的归一化
            if s_std > 0.01:  # 有合理的分布
                # 使用3-sigma规则裁剪异常值
                lower_bound = s_mean - 3 * s_std
                upper_bound = s_mean + 3 * s_std
                s = torch.clamp(s, min=lower_bound, max=upper_bound)
                
                # 重新计算统计量
                s_min = torch.min(s)
                s_max = torch.max(s)
            
            # 方法2: 分位数归一化（更鲁棒）
            try:
                # 使用分位数避免极端值影响
                q_low = torch.quantile(s, 0.01)
                q_high = torch.quantile(s, 0.99)
                s = torch.clamp(s, min=q_low, max=q_high)
                
                # 重新计算统计量
                s_min = torch.min(s)
                s_max = torch.max(s)
            except:
                pass  # 分位数计算失败时使用原有方法
            
            # 最终归一化到[0,1]
            if s_max - s_min > 1e-6:
                s = (s - s_min) / (s_max - s_min)
            else:
                s = torch.zeros_like(s)  # 全零情况
        
        elif s_max > 1.0 or s_min < 0.0:
            # 轻微范围偏移
            print("⚠️ GPU检测到轻微数值偏移，进行裁剪归一化")
            
            # 限制到合理范围
            s = torch.clamp(s, min=0.0, max=s_max)
            
            # 如果最大值仍然大于1，进行缩放
            if s_max > 1.0:
                s = s / s_max
        
        else:
            # 正常范围，直接限制
            s = torch.clamp(s, min=0.0, max=1.0)
        
        # 最终确保在[0,1]范围内
        s = torch.clamp(s, min=0.0, max=1.0)
        
        # 最终统计验证
        final_min = torch.min(s)
        final_max = torch.max(s)
        final_mean = torch.mean(s)
        
        print(f"✅ GPU处理后统计 - 最小值: {final_min:.4f}, 最大值: {final_max:.4f}, 平均值: {final_mean:.4f}")
        print(f"🔧 GPU增强后处理完成，输出设备: {s.device}")
        
        return s

    def _smart_memory_management(self, result, upscale_model, device):
        """智能显存管理：根据显存情况决定输出设备"""
        print("🔍 开始智能显存管理检查...")
        print(f"🔍 设备跟踪 - _smart_memory_management入口: 输入设备={result[0].device if result else 'None'}")
        
        if result is None or len(result) == 0:
            print("❓ 结果为空，跳过显存管理")
            return result
            
        output_tensor = result[0]
        print(f"📊 输出张量设备: {output_tensor.device}, 形状: {output_tensor.shape}")
        
        if output_tensor.device.type != 'cuda':
            print(f"📋 输出张量已在 {output_tensor.device}，跳过显存管理")
            return result
        
        try:
            # 计算输出张量的显存需求
            output_memory = output_tensor.nelement() * output_tensor.element_size()
            print(f"📊 输出张量显存需求: {output_memory/1024**3:.2f}GB")
            
            # 获取当前GPU显存状态
            if hasattr(torch.cuda, 'memory_reserved'):
                reserved = torch.cuda.memory_reserved(device)
                allocated = torch.cuda.memory_allocated(device)
                
                # 获取总显存和可用显存
                if hasattr(torch.cuda, 'get_device_properties'):
                    total_memory = torch.cuda.get_device_properties(device).total_memory
                    # 计算真正的可用显存：总显存 - 已分配显存
                    actual_available_memory = total_memory - allocated
                    
                    # 安全余量：保留2GB的显存用于后续操作
                    safety_margin = 2 * 1024**3  # 2GB
                    available_memory = actual_available_memory - safety_margin
                    
                    print(f"💾 显存状态 - 总显存: {total_memory/1024**3:.2f}GB, 已分配: {allocated/1024**3:.2f}GB")
                    print(f"💾 可用显存计算 - 实际可用: {actual_available_memory/1024**3:.2f}GB, 安全余量后: {available_memory/1024**3:.2f}GB")
                    print(f"📊 输出张量需求: {output_memory/1024**3:.2f}GB")
                    
                    # 如果可用显存足够，直接保留在GPU上
                    if available_memory >= output_memory:
                        print("🚀 显存充足，结果保留在GPU直接导出")
                        return result
                    else:
                        print("💾 显存不足，结果移动到CPU导出")
                        # 异步移动到CPU，减少阻塞时间
                        with torch.cuda.stream(torch.cuda.Stream(device)):
                            cpu_tensor = output_tensor.cpu()
                        print("✅ 结果已移动到CPU")
                        return (cpu_tensor,)
                else:
                    # 如果没有获取总显存功能，使用旧的逻辑
                    free_memory = reserved - allocated
                    safety_margin = reserved * 0.2
                    available_memory = free_memory - safety_margin
                    
                    print(f"💾 显存状态 (旧方法) - 已分配: {allocated/1024**3:.2f}GB, 保留: {reserved/1024**3:.2f}GB, 可用: {available_memory/1024**3:.2f}GB")
                    
                    if available_memory >= output_memory:
                        print("🚀 显存充足，结果保留在GPU直接导出")
                        return result
                    else:
                        print("💾 显存不足，结果移动到CPU导出")
                        with torch.cuda.stream(torch.cuda.Stream(device)):
                            cpu_tensor = output_tensor.cpu()
                        print("✅ 结果已移动到CPU")
                        return (cpu_tensor,)
            else:
                # 如果没有显存查询功能，保守策略：移动到CPU
                print("💾 无法获取显存信息，结果移动到CPU导出")
                with torch.cuda.stream(torch.cuda.Stream(device)):
                    cpu_tensor = output_tensor.cpu()
                return (cpu_tensor,)
                
        except Exception as e:
            print(f"⚠️ 显存管理异常，使用保守策略: {e}")
            # 异常情况下使用保守策略
            with torch.cuda.stream(torch.cuda.Stream(device)):
                cpu_tensor = output_tensor.cpu()
            return (cpu_tensor,)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("NaN")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "UpscaleModelLoader": UpscaleModelLoader,
    "ImageUpscaleWithModelCUDAspeedFixed": ImageUpscaleWithModelCUDAspeedFixed
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageUpscaleWithModelCUDAspeedFixed": "🚀 Upscale Image CUDAspeed",
}