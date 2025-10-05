"""
ComfyUI Upscale CUDAspeed - 高性能图像放大节点
实现多GPU支持、自动混合精度和Tensor Core优化
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


class ImageUpscaleWithModelCUDAspeed:
    """高性能放大节点，支持多GPU、自动混合精度和Tensor Core优化"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "use_autocast": (["enable", "disable"], {"default": "enable"}),
                "precision": (["auto", "fp16", "fp32", "bf16"], {"default": "auto"}),
                "multi_gpu_mode": (["auto", "primary_only", "dual_gpu"], {"default": "auto"}),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
            },
            "optional": {
                "gpu_load_balance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image, use_autocast="enable", precision="auto", multi_gpu_mode="auto", tile_size=512, overlap=32, gpu_load_balance=0.0):
        print(f"开始图像放大处理，输入图像尺寸: {image.shape}")
        # 尝试获取更具体的模型名称
        model_name = getattr(upscale_model, 'name', None)
        if model_name is None:
            # 尝试从模型属性中获取更多信息
            model_name = getattr(upscale_model, '__class__', type(upscale_model)).__name__
            # 如果是ImageModelDescriptor实例，尝试获取底层模型信息
            if hasattr(upscale_model, 'model'):
                underlying_model = getattr(upscale_model.model, '__class__', None)
                if underlying_model:
                    model_name = f"{model_name}({underlying_model.__name__})"
            else:
                model_name = type(upscale_model).__name__
        print(f"使用放大模型: {model_name}, 模型缩放比例: {upscale_model.scale}")
        print(f"使用参数 - 自动混合精度: {use_autocast}, 精度: {precision}, 多GPU模式: {multi_gpu_mode}")
        
        # 确定精度
        if precision == "auto":
            if model_management.should_use_fp16():
                precision = "fp16"
            else:
                precision = "fp32"
        
        # 确定数据类型
        dtype = torch.float32
        if precision == "fp16" and use_autocast == "enable":
            dtype = torch.float16
        elif precision == "bf16" and use_autocast == "enable":
            dtype = torch.bfloat16

        # 确定多GPU模式
        if multi_gpu_mode == "auto":
            available_gpus = torch.cuda.device_count()
            print(f"检测到 {available_gpus} 个可用GPU")
            if available_gpus > 1:
                multi_gpu_mode = "dual_gpu"
            else:
                multi_gpu_mode = "primary_only"
        
        # 根据GPU配置使用适当的放大方法
        print(f"选择放大模式: {multi_gpu_mode}")
        if multi_gpu_mode == "dual_gpu" and torch.cuda.device_count() > 1:
            result = self.upscale_multi_gpu(upscale_model, image, use_autocast, dtype, tile_size, overlap, gpu_load_balance)
        else:
            result = self.upscale_single_gpu(upscale_model, image, use_autocast, dtype, tile_size, overlap)
        print(f"图像放大处理完成，输出图像尺寸: {result[0].shape}")
        return result

    def upscale_single_gpu(self, upscale_model, image, use_autocast, dtype, tile_size, overlap):
        """单GPU放大，使用混合精度和Tensor Core优化"""
        device = model_management.get_torch_device()
        upscale_model.to(device)

        print(f"开始单GPU放大处理，图像尺寸: {image.shape}, 设备: {device}")

        # 创建CUDA流以重叠计算和内存传输
        compute_stream = torch.cuda.Stream(device)
        default_stream = torch.cuda.current_stream(device)

        # 计算内存需求
        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (tile_size * tile_size * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        # 使用非阻塞传输将图像移至设备
        in_img = image.movedim(-1, -3).to(device, non_blocking=True)

        # 确定是否使用自动混合精度
        autocast_enabled = use_autocast == "enable" and dtype in [torch.float16, torch.bfloat16]
        autocast_dtype = dtype if autocast_enabled else None

        oom = True
        current_tile_size = tile_size
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2],
                    tile_x=current_tile_size, tile_y=current_tile_size,
                    overlap=overlap
                )
                print(f"预计处理步骤数: {steps}, 当前瓦片大小: {current_tile_size}x{current_tile_size}")
                
                # 如果有tqdm则创建进度条，否则使用原有进度条
                if tqdm_available:
                    tqdm_pbar = tqdm(total=steps, desc="单GPU放大处理", unit="tile", leave=False)
                    # 创建一个包装类来桥接tqdm和ComfyUI的ProgressBar
                    class TqdmProgressBar:
                        def __init__(self, pbar):
                            self.pbar = pbar
                        
                        def update(self, value):
                            self.pbar.update(value)
                        
                        def close(self):
                            self.pbar.close()
                    
                    wrapped_pbar = TqdmProgressBar(tqdm_pbar)
                    actual_pbar = wrapped_pbar
                else:
                    actual_pbar = comfy.utils.ProgressBar(steps)
                
                # 如果启用则使用自动混合精度
                with torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_enabled else torch.no_grad():
                    # 启用Tensor Core优化
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    
                    # 使用直接调用而非lambda函数，确保混合精度正确处理
                    def upscale_fn(x):
                        # 切换到计算流以重叠计算和内存传输
                        with torch.cuda.stream(compute_stream):
                            result = upscale_model(x)
                            # 如果使用混合精度，确保输出张量是正确的数据类型
                            if autocast_enabled:
                                result = result.float()
                        # 等待计算流完成，确保结果正确
                        compute_stream.synchronize()
                        return result
                    
                    s = comfy.utils.tiled_scale(
                        in_img,
                        upscale_fn,
                        tile_x=current_tile_size,
                        tile_y=current_tile_size,
                        overlap=overlap,
                        upscale_amount=upscale_model.scale,
                        pbar=actual_pbar
                    )
                
                # 等待计算流完成
                compute_stream.synchronize()
                oom = False
                
                # 关闭进度条
                if tqdm_available and 'tqdm_pbar' in locals():
                    tqdm_pbar.close()
            except model_management.OOM_EXCEPTION as e:
                if tqdm_available and 'tqdm_pbar' in locals():
                    tqdm_pbar.close()
                current_tile_size //= 2
                print(f"内存不足，减小瓦片大小到 {current_tile_size}x{current_tile_size}")
                if current_tile_size < 128:
                    raise e

        # 改进输出处理以增强模型兼容性
        s = s.movedim(-3, -1)
        # 某些模型可能输出超出[0,1]范围的值或包含NaN/无穷大，需要适当处理
        # 检查是否存在非数值或极值
        s = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 简化输出值范围处理
        s_min = torch.min(s)
        s_max = torch.max(s)
        
        # 如果值范围异常（例如全部接近0或范围过大），进行适当的归一化
        if s_max <= 1.0 and s_min >= 0.0:
            # 正常范围，直接限制
            s = torch.clamp(s, min=0.0, max=1.0)
        elif s_max - s_min > 1e-6:
            # 有合理范围，进行归一化到[0,1]
            s = (s - s_min) / (s_max - s_min)
        else:
            # 所有值几乎相同，限制到[0,1]
            s = torch.clamp(s, min=0.0, max=1.0)
        
        s = torch.clamp(s, min=0.0, max=1.0)
        
        # 将模型移回CPU以释放GPU内存，然后再返回结果
        upscale_model.to("cpu")
        return (s,)

    def upscale_multi_gpu(self, upscale_model, image, use_autocast, dtype, tile_size, overlap, gpu_load_balance=0.0):
        """多GPU放大实现，使用两个GPU进行处理"""
        device_primary = torch.device("cuda:0")
        device_secondary = torch.device("cuda:1")
        num_gpus = torch.cuda.device_count()
        
        print(f"开始多GPU放大处理，图像尺寸: {image.shape}, 主GPU: {device_primary}, 副GPU: {device_secondary}")
        print(f"使用参数 - 瓦片大小: {tile_size}, 重叠: {overlap}, 负载平衡: {gpu_load_balance}")

        if num_gpus < 2:
            print("检测到少于2个GPU，回退到单GPU模式")
            # 如果只有一个GPU可用，则回退到单GPU模式
            return self.upscale_single_gpu(upscale_model, image, use_autocast, dtype, tile_size, overlap)
        
        # 为两个GPU计算内存需求
        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (tile_size * tile_size * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device_primary)
        model_management.free_memory(memory_required, device_secondary)

        # 准备输入图像
        in_img = image.movedim(-1, -3)
        batch_size = in_img.shape[0]

        # 确定是否使用自动混合精度
        autocast_enabled = use_autocast == "enable" and dtype in [torch.float16, torch.bfloat16]
        autocast_dtype = dtype if autocast_enabled else None

        # 初始将模型移至主GPU
        upscale_model.to(device_primary)

        # 如果批次大小大于1，则在GPU间分割批次
        if batch_size > 1:
            # 根据性能计算每个GPU应处理的图像数量
            if gpu_load_balance > 0.0:
                # 使用自定义负载平衡比例
                primary_gpu_share = int(batch_size * gpu_load_balance)
                secondary_gpu_share = batch_size - primary_gpu_share
            else:
                # 自动计算基于GPU性能的负载分配
                primary_gpu_name = torch.cuda.get_device_name(device_primary)
                secondary_gpu_name = torch.cuda.get_device_name(device_secondary)
                
                # 获取GPU属性以估算性能比例
                primary_props = torch.cuda.get_device_properties(device_primary)
                secondary_props = torch.cuda.get_device_properties(device_secondary)
                
                # 基于可用GPU属性估算性能比例
                # 使用影响性能的因素加权组合：
                # 1. 计算能力（主要次要版本号）
                # 2. 多处理器数量（CUDA核心数的代理）
                # 3. 总内存大小
                primary_compute_capability = primary_props.major * 10 + primary_props.minor
                secondary_compute_capability = secondary_props.major * 10 + secondary_props.minor
                
                # 更重地加权多处理器数量，因为它是最接近CUDA核心数的代理
                # 同时将内存大小视为上采样工作负载的重要因素
                primary_performance_score = (
                    primary_compute_capability * 0.2 +  # 计算能力权重（较低，因为都是Ada Lovelace架构）
                    primary_props.multi_processor_count * 0.6 +  # 多处理器数量权重（CUDA核心数的最佳代理）
                    (primary_props.total_memory / (1024**3)) * 0.2  # 内存大小权重（GB）
                )
                
                secondary_performance_score = (
                    secondary_compute_capability * 0.2 +
                    secondary_props.multi_processor_count * 0.6 +
                    (secondary_props.total_memory / (1024**3)) * 0.2
                )
                
                # 规范化性能评分以获得比例
                total_performance_score = primary_performance_score + secondary_performance_score
                if total_performance_score > 0:
                    primary_gpu_performance_ratio = primary_performance_score / total_performance_score
                    secondary_gpu_performance_ratio = secondary_performance_score / total_performance_score
                else:
                    # 如果计算失败则回退到平均分配
                    primary_gpu_performance_ratio = 0.5
                    secondary_gpu_performance_ratio = 0.5
                
                # 计算基于性能的批次分割
                primary_gpu_share = int(batch_size * primary_gpu_performance_ratio)
                secondary_gpu_share = batch_size - primary_gpu_share
            
            # 确保两个GPU至少获得1个图像（如果可能）
            if secondary_gpu_share == 0 and batch_size > 1:
                secondary_gpu_share = 1
                primary_gpu_share = batch_size - 1
            # 输出批次分割信息
            print(f"批次分割: 主GPU处理 {primary_gpu_share} 张图像, 副GPU处理 {secondary_gpu_share} 张图像")

            # 创建模型副本用于副GPU，避免模型移动
            model_secondary = self._copy_model_to_device(upscale_model, device_secondary)
            
            # 结果存储容器
            primary_result = [None]  # 使用列表来在子线程中修改值
            secondary_result = [None]
            
            # 定义在主GPU上处理图像的函数
            def process_primary_images():
                import time  # 在函数内部导入time
                start_time = time.time()
                print(f"主GPU线程启动于 {start_time:.2f}")
                
                # 创建CUDA流以重叠计算和内存传输
                primary_compute_stream = torch.cuda.Stream(device_primary)
                
                img_primary = in_img[:primary_gpu_share].to(device_primary, non_blocking=True)
                
                if img_primary.shape[0] > 0:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_enabled else torch.no_grad():
                        # 启用Tensor Core优化
                        torch.backends.cudnn.allow_tf32 = True
                        torch.backends.cuda.matmul.allow_tf32 = True
                        
                        steps_primary = img_primary.shape[0] * comfy.utils.get_tiled_scale_steps(
                            img_primary.shape[3], img_primary.shape[2],
                            tile_x=tile_size, tile_y=tile_size,
                            overlap=overlap
                        )
                        print(f"主GPU处理 {img_primary.shape[0]} 张图像, 共 {steps_primary} 步骤")
                        
                        # 创建主GPU进度条
                        if tqdm_available:
                            tqdm_pbar_primary = tqdm(total=steps_primary, desc="主GPU处理", unit="tile", leave=False)
                            class TqdmProgressBar:
                                def __init__(self, pbar):
                                    self.pbar = pbar
                                
                                def update(self, value):
                                    self.pbar.update(value)
                                
                                def close(self):
                                    self.pbar.close()
                            
                            wrapped_pbar_primary = TqdmProgressBar(tqdm_pbar_primary)
                            actual_pbar_primary = wrapped_pbar_primary
                        else:
                            actual_pbar_primary = comfy.utils.ProgressBar(steps_primary)
                        
                        # 使用直接调用而非lambda函数，确保混合精度正确处理
                        def upscale_primary_fn(x):
                            # 切换到计算流以重叠计算和内存传输
                            with torch.cuda.stream(primary_compute_stream):
                                result = upscale_model(x)
                                # 如果使用混合精度，确保输出张量是正确的数据类型
                                if autocast_enabled:
                                    result = result.float()
                            # 等待计算流完成，确保结果正确
                            primary_compute_stream.synchronize()
                            return result
                        
                        result_primary = comfy.utils.tiled_scale(
                            img_primary,
                            upscale_primary_fn,  # 模型已在device_primary上
                            tile_x=tile_size,
                            tile_y=tile_size,
                            overlap=overlap,
                            upscale_amount=upscale_model.scale,
                            pbar=actual_pbar_primary
                        )
                        # 等待计算流完成
                        primary_compute_stream.synchronize()
                        # 在存储前检查值范围
                        result_primary = torch.clamp(result_primary, min=0, max=1e5) # 限制异常大值
                        
                        # 尽早将结果移至CPU以释放GPU内存
                        primary_result[0] = result_primary.cpu()
                        
                        # 关闭主GPU进度条
                        if tqdm_available and 'tqdm_pbar_primary' in locals():
                            tqdm_pbar_primary.close()
                        
                print(f"主GPU线程结束于 {time.time():.2f}, 耗时 {(time.time()-start_time):.2f} 秒")
                 
            # 定义在副GPU上处理图像的函数
            def process_secondary_images():
                import time  # 在函数内部导入time
                start_time = time.time()
                print(f"副GPU线程启动于 {start_time:.2f}")
                
                # 创建CUDA流以重叠计算和内存传输
                secondary_compute_stream = torch.cuda.Stream(device_secondary)
                
                img_secondary = in_img[primary_gpu_share:primary_gpu_share + secondary_gpu_share].to(device_secondary, non_blocking=True)
                
                if img_secondary.shape[0] > 0:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_enabled else torch.no_grad():
                        # 启用Tensor Core优化
                        torch.backends.cudnn.allow_tf32 = True
                        torch.backends.cuda.matmul.allow_tf32 = True
                        
                        steps_secondary = img_secondary.shape[0] * comfy.utils.get_tiled_scale_steps(
                            img_secondary.shape[3], img_secondary.shape[2],
                            tile_x=tile_size, tile_y=tile_size,
                            overlap=overlap
                        )
                        print(f"副GPU处理 {img_secondary.shape[0]} 张图像, 共 {steps_secondary} 步骤")
                        
                        # 创建副GPU进度条
                        if tqdm_available:
                            tqdm_pbar_secondary = tqdm(total=steps_secondary, desc="副GPU处理", unit="tile", leave=False)
                            class TqdmProgressBar:
                                def __init__(self, pbar):
                                    self.pbar = pbar
                                
                                def update(self, value):
                                    self.pbar.update(value)
                                
                                def close(self):
                                    self.pbar.close()
                            
                            wrapped_pbar_secondary = TqdmProgressBar(tqdm_pbar_secondary)
                            actual_pbar_secondary = wrapped_pbar_secondary
                        else:
                            actual_pbar_secondary = comfy.utils.ProgressBar(steps_secondary)
                        
                        # 使用直接调用而非lambda函数，确保混合精度正确处理
                        def upscale_secondary_fn(x):
                            # 切换到计算流以重叠计算和内存传输
                            with torch.cuda.stream(secondary_compute_stream):
                                result = model_secondary(x)
                                # 如果使用混合精度，确保输出张量是正确的数据类型
                                if autocast_enabled:
                                    result = result.float()
                            # 等待计算流完成，确保结果正确
                            secondary_compute_stream.synchronize()
                            return result
                        
                        result_secondary = comfy.utils.tiled_scale(
                            img_secondary,
                            upscale_secondary_fn,  # 使用副GPU的模型副本
                            tile_x=tile_size,
                            tile_y=tile_size,
                            overlap=overlap,
                            upscale_amount=upscale_model.scale, # 使用移动模型的比例
                            pbar=actual_pbar_secondary
                        )
                        # 等待计算流完成
                        secondary_compute_stream.synchronize()
                        # 尽早将结果移至CPU以释放GPU内存
                        # 在移动前检查值范围
                        result_secondary = torch.clamp(result_secondary, min=0, max=1e5) # 限制异常大值
                        secondary_result[0] = result_secondary.cpu()
                        
                        # 关闭副GPU进度条
                        if tqdm_available and 'tqdm_pbar_secondary' in locals():
                            tqdm_pbar_secondary.close()
                        
                print(f"副GPU线程结束于 {time.time():.2f}, 耗时 {(time.time()-start_time):.2f} 秒")
                 
            
            # 并行运行两个处理函数
            import threading
            start_processing = time.time()
            print(f"准备启动两个GPU处理线程于 {start_processing:.2f}")
            primary_thread = threading.Thread(target=process_primary_images)
            secondary_thread = threading.Thread(target=process_secondary_images)
            
            # 启动线程
            primary_thread.start()
            secondary_thread.start()
            print(f"两个GPU处理线程已启动于 {time.time():.2f}")
            
            # 等待线程完成
            primary_thread.join()
            secondary_thread.join()
            end_processing = time.time()
            print(f"两个GPU处理线程完成于 {end_processing:.2f}, 总处理耗时 {(end_processing-start_processing):.2f} 秒")
            
            # 收集结果
            results = []
            if primary_result[0] is not None:
                results.append(primary_result[0])
            if secondary_result[0] is not None:
                results.append(secondary_result[0])
            
            # 处理完成后将模型移回主GPU
            upscale_model.to(device_primary)
            del model_secondary  # 删除副GPU模型副本以释放内存
            
            # 确保所有结果都在同一设备上再进行连接
            # 使用更高效的内存分配策略：预分配结果张量
            if len(results) > 1:
                # 预计算总batch大小
                total_batch = sum(result.shape[0] for result in results)
                # 预分配结果张量，避免多次内存分配
                first_result = results[0]
                s = torch.empty(
                    (total_batch, first_result.shape[1], first_result.shape[2], first_result.shape[3]),
                    dtype=first_result.dtype,
                    device=torch.device("cpu"),  # 直接在CPU上分配
                    memory_format=torch.channels_last if first_result.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
                )
                
                # 逐个复制结果到预分配的张量中
                current_idx = 0
                for result in results:
                    # 结果已经在CPU上，直接复制
                    batch_size = result.shape[0]
                    s[current_idx:current_idx + batch_size] = result
                    current_idx += batch_size
            else:
                s = results[0] if results else torch.empty(0, device=torch.device("cpu"))
        else:
            # 单图像情况：在两个GPU上并行处理瓦片
            print("处理单图像并行瓦片放大")
            s = self.process_single_image_parallel(upscale_model, in_img, use_autocast, autocast_dtype, tile_size, overlap, device_primary, device_secondary, gpu_load_balance)

        # 在最终处理前将结果移至CPU以避免OOM
        if s.device != torch.device("cpu"):
            # 检查是否有足够的CPU内存来容纳结果
            try:
                s = s.cpu()
            except Exception as e:
                print(f"移动结果到CPU失败: {e}")
                # 如果无法移至CPU，则保持在GPU上但限制大小
                s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
                # 将模型移回CPU以释放GPU内存
                upscale_model.to("cpu")
                return (s,)
         
        print(f"图像放大处理完成，最终尺寸: {s.shape}")

        # 改进输出处理以增强模型兼容性
        s = s.movedim(-3, -1)
        # 某些模型可能输出超出[0,1]范围的值或包含NaN/无穷大，需要适当处理
        # 检查是否存在非数值或极值
        s = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 简化输出值范围处理
        s_min = torch.min(s)
        s_max = torch.max(s)
        
        # 如果值范围异常（例如全部接近0或范围过大），进行适当的归一化
        if s_max <= 1.0 and s_min >= 0.0:
            # 正常范围，直接限制
            s = torch.clamp(s, min=0.0, max=1.0)
        elif s_max - s_min > 1e-6:
            # 有合理范围，进行归一化到[0,1]
            s = (s - s_min) / (s_max - s_min)
        else:
            # 所有值几乎相同，限制到[0,1]
            s = torch.clamp(s, min=0.0, max=1.0)
        
        s = torch.clamp(s, min=0.0, max=1.0)
        
        # 将模型移回CPU以释放GPU内存
        upscale_model.to("cpu")
        return (s,)

    def process_single_image_parallel(self, upscale_model, image, use_autocast, autocast_dtype, tile_size, overlap, device_primary, device_secondary, gpu_load_balance=0.0):
        """在两个GPU上并行处理单个图像的瓦片"""
        import time  # 在函数内部导入time
        import threading
        start_time = time.time()
        
        print(f"开始并行瓦片放大处理，图像尺寸: {image.shape}")
        print(f"瓦片大小: {tile_size}, 重叠: {overlap}")
        
        height, width = image.shape[-2], image.shape[-1]
        scale_factor = upscale_model.scale
        
        # 计算瓦片位置
        tile_positions = []
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                tile_positions.append((y, min(y + tile_size, height), x, min(x + tile_size, width)))
        
        print(f"需要处理的瓦片总数: {len(tile_positions)}")
        print(f"图像尺寸: {height}x{width}, 放大倍数: {scale_factor}")
        
        # 准备结果张量
        result = torch.zeros(
            (image.shape[0], image.shape[1], int(height * scale_factor), int(width * scale_factor)),
            dtype=image.dtype,
            device=device_primary
        )
        
        # 静态分配瓦片：奇偶分配给两个GPU
        primary_tiles = tile_positions[::2] # 偶数索引瓦片分配给主GPU
        secondary_tiles = tile_positions[1::2]  # 奇数索引瓦片分配给副GPU

        print(f"瓦片分配 - 主GPU: {len(primary_tiles)} 个瓦片, 副GPU: {len(secondary_tiles)} 个瓦片")
        
        # 创建模型的副本用于每个GPU，以实现真正的并行处理
        model_primary = self._copy_model_to_device(upscale_model, device_primary)
        model_secondary = self._copy_model_to_device(upscale_model, device_secondary)
        
        # 结果存储
        results_primary = {}
        results_secondary = {}
        
        # 函数处理主GPU上的瓦片
        def process_primary_tiles():
            import time # 在函数内部导入time
            start_gpu_time = time.time()
            print(f"主GPU瓦片线程启动于 {start_gpu_time:.2f}")
            model_primary.to(device_primary) # 确保模型副本在主GPU上
            
            # 创建CUDA流以重叠计算和内存传输
            primary_tile_stream = torch.cuda.Stream(device_primary)
            
            # 创建主GPU瓦片进度条
            if tqdm_available:
                tqdm_pbar_primary = tqdm(total=len(primary_tiles), desc="主GPU瓦片处理", unit="tile", leave=False)
            else:
                print(f"主GPU需处理 {len(primary_tiles)} 个瓦片")
            
            for i, (y1, y2, x1, x2) in enumerate(primary_tiles):
                tile_start_time = time.time()
                tile = image[:, :, y1:y2, x1:x2].to(device_primary, non_blocking=True)
                
                with torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else torch.no_grad():
                    # 启用Tensor Core优化
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    
                    # 切换到计算流以重叠计算和内存传输
                    with torch.cuda.stream(primary_tile_stream):
                        upscaled_tile = model_primary(tile)
                        # 如果使用混合精度，确保输出张量是正确的数据类型
                        if use_autocast:
                            upscaled_tile = upscaled_tile.float()  # 确保结果是float32类型
                
                # 等待计算流完成
                primary_tile_stream.synchronize()
                
                # 计算结果位置
                ry1, ry2 = int(y1 * scale_factor), int(y2 * scale_factor)
                rx1, rx2 = int(x1 * scale_factor), int(x2 * scale_factor)
                
                # 存储结果前先限制值范围，避免异常值
                upscaled_tile = torch.clamp(upscaled_tile, min=0, max=1e5)
                # 存储结果
                results_primary[(ry1, ry2, rx1, rx2)] = upscaled_tile
                
                # 更新进度条
                if tqdm_available:
                    tqdm_pbar_primary.update(1)
                elif (i + 1) % max(1, len(primary_tiles) // 10) == 0 or i == 0:
                    print(f"主GPU处理进度: {i+1}/{len(primary_tiles)} 个瓦片 ({(i+1)/len(primary_tiles)*100:.1f}%)")
                    
                tile_end_time = time.time()
                # 只在调试模式下输出每个瓦片的详细信息
                # print(f"主GPU完成瓦片: ({y1},{y2},{x1},{x2}) -> ({ry1},{y2},{rx1},{rx2}) 用时: {tile_end_time - tile_start_time:.2f}秒")
            
            # 关闭主GPU进度条
            if tqdm_available:
                tqdm_pbar_primary.close()
                
            print(f"主GPU瓦片线程结束于 {time.time():.2f}, 耗时 {(time.time()-start_gpu_time):.2f} 秒")
         
        # 函数处理副GPU上的瓦片
        def process_secondary_tiles():
            import time # 在函数内部导入time
            start_gpu_time = time.time()
            print(f"副GPU瓦片线程启动于 {start_gpu_time:.2f}")
            model_secondary.to(device_secondary)  # 确保模型副本在副GPU上
            
            # 创建CUDA流以重叠计算和内存传输
            secondary_tile_stream = torch.cuda.Stream(device_secondary)
            
            # 创建副GPU瓦片进度条
            if tqdm_available:
                tqdm_pbar_secondary = tqdm(total=len(secondary_tiles), desc="副GPU瓦片处理", unit="tile", leave=False)
            else:
                print(f"副GPU需处理 {len(secondary_tiles)} 个瓦片")
            
            for i, (y1, y2, x1, x2) in enumerate(secondary_tiles):
                tile_start_time = time.time()
                tile = image[:, :, y1:y2, x1:x2].to(device_secondary, non_blocking=True)
                
                with torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else torch.no_grad():
                    # 启用Tensor Core优化
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    
                    # 切换到计算流以重叠计算和内存传输
                    with torch.cuda.stream(secondary_tile_stream):
                        upscaled_tile = model_secondary(tile)
                        # 如果使用混合精度，确保输出张量是正确的数据类型
                        if use_autocast:
                            upscaled_tile = upscaled_tile.float()  # 确保结果是float32类型
                
                # 等待计算流完成
                secondary_tile_stream.synchronize()
                
                # 计算结果位置
                ry1, ry2 = int(y1 * scale_factor), int(y2 * scale_factor)
                rx1, rx2 = int(x1 * scale_factor), int(x2 * scale_factor)
                
                # 尽早将结果移至CPU以释放GPU内存
                # 存储结果前先限制值范围，避免异常值
                upscaled_tile = torch.clamp(upscaled_tile, min=0, max=1e5)
                results_secondary[(ry1, ry2, rx1, rx2)] = upscaled_tile.cpu()
                
                # 更新进度条
                if tqdm_available:
                    tqdm_pbar_secondary.update(1)
                elif (i + 1) % max(1, len(secondary_tiles) // 10) == 0 or i == 0:
                    print(f"副GPU处理进度: {i+1}/{len(secondary_tiles)} 个瓦片 ({(i+1)/len(secondary_tiles)*100:.1f}%)")
                    
                tile_end_time = time.time()
                # 只在调试模式下输出每个瓦片的详细信息
                # print(f"副GPU完成瓦片: ({y1},{y2},{x1},{x2}) -> ({ry1},{y2},{rx1},{rx2}) 用时: {tile_end_time - tile_start_time:.2f}秒")
            
            # 关闭副GPU进度条
            if tqdm_available:
                tqdm_pbar_secondary.close()
                
            print(f"副GPU瓦片线程结束于 {time.time():.2f}, 耗时 {(time.time()-start_gpu_time):.2f} 秒")
         
        # 并行运行两个函数
        print(f"准备启动两个GPU瓦片线程于 {time.time():.2f}")
        thread_primary = threading.Thread(target=process_primary_tiles)
        thread_secondary = threading.Thread(target=process_secondary_tiles)
        
        # 启动线程
        thread_primary.start()
        thread_secondary.start()
        print(f"两个GPU瓦片线程已启动于 {time.time():.2f}")
        
        # 等待两个线程完成
        thread_primary.join()
        thread_secondary.join()
        print(f"两个GPU瓦片线程完成于 {time.time():.2f}")
        
        # 将主GPU的瓦片结果从CPU移回GPU以进行合并操作
        print("开始合并瓦片结果...")
        for (ry1, ry2, rx1, rx2), tile_result in results_primary.items():
            # 将CPU上的瓦片结果移至GPU以进行合并
            tile_result_gpu = tile_result.to(device_primary, non_blocking=True)
            # 等待传输完成
            torch.cuda.synchronize(device_primary)
            result[:, :, ry1:ry2, rx1:rx2] = tile_result_gpu
        
        # 将结果从CPU移回GPU以进行合并操作
        for (ry1, ry2, rx1, rx2), tile_result in results_secondary.items():
            # 将CPU上的瓦片结果移至GPU以进行合并
            tile_result_gpu = tile_result.to(device_primary, non_blocking=True)
            # 等待传输完成
            torch.cuda.synchronize(device_primary)
            # 处理重叠区域，通过在重叠区域取平均值
            current = result[:, :, ry1:ry2, rx1:rx2]
            new_val = tile_result_gpu
            
            # 对重叠区域进行简单平均
            combined = (current + new_val) / 2.0
            result[:, :, ry1:ry2, rx1:rx2] = combined
            
        end_time = time.time()
        print(f"并行瓦片放大完成, 总耗时 {end_time - start_time:.2f} 秒")
        return result

    def _copy_model_to_device(self, original_model, device):
        """创建模型的副本并将其移动到指定设备"""
        import copy
        import torch
        
        # 创建模型的深层副本
        model_copy = copy.deepcopy(original_model)
        # 将副本移动到目标设备
        model_copy.to(device)
        return model_copy


NODE_CLASS_MAPPINGS = {
    "UpscaleModelLoader": UpscaleModelLoader,
    "ImageUpscaleWithModelCUDAspeed": ImageUpscaleWithModelCUDAspeed
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageUpscaleWithModelCUDAspeed": "Upscale Image CUDAspeed (Multi-GPU & Mixed Precision)",
}
