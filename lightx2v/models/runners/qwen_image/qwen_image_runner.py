import gc
import math

import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from loguru import logger

from lightx2v.models.input_encoders.hf.qwen25.qwen25_vlforconditionalgeneration import Qwen25_VLForConditionalGeneration_TextEncoder
from lightx2v.models.networks.lora_adapter import LoraAdapter
from lightx2v.models.networks.qwen_image.model import QwenImageTransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.qwen_image.scheduler import QwenImageScheduler
from lightx2v.models.video_encoders.hf.qwen_image.vae import AutoencoderKLQwenImageVAE
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


def build_qwen_image_model_with_lora(qwen_module, config, model_kwargs, lora_configs):
    lora_dynamic_apply = config.get("lora_dynamic_apply", False)

    if lora_dynamic_apply:
        lora_path = lora_configs[0]["path"]
        lora_strength = lora_configs[0]["strength"]
        model_kwargs["lora_path"] = lora_path
        model_kwargs["lora_strength"] = lora_strength
        model = qwen_module(**model_kwargs)
    else:
        assert not config.get("dit_quantized", False), "Online LoRA only for quantized models; merging LoRA is unsupported."
        assert not config.get("lazy_load", False), "Lazy load mode does not support LoRA merging."
        model = qwen_module(**model_kwargs)
        lora_adapter = LoraAdapter(model)
        lora_adapter.apply_lora(lora_configs)
    return model


@RUNNER_REGISTER("qwen_image")
class QwenImageRunner(DefaultRunner):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self, config):
        # Initialize multi-GPU device config before super().__init__() since
        # the parent calls init_scheduler() which needs self.dit_device.
        self.multi_gpu_config = config.get("multi_gpu_config", None)
        if self.multi_gpu_config:
            self.text_encoder_device = torch.device(self.multi_gpu_config.get("text_encoder_device", AI_DEVICE))
            self.vae_device = torch.device(self.multi_gpu_config.get("vae_device", AI_DEVICE))
            self.dit_device = torch.device(self.multi_gpu_config.get("dit_device", AI_DEVICE))
        else:
            self.text_encoder_device = None
            self.vae_device = None
            self.dit_device = None
        super().__init__(config)
        self.is_layered = self.config.get("layered", False)
        if self.is_layered:
            self.layers = self.config.get("layers", 4)
        self.resolution = self.config.get("resolution", 1024)

        # Text encoder type: "lightllm_service", "lightllm_kernel", or default (baseline)
        self.text_encoder_type = config.get("text_encoder_type", "baseline")

        if self.text_encoder_type in ["lightllm_service", "lightllm_kernel"]:
            logger.info(f"Using LightLLM text encoder: {self.text_encoder_type}")
        
        if self.multi_gpu_config:
            logger.info(f"多 GPU 模式已启用: Text Encoder -> {self.text_encoder_device}, VAE -> {self.vae_device}, DiT -> {self.dit_device}")

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.vae = self.load_vae()

    def load_transformer(self):
        # 多 GPU 模式下使用指定的 DiT 设备
        if self.dit_device is not None:
            device = self.dit_device
            logger.info(f"DiT (Transformer) 将加载到设备: {device}")
        else:
            device = self.init_device
        
        qwen_image_model_kwargs = {
            "model_path": os.path.join(self.config["model_path"], "transformer"),
            "config": self.config,
            "device": device,
        }
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            model = QwenImageTransformerModel(**qwen_image_model_kwargs)
        else:
            model = build_qwen_image_model_with_lora(QwenImageTransformerModel, self.config, qwen_image_model_kwargs, lora_configs)
        return model

    def load_text_encoder(self):
        """Load text encoder based on text_encoder_type configuration.

        Supported types:
        - "lightllm_service": LightLLM HTTP service mode
        - "lightllm_kernel": HuggingFace model with Triton kernel optimizations
        - "baseline" (default): HuggingFace baseline implementation
        """
        # Prepare encoder config by merging lightllm_config if present
        encoder_config = self.config.copy()
        lightllm_config = self.config.get("lightllm_config", {})
        encoder_config.update(lightllm_config)
        
        # 多 GPU 模式下添加 text encoder 设备配置
        if self.text_encoder_device is not None:
            encoder_config["text_encoder_device"] = str(self.text_encoder_device)
            logger.info(f"Text Encoder 将加载到设备: {self.text_encoder_device}")

        if self.text_encoder_type == "lightllm_service":
            from lightx2v.models.input_encoders.lightllm import LightLLMServiceTextEncoder

            logger.info("Loading LightLLM service-based text encoder")
            text_encoder = LightLLMServiceTextEncoder(encoder_config)
        elif self.text_encoder_type == "lightllm_kernel":
            from lightx2v.models.input_encoders.lightllm import LightLLMKernelTextEncoder

            logger.info("Loading LightLLM Kernel-optimized text encoder")
            text_encoder = LightLLMKernelTextEncoder(encoder_config)
        else:  # baseline or default
            logger.info("Loading HuggingFace baseline text encoder")
            text_encoder = Qwen25_VLForConditionalGeneration_TextEncoder(encoder_config)

        text_encoders = [text_encoder]
        return text_encoders

    def load_image_encoder(self):
        pass

    def load_vae(self):
        # 多 GPU 模式下添加 VAE 设备配置
        vae_config = self.config.copy()
        if self.vae_device is not None:
            vae_config["vae_device"] = str(self.vae_device)
            vae_config["vae_cpu_offload"] = False  # 禁用 CPU offload
            logger.info(f"VAE 将加载到设备: {self.vae_device}")
        
        vae = AutoencoderKLQwenImageVAE(vae_config)
        return vae

    def init_modules(self):
        logger.info("Initializing runner modules...")
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.load_model()
            self.model.set_scheduler(self.scheduler)
        elif self.config.get("lazy_load", False):
            assert self.config.get("cpu_offload", False)
        self.run_dit = self._run_dit_local
        if self.config["task"] == "t2i":
            self.run_input_encoder = self._run_input_encoder_local_t2i
        elif self.config["task"] == "i2i":
            self.run_input_encoder = self._run_input_encoder_local_i2i
        else:
            assert NotImplementedError

    @ProfilingContext4DebugL2("Run DiT")
    def _run_dit_local(self, total_steps=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)
        self.model.scheduler.prepare(self.input_info)
        latents, generator = self.run(total_steps)
        return latents, generator

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_t2i(self):
        prompt = self.input_info.prompt
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()
        text_encoder_output = self.run_text_encoder(prompt, neg_prompt=self.input_info.negative_prompt)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.text_encoders[0]
        torch_device_module.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    def read_image_input(self, img_path):
        if self.config.get("layered", False):
            target_mode = "RGBA"
        else:
            target_mode = "RGB"
        if isinstance(img_path, Image.Image):
            img_ori = img_path if img_path.mode == target_mode else img_path.convert(target_mode)
        else:
            img_ori = Image.open(img_path).convert(target_mode)
        if GET_RECORDER_MODE():
            width, height = img_ori.size
            monitor_cli.lightx2v_input_image_len.observe(width * height)
        img = TF.to_tensor(img_ori).sub_(0.5).div_(0.5).unsqueeze(0).to(AI_DEVICE)
        self.input_info.original_size.append(img_ori.size)
        return img, img_ori

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2i(self):
        raw = self.input_info.image_path
        if isinstance(raw, Image.Image):
            image_paths_list = [raw]
        elif isinstance(raw, list):
            image_paths_list = raw
        else:
            image_paths_list = raw.split(",")
        images_list = []
        for image_path in image_paths_list:
            _, image = self.read_image_input(image_path)
            images_list.append(image)

        prompt = self.input_info.prompt
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()
        text_encoder_output = self.run_text_encoder(prompt, images_list, neg_prompt=self.input_info.negative_prompt)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            # Offload text encoder (service mode doesn't need offload)
            if self.text_encoder_type == "lightllm_service":
                pass  # Service mode: no local model to offload
            else:
                del self.text_encoders[0]
        image_encoder_output_list = []
        for vae_image in text_encoder_output["image_info"]["vae_image_list"]:
            image_encoder_output = self.run_vae_encoder(image=vae_image)
            image_encoder_output_list.append(image_encoder_output)
        torch_device_module.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output_list,
        }

    @ProfilingContext4DebugL1("Run Text Encoder", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_run_text_encode_duration, metrics_labels=["QwenImageRunner"])
    def run_text_encoder(self, text, image_list=None, neg_prompt=None):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_input_prompt_len.observe(len(text))
        text_encoder_output = {}
        if self.config["task"] == "t2i":
            prompt_embeds, _, _ = self.text_encoders[0].infer([text])
            self.input_info.txt_seq_lens = [prompt_embeds.shape[1]]
            text_encoder_output["prompt_embeds"] = prompt_embeds
            if self.config["enable_cfg"] and neg_prompt is not None:
                neg_prompt_embeds, _, _ = self.text_encoders[0].infer([neg_prompt])
                self.input_info.txt_seq_lens.append(neg_prompt_embeds.shape[1])
                text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds
        elif self.config["task"] == "i2i":
            prompt_embeds, _, image_info = self.text_encoders[0].infer([text], image_list)
            self.input_info.txt_seq_lens = [prompt_embeds.shape[1]]
            text_encoder_output["prompt_embeds"] = prompt_embeds
            text_encoder_output["image_info"] = image_info
            if self.config["enable_cfg"] and neg_prompt is not None:
                neg_prompt_embeds, _, _ = self.text_encoders[0].infer([neg_prompt], image_list)
                self.input_info.txt_seq_lens.append(neg_prompt_embeds.shape[1])
                text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds
        
        # 多 GPU 模式下，将 text encoder 输出移动到 DiT 设备
        if self.dit_device is not None:
            text_encoder_output = self._move_to_device(text_encoder_output, self.dit_device)
            logger.debug(f"Text Encoder 输出已传输到 DiT 设备: {self.dit_device}")
        
        return text_encoder_output
    
    def _move_to_device(self, obj, device):
        """递归地将 tensor 或包含 tensor 的字典/列表移动到指定设备"""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._move_to_device(item, device) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._move_to_device(item, device) for item in obj)
        else:
            return obj

    @ProfilingContext4DebugL1("Run VAE Encoder", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration, metrics_labels=["QwenImageRunner"])
    def run_vae_encoder(self, image):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae = self.load_vae()
        
        # 多 GPU 模式下，将图像移动到 VAE 设备
        if self.vae_device is not None:
            image = image.to(self.vae_device)
        
        image_latents = self.vae.encode_vae_image(image.to(GET_DTYPE()))
        
        # 多 GPU 模式下，将 VAE 输出移动到 DiT 设备
        if self.dit_device is not None:
            image_latents = image_latents.to(self.dit_device)
            logger.debug(f"VAE Encoder 输出已传输到 DiT 设备: {self.dit_device}")
        
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae
            torch_device_module.empty_cache()
            gc.collect()
        return {"image_latents": image_latents}

    @ProfilingContext4DebugL1(
        "Run VAE Decoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_decode_duration,
        metrics_labels=["QwenImageRunner"],
    )
    def run_vae_decoder(self, latents):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae = self.load_vae()
        
        # 多 GPU 模式下，将 latents 从 DiT 设备移动到 VAE 设备
        if self.vae_device is not None:
            latents = latents.to(self.vae_device)
            logger.debug(f"Latents 已传输到 VAE 设备: {self.vae_device}")
        
        images = self.vae.decode(latents, self.input_info)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae
            torch_device_module.empty_cache()
            gc.collect()
        return images

    def run(self, total_steps=None):
        if total_steps is None:
            total_steps = self.model.scheduler.infer_steps
        for step_index in range(total_steps):
            logger.info(f"==> step_index: {step_index + 1} / {total_steps}")

            with ProfilingContext4DebugL1("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)

            with ProfilingContext4DebugL1("🚀 infer_main"):
                self.model.infer(self.inputs)

            with ProfilingContext4DebugL1("step_post"):
                self.model.scheduler.step_post()

            if self.progress_callback:
                self.progress_callback(((step_index + 1) / total_steps) * 100, 100)

        return self.model.scheduler.latents, self.model.scheduler.generator

    def get_custom_shape(self):
        default_aspect_ratios = {
            "16:9": [1664, 928],
            "9:16": [928, 1664],
            "1:1": [1328, 1328],
            "4:3": [1472, 1140],
            "3:4": [768, 1024],
        }
        as_maps = self.config.get("aspect_ratios", {})
        as_maps.update(default_aspect_ratios)
        max_size = self.config.get("max_custom_size", 1664)
        min_size = self.config.get("min_custom_size", 256)

        if len(self.input_info.target_shape) == 2:
            height, width = self.input_info.target_shape
            height, width = int(height), int(width)
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                width, height = int(width * scale), int(height * scale)
                logger.warning(f"Custom shape is too large, scaled to {width}x{height}")
            width, height = max(width, min_size), max(height, min_size)
            logger.info(f"Qwen Image Runner got custom shape: {width}x{height}")
            return (width, height)

        aspect_ratio = self.input_info.aspect_ratio if self.input_info.aspect_ratio else self.config.get("aspect_ratio", None)
        if aspect_ratio in as_maps:
            logger.info(f"Qwen Image Runner got aspect ratio: {aspect_ratio}")
            width, height = as_maps[aspect_ratio]
            return (width, height)
        logger.warning(f"Invalid aspect ratio: {aspect_ratio}, not in {as_maps.keys()}")

        return None

    def set_target_shape(self):
        # custom_shape = self.get_custom_shape()
        # if custom_shape is not None:
        #     width, height = custom_shape
        # else:
        width, height = self.input_info.original_size[0]
        calculated_width, calculated_height, _ = calculate_dimensions(self.resolution * self.resolution, width / height)
        multiple_of = self.config["vae_scale_factor"] * 2
        width = calculated_width // multiple_of * multiple_of
        height = calculated_height // multiple_of * multiple_of
        logger.info(f"Qwen Image Runner set target shape: {width}x{height}")
        self.input_info.auto_width = width
        self.input_info.auto_height = height

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.config["vae_scale_factor"] * 2))
        width = 2 * (int(width) // (self.config["vae_scale_factor"] * 2))
        num_channels_latents = self.config["in_channels"] // 4
        if not self.is_layered:
            self.input_info.target_shape = (1, 1, num_channels_latents, height, width)
        else:
            self.input_info.target_shape = (1, self.layers + 1, num_channels_latents, height, width)

    def set_img_shapes(self):
        width, height = self.input_info.auto_width, self.input_info.auto_height
        if self.config["task"] == "t2i":
            image_shapes = [(1, height // self.config["vae_scale_factor"] // 2, width // self.config["vae_scale_factor"] // 2)] * 1
        elif self.config["task"] == "i2i":
            if self.is_layered:
                image_shapes = [
                    [
                        *[(1, height // self.config["vae_scale_factor"] // 2, width // self.config["vae_scale_factor"] // 2) for _ in range(self.layers + 1)],
                        (1, height // self.config["vae_scale_factor"] // 2, width // self.config["vae_scale_factor"] // 2),
                    ]
                ]
            else:
                image_shapes = [[(1, height // self.config["vae_scale_factor"] // 2, width // self.config["vae_scale_factor"] // 2)]]
                for image_height, image_width in self.inputs["text_encoder_output"]["image_info"]["vae_image_info_list"]:
                    image_shapes[0].append((1, image_height // self.config["vae_scale_factor"] // 2, image_width // self.config["vae_scale_factor"] // 2))
        self.input_info.image_shapes = image_shapes

    def init_scheduler(self):
        scheduler_config = self.config.copy()
        if self.dit_device is not None:
            scheduler_config["dit_device"] = str(self.dit_device)
        self.scheduler = QwenImageScheduler(scheduler_config)

    def get_encoder_output_i2v(self):
        pass

    def run_image_encoder(self):
        pass

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.image_encoder = self.load_image_encoder()
        self.vae = self.load_vae()
        self.vfi_model = self.load_vfi_model() if "video_frame_interpolation" in self.config else None

    @ProfilingContext4DebugL1("RUN pipeline")
    def run_pipeline(self, input_info):
        self.input_info = input_info

        self.inputs = self.run_input_encoder()
        self.set_target_shape()
        self.set_img_shapes()
        logger.info(f"input_info: {self.input_info}")
        latents, generator = self.run_dit()
        images = self.run_vae_decoder(latents)
        self.end_run()

        if not dist.is_initialized() or dist.get_rank() == 0:
            if not input_info.return_result_tensor and input_info.save_result_path is not None:
                save_path = input_info.save_result_path
                # 若 save_path 是目录（以 / 结尾或本身是已存在目录），则在其下生成默认文件名
                if os.path.isdir(save_path) or save_path.endswith("/") or save_path.endswith(os.sep):
                    image_prefix = os.path.join(save_path, "output")
                    image_suffix = "png"
                else:
                    parts = save_path.rsplit(".", 1)
                    image_prefix = parts[0]
                    image_suffix = parts[1] if len(parts) > 1 else "png"
                if isinstance(images[0], list) and len(images[0]) > 1:
                    for idx, image in enumerate(images[0]):
                        image.save(f"{image_prefix}_{idx:05d}.{image_suffix}")
                        logger.info(f"Image saved: {image_prefix}_{idx:05d}.{image_suffix}")
                else:
                    image = images[0]
                    image.save(f"{image_prefix}.{image_suffix}")
                    logger.info(f"Image saved: {image_prefix}.{image_suffix}")

        del latents, generator
        torch_device_module.empty_cache()
        gc.collect()

        if input_info.return_result_tensor:
            return {"images": images}  # pt tensors
        else:
            return {"images": images}  # PIL images (saved to disk if save_result_path was set)
