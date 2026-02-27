"""
QwenImagePipeline: 从 JSON 配置文件加载参数的推理封装类，支持 Layered 模式和普通 i2i 模式。

使用方式：
    pipeline = QwenImagePipeline("config.json")
    images = pipeline.generate(image_path=img, seed=42, prompt="")
"""

import json
import os
from PIL import Image
from lightx2v import LightX2VPipeline


class QwenImagePipeline:
    def __init__(self, cfg):
        """
        初始化并根据 JSON 配置文件构建推理 pipeline。

        JSON 配置文件结构见 qwen_image_layered_config.json。

        Args:
            cfg: 配置字典或 JSON 配置文件路径。
        """
        if isinstance(cfg, str):
            with open(cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        self.pipe = LightX2VPipeline(
            model_path=cfg["model_path"],
            model_cls=cfg.get("model_cls", "qwen_image"),
            task=cfg.get("task", "i2i"),
        )

        # LoRA
        lora_configs = cfg.get("lora_configs", [])
        if lora_configs:
            self.pipe.enable_lora(
                lora_configs=lora_configs,
                lora_dynamic_apply=cfg.get("lora_dynamic_apply", False),
            )

        # CPU Offload
        offload = cfg.get("offload", {})
        if offload:
            self.pipe.enable_offload(
                cpu_offload=offload.get("cpu_offload", False),
                offload_granularity=offload.get("offload_granularity", "block"),
                text_encoder_offload=offload.get("text_encoder_offload", False),
                vae_offload=offload.get("vae_offload", False),
            )

        # 多 GPU 配置
        multi_gpu = cfg.get("multi_gpu_config", {})
        if multi_gpu:
            self.pipe.multi_gpu_config = multi_gpu

        # 模型参数（layered、resolution 等）
        model_params = cfg.get("model_params", {})
        if model_params:
            self.pipe.update(model_params)

        # 生成器参数
        gen_params = cfg.get("generator_params", {})
        self.pipe.create_generator(
            attn_mode=gen_params.get("attn_mode", "flash_attn2"),
            rope_type=gen_params.get("rope_type", "torch"),
            infer_steps=gen_params.get("infer_steps", 20),
            guidance_scale=gen_params.get("guidance_scale", 1),
            resize_mode=gen_params.get("resize_mode", "adaptive"),
        )

    def generate(
        self,
        image_path,
        prompt: str = "",
        negative_prompt: str = " ",
        seed: int = 42,
    ):
        """
        运行推理，返回 PIL Image 列表。

        Args:
            image_path: 输入图像，支持：
                - 文件路径字符串（单张）
                - 逗号分隔的文件路径字符串（多张）
                - PIL.Image.Image 对象
                - PIL.Image.Image 列表
            prompt: 文本提示，Layered 模式留空会自动生成 caption。
            negative_prompt: 负向提示。
            seed: 随机种子。

        Returns:
            list[PIL.Image.Image]: 生成的图片列表。
                Layered 模式：result[0] 是包含多层图片的列表。
                普通模式：每个元素是一张 PIL Image。
        """
        result = self.pipe.generate(
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_path=image_path,
        )
        return result["images"]

if __name__ == "__main__":
    from config import qwen_image_2511_json, qwen_image_layered_json
    # pipeline = QwenImagePipeline(qwen_image_2511_json)
    # image_path = [
    # Image.open("/home/lilonghao/model/trush/output/gubao_fp_v1/46/古宝-046-6-线稿/lineart_crop.png"),
    # Image.open("/home/lilonghao/model/trush/output/gubao_fp_v1/46/古宝-046-6-线稿/ref_4.png"),
    # Image.open("/home/lilonghao/model/trush/output/gubao_fp_v1/46/古宝-046-6-线稿/ref_6.png"),
    # ]
    # images = pipeline.generate(
    #     image_path=image_path,
    #     prompt="Please keep the original lineart of Picture 1 unchanged, and colorize it based on the color information, lighting, shading, and material styles from Picture 2 and Picture 3.",
    #     negative_prompt="",
    #     seed=42,
    # )
    # for idx, img in enumerate(images):
    #     img.save(f"./tmp/output_2511_{idx}.png")

    images = [Image.open("/home/lilonghao/model/docker/LightX2V/tmp/output_5.png").convert("RGBA")]
    pipeline = QwenImagePipeline(qwen_image_layered_json)
    images = pipeline.generate(
        image_path=images,  # 上一步生成的图片作为输入
        prompt="",
        negative_prompt=" ",
        seed=42,
    )[0]  # Layered 模式下 result["images"][0] 是一个包含多层图片的列表
    for idx, img in enumerate(images):
        img.save(f"./tmp/output_{idx}.png")
