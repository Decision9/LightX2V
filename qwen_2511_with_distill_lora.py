"""
Qwen-image-edit image-to-image generation example.
This example demonstrates how to use LightX2V with Qwen-Image-Edit model for I2I generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Qwen-image-edit I2I task
# For Qwen-Image-Edit-2509, use model_cls="qwen-image-edit-2509"
pipe = LightX2VPipeline(
    model_path="/mnt/workspace/hf_cache_root/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
    model_cls="qwen-image-edit-2511",
    task="i2i",
)


# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="../configs/qwen_image/qwen_image_i2i_2511_lora.json"
# )

# Enable offloading to significantly reduce VRAM usage with minimal speed impact
# Suitable for RTX 30/40/50 consumer GPUs
# pipe.enable_offload(
#     cpu_offload=False,
#     offload_granularity="phase", #["block", "phase"]
#     text_encoder_offload=True,
#     vae_offload=False,
# )

# Load distilled LoRA weights
pipe.enable_lora(
    [
        # {"path": "/home/lilonghao/model/LightX2V/examples/qwen_image/weights/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength": 1.0},
        {"path": "/home/lilonghao/model/LightX2V/examples/qwen_image/weights/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors", "strength": 1.0},
        {"path": "/home/lilonghao/model/LightX2V/examples/qwen_image/weights/qwen_edit_2511_lora_16-000004_.safetensors", "strength": 1.0}
    ],
    lora_dynamic_apply=False,  # Support inference with LoRA weights, save memory but slower, default is False
)

# ========== 多 GPU 部署配置 ==========
# 配置不同组件部署到不同的 GPU 上
# 取消下面的注释以启用多 GPU 模式
# pipe.multi_gpu_config = {
#     "text_encoder_device": "cuda:1",  # Text Encoder 部署的 GPU
#     "vae_device": "cuda:1",           # VAE 部署的 GPU
#     "dit_device": "cuda:0",           # DiT (Transformer) 部署的 GPU
# }

pipe.create_generator(
    attn_mode="sage_attn2",
    resize_mode="adaptive",
    infer_steps=8,
    guidance_scale=1,
)

# Generation parameters
seed = 42
prompt = "Please keep the original lineart of Picture 1 unchanged, and colorize it based on the color information, lighting, shading, and material styles from Picture 2 and Picture 3."
negative_prompt = ""

# image_path 支持三种方式：
# 1. 逗号分隔的文件路径字符串（原有方式）
# image_path = "/path/to/img0.png,/path/to/img1.png,/path/to/img2.png"
# 2. PIL Image 对象（单张）
# image_path = Image.open("/path/to/img.png")
# 3. PIL Image 列表（多张）
from PIL import Image
image_path = [
    Image.open("/home/lilonghao/model/trush/output/gubao_fp_v1/46/古宝-046-6-线稿/lineart_crop.png"),
    Image.open("/home/lilonghao/model/trush/output/gubao_fp_v1/46/古宝-046-6-线稿/ref_4.png"),
    Image.open("/home/lilonghao/model/trush/output/gubao_fp_v1/46/古宝-046-6-线稿/ref_6.png"),
]

# time_ = time.time()
# save_result_path = f"./output/output_{int(time_)}.png"

# Generate，不传 save_result_path 则直接返回 PIL Image 列表
result = pipe.generate(
    seed=seed,
    image_path=image_path,
    prompt=prompt,
    negative_prompt=negative_prompt,
    # save_result_path=save_result_path,  # 传入则同时保存到磁盘
)
# result["images"] 结构：list of batch，每个 batch 是一张 PIL Image（非 layered 模式）
images = result["images"]
print(f"Got {len(images)} images")
import os; os.makedirs("./tmp", exist_ok=True)
for idx, img in enumerate(images):
    img.save(f"./tmp/output_{idx}.png")
