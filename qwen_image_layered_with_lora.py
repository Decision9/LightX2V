"""
Qwen-Image Layered 模式 + LoRA 推理示例

Layered 模式：输入一张 RGBA 图像，模型输出按层分解的多层图像结果（layers 层）。
LoRA：在 Layered 模型基础上叠加风格/任务 LoRA 权重。
"""

from lightx2v import LightX2VPipeline
from PIL import Image
pipe = LightX2VPipeline(
    model_path="/mnt/workspace/hf_cache_root/hub/modelscope/models/Qwen/Qwen-Image-Layered",
    model_cls="qwen_image",
    task="i2i",
)

# （可选）CPU Offload，适合显存不足场景
# pipe.enable_offload(
#     cpu_offload=True,
#     offload_granularity="block",  # 或 "phase"
#     text_encoder_offload=True,
#     vae_offload=False,
# )

pipe.enable_lora(
    lora_configs=[
        {"path": "/home/lilonghao/model/DiffSynth-Studio/models/train/Qwen-Image-Layered_lora_wuduan+2dcomics_bk-3_p/epoch-8.safetensors", "strength": 1.0},
        # 可以叠加多个 LoRA
        # {"path": "/path/to/another_lora.safetensors", "strength": 0.8},
    ],
    lora_dynamic_apply=False,
)

# （可选）多 GPU 部署：将 Text Encoder / VAE 与 DiT 分离到不同 GPU，不启动的时候默认全在一个 GPU 上
# 取消下面的注释以启用多 GPU 模式，适合单卡显存不足但有多卡的场景,要求dit_device必须是cuda:0,但可以通过指定CUDA_VISIBLE_DEVICES的顺序来改变cuda:0指向那一张显卡。
# 实例：CUDA_VISIBLE_DEVICES=6,7 python qwen_image_layered_with_lora.py，此时使用的是GPU6和GPU7，dit_device设置为cuda:0即指GPU6，text_encoder_device和vae_device设置为cuda:1即指GPU7
# 若CUDA_VISIBLE_DEVICES=7,6 python qwen_image_layered_with_lora.py，此时使用的是GPU7和GPU6，dit_device设置为cuda:0即指GPU7，text_encoder_device和vae_device设置为cuda:1即指GPU6
# pipe.multi_gpu_config = {
#     "text_encoder_device": "cuda:1",  # Text Encoder 部署的 GPU
#     "vae_device": "cuda:1",           # VAE 部署的 GPU
#     "dit_device": "cuda:0",           # DiT (Transformer) 部署的 GPU
# }

# Layered 模式专属参数（原 JSON 中的字段）
pipe.update({
    "layered": True,
    "layers": 6,
    "use_layer3d_rope": True,
    "use_en_prompt": True,
    "use_additional_t_cond": True,
    "USE_IMAGE_ID_IN_PROMPT": True,
    "resolution": 1024,
    "prompt_template_encode": (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "prompt_template_encode_start_idx": 34,
})

# h200用flash_attn3, A100, RTX 30/40系列用flash_attn2
pipe.create_generator(
    attn_mode="flash_attn3",
    rope_type="torch",
    infer_steps=20,
    guidance_scale=1,   # guidance_scale=1 → enable_cfg=False
    resize_mode="adaptive",
)

input_image = Image.open("/home/lilonghao/model/DiffSynth-Studio/scripts/[2019-10-27T07;49;10] 白髪たぴガール　原寸＋PSD2.png").convert("RGBA")

result = pipe.generate(
    seed=42,
    prompt="",           # Layered 模式会自动从输入图生成 caption，此处留空即可
    negative_prompt=" ",
    image_path=input_image,
    # return_result_tensor=True,  # 直接返回图片列表，不保存到磁盘
    # save_result_path="/home/lilonghao/model/docker/LightX2V/examples/qwen_image/tmp/50",  # 若需保存则取消注释
)
# result["images"][0] 是一个 PIL Image 列表（layered 模式下包含多层图片）
images = result["images"][0]
print(f"Got {len(images)} images")
import os; os.makedirs("./tmp", exist_ok=True)
for idx, img in enumerate(images):
    img.save(f"./tmp/output_{idx}.png")