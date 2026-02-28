qwen_image_2511_json = {
    "model_path": "/mnt/workspace/hf_cache_root/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
    "model_cls": "qwen-image-edit-2511",
    "task": "i2i",

    # "multi_gpu_config": {
    #     "text_encoder_device": "cuda:1",
    #     "vae_device": "cuda:1",
    #     "dit_device": "cuda:0"
    # },

    # "offload": {
    #     "cpu_offload": True,
    #     "offload_granularity": "block",
    #     "text_encoder_offload": True,
    #     "vae_offload": False
    # },
    # "model_params": {
    #     "resolution": 1536,
    #     "vae_image_size": 1536*1536,
    # },

    "lora_configs": [
        {
            "path": "/home/lilonghao/model/LightX2V/examples/qwen_image/weights/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
            "strength": 1.0
        },
        {
            "path": "/home/lilonghao/model/LightX2V/examples/qwen_image/weights/qwen_edit_2511_lora_16-000004_.safetensors",
            "strength": 1.0
        }
    ],
    "lora_dynamic_apply": False,

    "generator_params": {
        "attn_mode": "sage_attn2",
        "infer_steps": 4,
        "guidance_scale": 1,
        "resize_mode": "adaptive"
    }
}

qwen_image_layered_json = {
    "model_path": "/mnt/workspace/hf_cache_root/hub/modelscope/models/Qwen/Qwen-Image-Layered",
    "model_cls": "qwen_image",
    "task": "i2i",

    # "multi_gpu_config": {
    #     "text_encoder_device": "cuda:1",
    #     "vae_device": "cuda:1",
    #     "dit_device": "cuda:0"
    # },

    # "offload": {
    #     "cpu_offload": True,
    #     "offload_granularity": "block",
    #     "text_encoder_offload": True,
    #     "vae_offload": False
    # },

    "lora_configs": [
        {
            "path": "/home/lilonghao/model/DiffSynth-Studio/models/train/Qwen-Image-Layered_lora_wuduan+2dcomics_bk-3_p/epoch-9.safetensors",
            "strength": 1.0
        }
    ],
    "lora_dynamic_apply": False,

    "model_params": {
        "layered": True,
        "layers": 6,
        "use_layer3d_rope": True,
        "use_en_prompt": True,
        "use_additional_t_cond": True,
        "USE_IMAGE_ID_IN_PROMPT": True,
        "resolution": 1024,
        "prompt_template_encode": "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        "prompt_template_encode_start_idx": 34
    },

    "generator_params": {
        "attn_mode": "sage_attn2",
        "rope_type": "torch",
        "infer_steps": 15,
        "guidance_scale": 1,
        "resize_mode": "adaptive"
    }
}
