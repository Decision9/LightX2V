"""
colorize_pipeline.py

端到端线稿上色流程：
  1. [qwen_image_2511] 线稿 + 多张参考图 → 上色图
  2. [qwen_image_layered] 上色图 → 分层图（多层 PIL Image）
  3. 对每层做纯色化（quantize_image），并缩放回原始分辨率

用法示例：
    from colorize_pipeline import colorize

    layers = colorize(
        lineart_path="lineart.png",
        ref_paths=["ref1.png", "ref2.png"],
        prompt="Please keep the original lineart...",
        seed=42,
    )
    for i, img in enumerate(layers):
        img.save(f"layer_{i}.png")
"""

from __future__ import annotations

import os
from typing import List, Union

import numpy as np
from PIL import Image

from config import qwen_image_2511_json, qwen_image_layered_json
from extract_color_masks import quantize_image
from qwen_image_pipeline import QwenImagePipeline


# ---------------------------------------------------------------------------
# Lazy pipeline singletons — loaded once, reused on repeated calls
# ---------------------------------------------------------------------------

_pipeline_2511: QwenImagePipeline | None = None
_pipeline_layered: QwenImagePipeline | None = None


def _get_pipeline_2511() -> QwenImagePipeline:
    global _pipeline_2511
    if _pipeline_2511 is None:
        _pipeline_2511 = QwenImagePipeline(qwen_image_2511_json)
    return _pipeline_2511


def _get_pipeline_layered() -> QwenImagePipeline:
    global _pipeline_layered
    if _pipeline_layered is None:
        _pipeline_layered = QwenImagePipeline(qwen_image_layered_json)
    return _pipeline_layered


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def colorize(
    lineart_path: Union[str, Image.Image],
    ref_paths: List[Union[str, Image.Image]],
    prompt: str = (
        "Please keep the original lineart of Picture 1 unchanged, "
        "and colorize it based on the color information, lighting, "
        "shading, and material styles from the reference pictures."
    ),
    negative_prompt: str = "",
    seed: int = 42,
    quantize_threshold: int = 4,
    quantize_connectivity: int = 4,
    quantize_min_pixels: int = 100,
    flat_color: tuple = (255, 255, 255),
) -> List[Image.Image]:
    """
    端到端线稿上色：线稿 + 参考图 → 分层纯色图（原始分辨率）。

    Args:
        lineart_path:          线稿图片，文件路径或 PIL Image。
        ref_paths:             参考图列表，每项可为路径或 PIL Image。
        prompt:                传给 qwen_image_2511 的文本提示。
        negative_prompt:       负向提示。
        seed:                  随机种子。
        quantize_threshold:    纯色化：相邻像素最大通道差（默认 4）。
        quantize_connectivity: 纯色化：连通性，4 或 8（默认 4）。
        flat_color:            索引 1 图层中 alpha > 20 的像素统一替换为该 RGB，默认 (255, 255, 255)。
        quantize_min_pixels:   纯色化：最小区域像素数（默认 100）。

    Returns:
        List[PIL.Image.Image]：各层纯色化后、缩放至原始分辨率的图片列表。
    """
    # ------------------------------------------------------------------
    # 0. 加载线稿并记录原始尺寸
    # ------------------------------------------------------------------
    if isinstance(lineart_path, str):
        lineart = Image.open(lineart_path)
    else:
        lineart = lineart_path
    original_size = lineart.size  # (W, H)

    # ------------------------------------------------------------------
    # 1. 组装输入图片列表：[线稿, ref1, ref2, ...]
    # ------------------------------------------------------------------
    input_images: List[Image.Image] = [lineart]
    for ref in ref_paths:
        if isinstance(ref, str):
            input_images.append(Image.open(ref))
        else:
            input_images.append(ref)

    # ------------------------------------------------------------------
    # 2. qwen_image_2511：上色
    # ------------------------------------------------------------------
    pipe_2511 = _get_pipeline_2511()
    colored_images = pipe_2511.generate(
        image_path=input_images,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
    )
    # 取第一张作为上色结果，转 RGBA 供后续处理
    colored: Image.Image = colored_images[0].convert("RGBA")
    colored.save("debug_colored.png")  # 调试：保存上色结果
    # ------------------------------------------------------------------
    # 3. qwen_image_layered：分层提取
    # ------------------------------------------------------------------
    pipe_layered = _get_pipeline_layered()
    layered_result = pipe_layered.generate(
        image_path=[colored],
        prompt="",
        negative_prompt=" ",
        seed=seed,
    )
    # layered_result[0] 是包含多层 PIL Image 的列表
    layers: List[Image.Image] = layered_result[0]

    # ------------------------------------------------------------------
    # 4. 对每层做处理，并缩放回原始分辨率
    #    - 索引 1：alpha > 20 的像素全部替换为 flat_color（单一 RGB）
    #    - 其余层：quantize_image 纯色化
    # ------------------------------------------------------------------
    output_images: List[Image.Image] = []
    for idx, layer in enumerate(layers):
        arr = np.array(layer.convert("RGBA"), dtype=np.uint8)  # H×W×4

        if idx == 1:
            # alpha > 20 的像素：RGB 统一设为 flat_color；alpha <= 20 的像素：RGBA 全清零
            result = arr.copy()
            high_alpha = result[:, :, 3] > 20
            low_alpha = ~high_alpha
            result[high_alpha, 0] = flat_color[0]
            result[high_alpha, 1] = flat_color[1]
            result[high_alpha, 2] = flat_color[2]
            result[low_alpha, :] = 0
            flat_img = Image.fromarray(result, mode="RGBA")
        else:
            # 普通纯色化
            flat_arr = quantize_image(
                arr,
                threshold=quantize_threshold,
                connectivity=quantize_connectivity,
                min_pixels=quantize_min_pixels,
            )
            flat_img = Image.fromarray(flat_arr, mode="RGBA" if flat_arr.shape[2] == 4 else "RGB")

        # 缩放回原始分辨率（LANCZOS 保持边缘锐利）
        if flat_img.size != original_size:
            flat_img = flat_img.resize(original_size, Image.LANCZOS)

        output_images.append(flat_img)

    return output_images


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lineart = Image.open("/home/lilonghao/model/docker/LightX2V/examples/qwen_image/output/concatenate/20260225_043457/ref_1771965296_0.png")
    refs = [
        Image.open("/home/lilonghao/model/docker/LightX2V/examples/qwen_image/output/concatenate/20260225_043457/ref_1771965296_1.png"),
        Image.open("/home/lilonghao/model/docker/LightX2V/examples/qwen_image/output/concatenate/20260225_043457/ref_1771965296_2.png"),
    ]

    out_dir = "./tmp"
    os.makedirs(out_dir, exist_ok=True)

    layers = colorize(
        lineart_path=lineart,
        ref_paths=refs,
        seed=42,
    )

    for idx, img in enumerate(layers):
        out_path = os.path.join(out_dir, f"layer_{idx}.png")
        img.save(out_path)
        print(f"已保存: {out_path}  size={img.size}  mode={img.mode}")
