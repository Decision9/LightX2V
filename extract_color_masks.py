"""
Fast connected-region colour quantization.

Adjacent pixels are "similar" when |ΔR|, |ΔG|, |ΔB| are all ≤ threshold.
Uses vectorised numpy + scipy sparse connected-components — no Python loops
per pixel, orders of magnitude faster than BFS on real images.
"""

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import List, Union


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_rgba(image: Union[str, np.ndarray]):
    """Return (H×W×3 uint8 RGB, H×W uint8 alpha, has_alpha bool)."""
    if isinstance(image, str):
        raw = Image.open(image)
        has_alpha = raw.mode in ("RGBA", "LA", "PA")
        rgba = np.array(raw.convert("RGBA"), dtype=np.uint8)
    else:
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            has_alpha = True
            rgba = arr
        else:
            has_alpha = False
            rgba = np.concatenate(
                [arr, np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)], axis=2
            )
    return rgba[:, :, :3], rgba[:, :, 3], has_alpha


def _connected_labels(
    img: np.ndarray,
    alpha: np.ndarray,
    has_alpha: bool,
    threshold: int,
    connectivity: int,
    alpha_min: int,
):
    """
    Build a sparse adjacency graph from per-channel difference masks and run
    scipy connected_components.  All pixel comparisons are vectorised.

    Returns (labels H×W int32, n_components int).
    Invalid pixels (alpha < alpha_min) get label -1.
    """
    H, W = img.shape[:2]
    N = H * W
    idx = np.arange(N, dtype=np.int32).reshape(H, W)

    valid = (alpha >= alpha_min) if has_alpha else np.ones((H, W), dtype=bool)

    all_rows, all_cols = [], []

    # --- horizontal neighbours (left <-> right) ---
    diff = np.abs(img[:, :-1].astype(np.int16) - img[:, 1:].astype(np.int16))
    keep = (
        (diff[..., 0] <= threshold) &
        (diff[..., 1] <= threshold) &
        (diff[..., 2] <= threshold) &
        valid[:, :-1] & valid[:, 1:]
    )
    r, c = idx[:, :-1][keep], idx[:, 1:][keep]
    all_rows += [r, c]; all_cols += [c, r]

    # --- vertical neighbours (top <-> bottom) ---
    diff = np.abs(img[:-1, :].astype(np.int16) - img[1:, :].astype(np.int16))
    keep = (
        (diff[..., 0] <= threshold) &
        (diff[..., 1] <= threshold) &
        (diff[..., 2] <= threshold) &
        valid[:-1, :] & valid[1:, :]
    )
    r, c = idx[:-1, :][keep], idx[1:, :][keep]
    all_rows += [r, c]; all_cols += [c, r]

    # --- diagonal neighbours (8-connectivity only) ---
    if connectivity == 8:
        for dr, dc in ((-1, -1), (-1, 1)):
            s0 = slice(max(0,  dr), H + min(0,  dr))
            s1 = slice(max(0,  dc), W + min(0,  dc))
            d0 = slice(max(0, -dr), H + min(0, -dr))
            d1 = slice(max(0, -dc), W + min(0, -dc))
            diff = np.abs(img[s0, s1].astype(np.int16) - img[d0, d1].astype(np.int16))
            keep = (
                (diff[..., 0] <= threshold) &
                (diff[..., 1] <= threshold) &
                (diff[..., 2] <= threshold) &
                valid[s0, s1] & valid[d0, d1]
            )
            r, c = idx[s0, s1][keep], idx[d0, d1][keep]
            all_rows += [r, c]; all_cols += [c, r]

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.bool_), (rows, cols)), shape=(N, N)
    )
    n_comp, labels_flat = connected_components(graph, directed=False)
    labels = labels_flat.reshape(H, W)
    labels[~valid] = -1
    return labels, n_comp


def extract_similar_color_masks(
    image: Union[str, np.ndarray],
    threshold: int = 4,
    connectivity: int = 4,
    min_pixels: int = 100,
    alpha_min: int = 20,
) -> List[np.ndarray]:
    """
    Extract boolean masks for connected regions of similar colour.

    Args:
        image:        File path or (H, W, 3/4) uint8 ndarray.
        threshold:    Max per-channel difference between adjacent pixels (default 4).
        connectivity: 4 or 8 (default 4).
        min_pixels:   Minimum region size to keep (default 100).
        alpha_min:    Pixels with alpha < this are excluded (default 20).

    Returns:
        List of (H, W) bool ndarrays.
    """
    img, alpha, has_alpha = _load_rgba(image)
    labels, n_comp = _connected_labels(
        img, alpha, has_alpha, threshold, connectivity, alpha_min
    )
    lbl_valid = labels[labels >= 0]
    counts = np.bincount(lbl_valid, minlength=n_comp)
    return [labels == lbl for lbl in range(n_comp) if counts[lbl] >= min_pixels]


# ---------------------------------------------------------------------------
# Optional helpers
# ---------------------------------------------------------------------------

def flatten_regions(
    image: Union[str, np.ndarray],
    masks: List[np.ndarray],
    alpha_min: int = 20,
) -> np.ndarray:
    """
    Replace each mask region's pixels with their mean RGB colour.
    Alpha channel is preserved. Pixels with alpha < alpha_min are set to black.
    """
    img, alpha, has_alpha = _load_rgba(image)
    H, W = img.shape[:2]
    N = H * W

    if has_alpha:
        result = np.concatenate([img, alpha[:, :, None]], axis=2).copy()
    else:
        result = img.copy()

    result_flat = result.reshape(N, -1)
    img_flat = img.reshape(N, 3).astype(np.float32)

    for mask in masks:
        mf = mask.ravel()
        mean_color = img_flat[mf].mean(axis=0).round().astype(np.uint8)
        result_flat[mf, :3] = mean_color

    if has_alpha:
        result_flat[alpha.ravel() < alpha_min, :3] = 0

    return result.reshape(H, W, -1)


def quantize_image(
    image: Union[str, np.ndarray],
    threshold: int = 4,
    connectivity: int = 4,
    min_pixels: int = 100,
    alpha_min: int = 20,
) -> np.ndarray:
    """
    One-step vectorised quantization: every pixel in a large-enough connected
    region is replaced with that region's mean RGB colour.

    Pixels with alpha < alpha_min are skipped during grouping and set to black
    in the output. Alpha channel is preserved unchanged.

    Args:
        image:        File path or (H, W, 3/4) uint8 ndarray.
        threshold:    Max per-channel diff between neighbours (default 4).
        connectivity: 4 or 8 (default 4).
        min_pixels:   Regions smaller than this are left unchanged (default 100).
        alpha_min:    Pixels with alpha < this are set to black (default 20).

    Returns:
        uint8 ndarray, same shape as input.
    """
    img, alpha, has_alpha = _load_rgba(image)
    H, W = img.shape[:2]
    N = H * W

    labels, n_comp = _connected_labels(
        img, alpha, has_alpha, threshold, connectivity, alpha_min
    )
    labels_flat = labels.ravel()          # (N,)
    valid = labels_flat >= 0              # pixels that belong to a component
    lbl_valid = labels_flat[valid]        # component ids for valid pixels

    # Pixel count per component
    counts = np.bincount(lbl_valid, minlength=n_comp)   # (n_comp,)
    large = counts >= min_pixels                         # (n_comp,) bool

    # Mean RGB per component — fully vectorised with bincount
    img_flat = img.reshape(N, 3).astype(np.float32)
    mean_rgb = np.zeros((n_comp, 3), dtype=np.float32)
    for ch in range(3):
        s = np.bincount(lbl_valid, weights=img_flat[valid, ch], minlength=n_comp)
        mean_rgb[:, ch] = s / np.maximum(counts, 1)
    mean_rgb = mean_rgb.round().astype(np.uint8)   # (n_comp, 3)

    # Build output array
    if has_alpha:
        result = np.concatenate([img, alpha[:, :, None]], axis=2).reshape(N, 4).copy()
    else:
        result = img.reshape(N, 3).copy()

    # Apply mean colour to large regions (single fancy-index write)
    apply = valid & large[labels_flat]
    result[apply, :3] = mean_rgb[labels_flat[apply]]

    # Black out low-alpha pixels
    if has_alpha:
        result[alpha.ravel() < alpha_min, :3] = 0

    return result.reshape(H, W, -1)


def masks_to_colored_image(masks: List[np.ndarray], seed: int = 42) -> np.ndarray:
    """Render each mask with a distinct random colour for quick visualisation."""
    rng = np.random.default_rng(seed)
    H, W = masks[0].shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for mask in masks:
        colour = rng.integers(0, 256, size=3, dtype=np.uint8)
        canvas[mask] = colour
    return canvas


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    import time

    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if image_path is None:
        test_img = np.array([
            [[200, 100,  50, 255], [202, 101,  51, 255], [  0,   0,   0,   5]],
            [[201,  99,  52, 255], [203, 102,  50, 255], [  0,   0,   0,   5]],
            [[  0,   0,   0,   5], [  0,   0,   0,   5], [  0,   0,   0,   5]],
        ], dtype=np.uint8)
        masks = extract_similar_color_masks(test_img, threshold=4)
        print(f"Masks from synthetic image: {len(masks)}")
        for i, m in enumerate(masks):
            print(f"  Mask {i}: {int(m.sum())} pixels")
    else:
        t0 = time.time()
        masks = extract_similar_color_masks(image_path, threshold=4)
        t1 = time.time()
        print(f"Image  : {image_path}")
        print(f"Masks  : {len(masks)}  ({t1 - t0:.2f}s)")
        if masks:
            sizes = [int(m.sum()) for m in masks]
            print(f"Sizes  : min={min(sizes)}, max={max(sizes)}, "
                  f"median={int(np.median(sizes))}")

        out_path = os.path.splitext(image_path)[0] + "_masks_vis.png"
        vis = masks_to_colored_image(masks)
        Image.fromarray(vis).save(out_path)
        print(f"Vis    : {out_path}")

        t0 = time.time()
        flat = quantize_image(image_path, threshold=4)
        t1 = time.time()
        flat_path = os.path.splitext(image_path)[0] + "_flattened.png"
        mode = "RGBA" if flat.shape[2] == 4 else "RGB"
        Image.fromarray(flat, mode).save(flat_path)
        print(f"Flat   : {flat_path}  ({t1 - t0:.2f}s)")
