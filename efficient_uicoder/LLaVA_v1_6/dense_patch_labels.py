import os
import math
import ast

import numpy as np
from shapely.geometry import box as shapely_box, LineString
from shapely.ops import unary_union
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from llava.mm_utils import (
    select_best_resolution,
    resize_and_pad_image,
    divide_to_patches,
)


def extract_dense_patch_labels(
    image,
    processor,
    grid_pinpoints,
    components=None,
    patch_size=14,
    line_width=3,
):
    """
    Build dense patch label maps for both resized full image and cropped patches.

    Labeling rules:
      - text regions → 0 (highest priority, cannot be overwritten)
      - compos outer boundary patches → 1 (cannot overwrite 0)
      - connection regions → 1 (cannot overwrite 0 or 1)
      - real background → -1
      - padding → -2 (should be fully masked out)

    Returns:
      dense_patch_labels_all: [np.ndarray(Hp, Wp), ...]
      image_list:             [PIL.Image, ...] aligned with above
    """
    if components is None:
        components = {}

    dense_patch_labels_all = []
    image_list = []
    orig_w, orig_h = image.size

    # ---- Crop components within a patch ----
    def crop_components_in_patch(components, x0, y0, x1, y1):
        def box_intersects(cx0, cy0, cx1, cy1, x0, y0, x1, y1):
            return (cx1 >= x0) and (cx0 <= x1) and (cy1 >= y0) and (cy0 <= y1)

        compos_new, texts_new, conns_new = [], [], []

        for c in components.get("compos", []):
            p = c["position"]
            if box_intersects(p["column_min"], p["row_min"], p["column_max"], p["row_max"], x0, y0, x1, y1):
                compos_new.append(c)

        for t in components.get("texts", []):
            p = t["position"]
            if box_intersects(p["column_min"], p["row_min"], p["column_max"], p["row_max"], x0, y0, x1, y1):
                texts_new.append(t)

        for conn in components.get("connections", []):
            ok = (
                (x0 <= conn["from"][0] <= x1 and y0 <= conn["from"][1] <= y1)
                or (x0 <= conn["to"][0] <= x1 and y0 <= conn["to"][1] <= y1)
            )
            if ok:
                conns_new.append(conn)

        return {"compos": compos_new, "texts": texts_new, "connections": conns_new}

    # ---- Core label computation ----
    def compute_dense_patch_labels(
        target_w,
        target_h,
        scale_x,
        scale_y,
        *,
        pad_offset_x=0,
        pad_offset_y=0,
        crop_offset_x=0,
        crop_offset_y=0,
        valid_w=None,
        valid_h=None,
        components=None,
    ):
        """
        Compute dense patch labels for a given resolution.

        Args:
            valid_w, valid_h: if not None, padding exists → padding patches labeled as -2
        """
        if components is None:
            components = {}

        patch_rows = target_h // patch_size
        patch_cols = target_w // patch_size
        dense_map = np.full((patch_rows, patch_cols), -1, dtype=np.int32)

        patch_polys = [
            shapely_box(
                j * patch_size,
                i * patch_size,
                (j + 1) * patch_size,
                (i + 1) * patch_size,
            )
            for i in range(patch_rows)
            for j in range(patch_cols)
        ]

        # [0] Padding → -2
        if valid_w is not None and valid_h is not None:
            valid_box = shapely_box(
                pad_offset_x - crop_offset_x,
                pad_offset_y - crop_offset_y,
                pad_offset_x + valid_w - crop_offset_x,
                pad_offset_y + valid_h - crop_offset_y,
            )
            for patch_idx, patch in enumerate(patch_polys):
                r = patch_idx // patch_cols
                c = patch_idx % patch_cols
                if not patch.intersects(valid_box):
                    dense_map[r, c] = -2

        # [1] Texts → 0
        for text in components.get("texts", []):
            pos = text["position"]
            x_min = pos["column_min"] * scale_x + pad_offset_x - crop_offset_x
            y_min = pos["row_min"] * scale_y + pad_offset_y - crop_offset_y
            x_max = pos["column_max"] * scale_x + pad_offset_x - crop_offset_x
            y_max = pos["row_max"] * scale_y + pad_offset_y - crop_offset_y
            text_poly = shapely_box(x_min, y_min, x_max, y_max)

            for patch_idx, patch in enumerate(patch_polys):
                r = patch_idx // patch_cols
                c = patch_idx % patch_cols
                if dense_map[r, c] == -2:  # skip padding
                    continue
                if patch.intersects(text_poly):
                    dense_map[r, c] = 0

        # [2] Compos outer boundary → 1
        for comp in components.get("compos", []):
            pos = comp["position"]
            x_min = pos["column_min"] * scale_x + pad_offset_x - crop_offset_x
            y_min = pos["row_min"] * scale_y + pad_offset_y - crop_offset_y
            x_max = pos["column_max"] * scale_x + pad_offset_x - crop_offset_x
            y_max = pos["row_max"] * scale_y + pad_offset_y - crop_offset_y
            comp_poly = shapely_box(x_min, y_min, x_max, y_max)

            covered = []
            for patch_idx, patch in enumerate(patch_polys):
                if patch.intersects(comp_poly):
                    covered.append(patch_idx)

            covered_set = set(covered)
            for patch_idx in covered:
                r = patch_idx // patch_cols
                c = patch_idx % patch_cols
                if dense_map[r, c] != -1:  # overwrite only background
                    continue
                neigh = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                is_border = False
                for nr, nc in neigh:
                    if not (0 <= nr < patch_rows and 0 <= nc < patch_cols):
                        is_border = True
                        break
                    n_idx = nr * patch_cols + nc
                    if n_idx not in covered_set:
                        is_border = True
                        break
                if is_border:
                    dense_map[r, c] = 1

        # [3] Connections → 1
        for conn in components.get("connections", []):
            x1 = conn["from"][0] * scale_x + pad_offset_x - crop_offset_x
            y1 = conn["from"][1] * scale_y + pad_offset_y - crop_offset_y
            x2 = conn["to"][0] * scale_x + pad_offset_x - crop_offset_x
            y2 = conn["to"][1] * scale_y + pad_offset_y - crop_offset_y
            line = LineString([(x1, y1), (x2, y2)])
            line_box = line.buffer(line_width / 2, cap_style=1)

            for patch_idx, patch in enumerate(patch_polys):
                r = patch_idx // patch_cols
                c = patch_idx % patch_cols
                if dense_map[r, c] in (0, 1, -2):
                    continue
                if patch.intersects(line_box):
                    dense_map[r, c] = 1

        return dense_map

    # ===== Stage 0: resized full image =====
    target_edge = processor.size["shortest_edge"]
    image_resized = image.resize((target_edge, target_edge))
    target_w, target_h = image_resized.size

    scale_x_0 = target_w / orig_w
    scale_y_0 = target_h / orig_h

    dense_map_0 = compute_dense_patch_labels(
        target_w, target_h, scale_x_0, scale_y_0, components=components
    )
    dense_patch_labels_all.append(dense_map_0)
    image_list.append(image_resized)

    # ===== Stage 1: resize + pad + crop =====
    if isinstance(grid_pinpoints, list):
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)

    best_resolution = select_best_resolution(image.size, possible_resolutions)

    image_padded = resize_and_pad_image(image, best_resolution)
    padded_w, padded_h = image_padded.size

    original_width, original_height = image.size
    target_width, target_height = best_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    offset_x_pad = (target_width - new_width) // 2
    offset_y_pad = (target_height - new_height) // 2
    scale_x_pad = new_width / original_width
    scale_y_pad = new_height / original_height

    patches = divide_to_patches(image_padded, processor.crop_size["height"])
    patch_size_crop = processor.crop_size["height"]
    num_cols = padded_w // patch_size_crop

    for idx, patch_img in enumerate(patches):
        col_idx = idx % num_cols
        row_idx = idx // num_cols
        crop_offset_x = col_idx * patch_size_crop
        crop_offset_y = row_idx * patch_size_crop

        # Compute crop bounding box in original coordinate system
        x0_crop = (crop_offset_x - offset_x_pad) / scale_x_pad
        y0_crop = (crop_offset_y - offset_y_pad) / scale_y_pad
        x1_crop = x0_crop + (patch_size_crop / scale_x_pad)
        y1_crop = y0_crop + (patch_size_crop / scale_y_pad)
        cropped_components = crop_components_in_patch(
            components, x0_crop, y0_crop, x1_crop, y1_crop
        )

        dense_map = compute_dense_patch_labels(
            patch_size_crop,
            patch_size_crop,
            scale_x_pad,
            scale_y_pad,
            pad_offset_x=offset_x_pad,
            pad_offset_y=offset_y_pad,
            crop_offset_x=crop_offset_x,
            crop_offset_y=crop_offset_y,
            valid_w=new_width,
            valid_h=new_height,
            components=cropped_components,
        )

        dense_patch_labels_all.append(dense_map)
        image_list.append(patch_img)

    return dense_patch_labels_all, image_list


def visualize_dense_patch_labels(
    image_list,
    dense_patch_labels_batch,
    save_dir,
    patch_size=14,
    figsize=(10, 10)
):
    """
    Visualize dense patch label maps for a batch of images.

    Args:
        image_list (list of PIL.Image.Image): images corresponding to label maps.
        dense_patch_labels_batch (list of np.ndarray): dense patch label maps.
        save_dir (str): directory to save visualization images.
        patch_size (int): patch size in pixels.
        figsize (tuple): figure size for display.
        alpha (int): transparency of overlay rectangles (0–255).
    """
    os.makedirs(save_dir, exist_ok=True)

    for idx, (image, label_map) in enumerate(zip(image_list, dense_patch_labels_batch)):
        vis_image = image.convert("RGBA")
        overlay = Image.new("RGBA", vis_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        patch_rows, patch_cols = label_map.shape

        for row in range(patch_rows):
            for col in range(patch_cols):
                label = label_map[row, col]
                x0 = col * patch_size
                y0 = row * patch_size
                x1 = x0 + patch_size
                y1 = y0 + patch_size

                if label in (-1, -2):
                    continue
                elif label == 0:
                    draw.rectangle([x0, y0, x1, y1], outline="red", width=1)
                else:
                    draw.rectangle([x0, y0, x1, y1], outline="red", width=1)

        vis_image = Image.alpha_composite(vis_image, overlay).convert("RGB")

        save_path = os.path.join(save_dir, f"patch_labels_{idx}.png")
        vis_image.save(save_path)

        plt.figure(figsize=figsize)
        plt.imshow(vis_image)
        plt.title(f"patch_labels[{idx}]")
        plt.axis("off")
        plt.show()
        plt.close()

        print(f"[Visualize] Saved patch_labels[{idx}] visualization to {save_path}")
