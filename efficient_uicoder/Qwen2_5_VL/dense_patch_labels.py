import os
import numpy as np
from shapely.geometry import box as shapely_box, LineString
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from qwen_vl_utils import smart_resize

def extract_dense_patch_labels(image, components=None,
                               patch_size=28,
                               line_width=3):
    """
    Generate a dense patch label map.

    Rules:
        1. Text regions → label = 0 (highest priority, cannot be overwritten)
        2. Outer contour of each component → label = 1 (cannot overwrite label=0)
        3. Connection lines → label = 1 (cannot overwrite label=0 or 1)
        4. Other regions remain -1
    """
    orig_w, orig_h = image.size
    resized_h, resized_w = smart_resize(orig_h, orig_w)
    scale_x = resized_w / orig_w
    scale_y = resized_h / orig_h

    # Step 0: Generate patch grid
    patch_rows = resized_h // patch_size
    patch_cols = resized_w // patch_size
    dense_map = np.full((patch_rows, patch_cols), -1, dtype=np.int32)

    patch_polys = [
        shapely_box(j * patch_size, i * patch_size,
                    (j + 1) * patch_size, (i + 1) * patch_size)
        for i in range(patch_rows) for j in range(patch_cols)
    ]

    # Step 1: Mark text regions as 0
    texts = components.get("texts", [])
    for text in texts:
        pos = text["position"]
        x_min = pos["column_min"] * scale_x
        y_min = pos["row_min"] * scale_y
        x_max = pos["column_max"] * scale_x
        y_max = pos["row_max"] * scale_y
        text_poly = shapely_box(x_min, y_min, x_max, y_max)

        for patch_idx, patch in enumerate(patch_polys):
            if patch.intersects(text_poly):
                row = patch_idx // patch_cols
                col = patch_idx % patch_cols
                dense_map[row, col] = 0  # Texts have the highest priority

    # Step 2: Mark outer contour of components as 1 (do not overwrite 0)
    compos = components.get("compos", [])
    for comp in compos:
        pos = comp["position"]
        x_min = pos["column_min"] * scale_x
        y_min = pos["row_min"] * scale_y
        x_max = pos["column_max"] * scale_x
        y_max = pos["row_max"] * scale_y
        comp_poly = shapely_box(x_min, y_min, x_max, y_max)

        # Find all patches intersecting with the component
        covered_patches = [
            patch_idx for patch_idx, patch in enumerate(patch_polys)
            if patch.intersects(comp_poly)
        ]
        covered_set = set(covered_patches)

        # Select border patches
        for patch_idx in covered_patches:
            row = patch_idx // patch_cols
            col = patch_idx % patch_cols

            neighbors = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1)
            ]
            is_border = False
            for nr, nc in neighbors:
                # Out-of-bound neighbors → border
                if not (0 <= nr < patch_rows and 0 <= nc < patch_cols):
                    is_border = True
                    break
                # In-bound neighbor not covered by component → border
                n_idx = nr * patch_cols + nc
                if n_idx not in covered_set:
                    is_border = True
                    break

            if is_border and dense_map[row, col] == -1:
                dense_map[row, col] = 1

    # Step 3: Mark connections as 1 (do not overwrite 0 or 1)
    connections = components.get("connections", [])
    for conn in connections:
        x1 = conn["from"][0] * scale_x
        y1 = conn["from"][1] * scale_y
        x2 = conn["to"][0] * scale_x
        y2 = conn["to"][1] * scale_y
        line = LineString([(x1, y1), (x2, y2)])
        line_box = line.buffer(line_width / 2, cap_style=1)

        for patch_idx, patch in enumerate(patch_polys):
            row = patch_idx // patch_cols
            col = patch_idx % patch_cols
            if dense_map[row, col] != -1:
                continue  # Do not overwrite
            if patch.intersects(line_box):
                dense_map[row, col] = 1

    return dense_map


def visualize_dense_patch_labels(image, dense_patch_labels,
                                 save_dir=None,
                                 patch_size=28,
                                 figsize=(10, 10)
                                ):
    """
    Visualize dense patch label map for a single resized image.

    Args:
        image (PIL.Image.Image): Resized image corresponding to dense patch labels.
        dense_patch_labels (np.ndarray): 2D array of patch labels (-1, 0, 1, 2, ...).
        save_dir (str): Directory to save the visualization image.
        patch_size (int): Patch size in pixels.
        figsize (tuple): Figure size for display.
        alpha (int): Transparency of overlay rectangles (0–255).
    """
    os.makedirs(save_dir, exist_ok=True)

    vis_image = image.convert('RGBA')  # Enable alpha overlay
    overlay = Image.new('RGBA', vis_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    patch_rows, patch_cols = dense_patch_labels.shape

    for row in range(patch_rows):
        for col in range(patch_cols):
            label = dense_patch_labels[row, col]
            x0 = col * patch_size
            y0 = row * patch_size
            x1 = x0 + patch_size
            y1 = y0 + patch_size

            if label == -1:
                continue
            elif label == 0:
                # Text or connection → red outline
                draw.rectangle([x0, y0, x1, y1], outline='red', width=1)
            else:
                # Component group → red outline
                draw.rectangle([x0, y0, x1, y1], outline='red', width=1)

    # Overlay patches on the original image
    vis_image = Image.alpha_composite(vis_image, overlay).convert('RGB')

    save_path = os.path.join(save_dir, "dense_patch_labels.png")
    vis_image.save(save_path)

    plt.figure(figsize=figsize)
    plt.imshow(vis_image)
    plt.title("Dense Patch Labels")
    plt.axis('off')
    plt.show()
    plt.close()

    print(f"[Visualize] Saved dense_patch_labels visualization to {save_path}")
