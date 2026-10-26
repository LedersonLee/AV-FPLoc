from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from matplotlib.colors import LinearSegmentedColormap
LIKELIHOOD_CMAP=LinearSegmentedColormap.from_list("relative_scores",["#3b4cc0","#67a9cf","#b4a6bd","#f4a582","#b40426"])
def world_to_map_pixel(x: float, z: float, metadata: dict[str, object]) -> tuple[int, int]:
    bounds = metadata["world_bounds_m"]
    spacing = float(metadata["map_spacing_m"])
    return round((float(x) - float(bounds["x_min"])) / spacing), round((float(z) - float(bounds["z_min"])) / spacing)

def likelihood_colour(relative_value: float) -> tuple[int, int, int]:
    return tuple(int(round(channel * 255)) for channel in LIKELIHOOD_CMAP(float(np.clip(relative_value, 0, 1)))[:3])

def _marker(draw: ImageDraw.ImageDraw, point: tuple[int, int], yaw: float, *, gt: bool, font: ImageFont.ImageFont) -> None:
    x, y = point
    colour = (10, 90, 240) if gt else (20, 155, 55)
    radius = 13
    if gt:
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=colour, outline="white", width=4)
    else:
        draw.polygon([(x,y-radius),(x+radius,y),(x,y+radius),(x-radius,y)], fill="white", outline=colour)
    length = 80
    endpoint = (x + int(length * math.cos(math.radians(yaw))), y + int(length * math.sin(math.radians(yaw))))
    draw.line((x, y, *endpoint), fill=colour, width=7)
    draw.ellipse((endpoint[0]-5,endpoint[1]-5,endpoint[0]+5,endpoint[1]+5), fill=colour, outline="white", width=2)
    draw.text((x + 15, y - 30), "GT" if gt else "MAP", fill=colour, stroke_width=3, stroke_fill="white", font=font)

def _likelihood_map(source_map: Path, metadata: dict[str, object], field: np.ndarray, free: np.ndarray,
                    gt_pose: list[float], map_pose: list[float], output: Path, *, draw_poses: bool = True) -> None:
    base = Image.open(source_map).convert("RGBA")
    values = np.asarray(field, dtype=np.float32)
    if values.shape != free.shape:
        raise ValueError(f"likelihood/free mask mismatch: {values.shape} vs {free.shape}")
    if not np.isfinite(values).all() or not free.any():
        raise ValueError("likelihood field must be finite with free candidates")
    valid = values[free]
    relative = np.zeros_like(values)
    relative[free] = (valid - valid.min()) / max(float(np.ptp(valid)), 1e-12)
    colours = (LIKELIHOOD_CMAP(relative)[..., :3] * 255).astype(np.uint8)
    alpha = np.where(free, 210, 0).astype(np.uint8)
    heat = Image.fromarray(np.dstack((colours, alpha)), mode="RGBA")
    ratio = round(float(metadata["desdf_spacing_m"]) / float(metadata["map_spacing_m"]))
    heat = heat.resize((values.shape[1] * ratio, values.shape[0] * ratio), Image.NEAREST)
    layer = Image.new("RGBA", base.size)
    layer.paste(heat, (0, 0))
    result = Image.alpha_composite(base, layer).convert("RGB")
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 23)
    except OSError:
        font = ImageFont.load_default()
    if draw_poses:
        _marker(draw, world_to_map_pixel(gt_pose[0], gt_pose[1], metadata), gt_pose[2], gt=True, font=font)
        _marker(draw, world_to_map_pixel(map_pose[0], map_pose[1], metadata), map_pose[2], gt=False, font=font)
    result.save(output, optimize=True)
