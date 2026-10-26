import numpy as np
UNLOC_IMAGE_WIDTH_PX=640
UNLOC_FEATURE_COLUMNS=40
UNLOC_FOCAL_LENGTH_OVER_WIDTH=3/8
def hssd_ray_bins(yaw_bin: int, count: int = 11) -> np.ndarray:
    """DESDF bins are clockwise: image-left rays precede the forward ray."""
    if count < 1 or count % 2 != 1:
        raise ValueError("ray count must be positive and odd")
    return (int(yaw_bin) + np.arange(count) - count // 2) % 36
def unloc_feature_column_bearings_deg(
    *, image_width_px: int = UNLOC_IMAGE_WIDTH_PX, feature_columns: int = UNLOC_FEATURE_COLUMNS,
    focal_length_over_width: float = UNLOC_FOCAL_LENGTH_OVER_WIDTH,
) -> np.ndarray:
    """Bearings of UnLoc's 40 horizontal feature columns, RGB left to right."""

    if image_width_px <= 0 or feature_columns <= 0 or focal_length_over_width <= 0:
        raise ValueError("image width, feature count, and focal-length ratio must be positive")
    # Each attention column pools one contiguous 16-pixel source interval.
    pixel_x = (np.arange(feature_columns, dtype=np.float64) + 0.5) * image_width_px / feature_columns - 0.5
    principal_x = (image_width_px - 1.0) / 2.0
    focal_length_px = focal_length_over_width * image_width_px
    return np.rad2deg(np.arctan((pixel_x - principal_x) / focal_length_px))
