from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CellposeSegProperties:
    model: str
    model_dimensions: str
    version: str
    custom_weights: Optional[str] = None


@dataclass(frozen=True)
class CellposeSegParameters:
    # Fields without default values
    nuclear_channel: str
    entity_fill_channel: str
    diameter: int
    flow_threshold: float
    min_size: int # Renamed from minimum_mask_size and moved

    # Fields with default values
    cellprob_threshold: Optional[float] = None
    mask_threshold: Optional[float] = None # For fallback
    stitch_threshold: Optional[float] = None
    tile: Optional[bool] = None # For tiled prediction
    augment: Optional[bool] = None # For test-time augmentation
    normalize: Optional[bool] = None # Master switch for normalization
    norm_percentile_low: Optional[float] = None
    norm_percentile_high: Optional[float] = None
    tile_norm_blocksize: Optional[int] = None # For tiled normalization
