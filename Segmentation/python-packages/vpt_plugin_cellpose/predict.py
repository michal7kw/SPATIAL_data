import warnings

import numpy as np
from cellpose import models

from vpt_core.io.image import ImageSet
from vpt_plugin_cellpose import CellposeSegProperties, CellposeSegParameters


def run(images: ImageSet, properties: CellposeSegProperties, parameters: CellposeSegParameters) -> np.ndarray:
    print(f"\n[Cellpose Plugin DEBUG] --- Starting run ---")
    print(f"[Cellpose Plugin DEBUG] Properties: {properties}")
    print(f"[Cellpose Plugin DEBUG] Parameters: {parameters}")
    warnings.filterwarnings("ignore", message=".*the `scipy.ndimage.filters` namespace is deprecated.*")

    is_valid_channels = parameters.nuclear_channel and parameters.entity_fill_channel
    image = (
        images.as_stack([parameters.nuclear_channel, parameters.entity_fill_channel])
        if is_valid_channels
        else images.as_stack()
    )
    print(f"[Cellpose Plugin DEBUG] Initial image stack shape: {image.shape}")

    empty_z_levels = set()
    for z_i, z_plane in enumerate(image):
        for channel_i in range(z_plane.shape[-1]):
            if z_plane[..., channel_i].std() < 0.1:
                empty_z_levels.add(z_i)
    print(f"[Cellpose Plugin DEBUG] Empty Z-levels found: {empty_z_levels}")
    if len(empty_z_levels) == image.shape[0]:
        print(f"[Cellpose Plugin DEBUG] All Z-levels are empty. Returning zeros.")
        return np.zeros((image.shape[0],) + image.shape[1:-1])

    if properties.custom_weights:
        model = models.CellposeModel(gpu=True, pretrained_model=properties.custom_weights)
    else:
        model = models.CellposeModel(gpu=True, model_type=properties.model)

    to_segment_z = sorted(list(set(range(image.shape[0])).difference(empty_z_levels))) # Ensure sorted
    print(f"[Cellpose Plugin DEBUG] Z-levels to segment: {to_segment_z}")
    if not to_segment_z:
        print(f"[Cellpose Plugin DEBUG] No Z-levels to segment after filtering empty ones. Returning zeros.")
        return np.zeros((image.shape[0],) + image.shape[1:-1], dtype=np.int32)
        
    img_stack_to_process = image[to_segment_z, ...] # Shape (num_planes_to_segment, Ly, Lx, C)
    print(f"[Cellpose Plugin DEBUG] Image stack to process shape: {img_stack_to_process.shape}")

    is_3D_model = (properties.model_dimensions == "3D")

    eval_kwargs = {
        "diameter": parameters.diameter,
        "flow_threshold": parameters.flow_threshold,
        "resample": False,  # Default from cellpose, not typically user-set here
        "do_3D": is_3D_model,
        "min_size": parameters.min_size, # From updated CellposeSegParameters
    }

    # Handle cell probability threshold (with fallback to mask_threshold)
    if parameters.cellprob_threshold is not None:
        eval_kwargs["cellprob_threshold"] = parameters.cellprob_threshold
    elif parameters.mask_threshold is not None:
        eval_kwargs["cellprob_threshold"] = parameters.mask_threshold

    # Handle test-time augmentation
    if parameters.augment is not None:
        eval_kwargs["augment"] = parameters.augment

    # Handle tiled prediction (for model.eval(tile=...))
    if parameters.tile is not None:
        eval_kwargs["tile"] = parameters.tile

    # Handle stitch threshold for tiled prediction
    if parameters.stitch_threshold is not None:
        eval_kwargs["stitch_threshold"] = parameters.stitch_threshold
    
    # Handle normalization parameters for model.eval(normalize=...)
    # This parameter can be:
    #   None (or not provided): Cellpose uses its default (usually on, percentile global)
    #   False: Turn off normalization
    #   True or {}: Use default normalization (percentile global)
    #   dict: Use specified normalization options
    normalize_param_for_eval = None 
    
    normalize_dict_options = {}
    if parameters.norm_percentile_low is not None:
        normalize_dict_options["percentile_low"] = parameters.norm_percentile_low
    if parameters.norm_percentile_high is not None:
        normalize_dict_options["percentile_high"] = parameters.norm_percentile_high
    if parameters.tile_norm_blocksize is not None:
        normalize_dict_options["tile_norm_blocksize"] = parameters.tile_norm_blocksize

    if parameters.normalize is False: 
        normalize_param_for_eval = False
    elif parameters.normalize is True: 
        normalize_param_for_eval = normalize_dict_options # Pass dict (empty for defaults, or populated)
    elif normalize_dict_options: # normalize is None, but sub-options are present
        normalize_param_for_eval = normalize_dict_options
    
    if normalize_param_for_eval is not None:
        eval_kwargs["normalize"] = normalize_param_for_eval

    if is_3D_model:
        actual_eval_input = img_stack_to_process 
        eval_kwargs["z_axis"] = 0
        eval_kwargs["channel_axis"] = actual_eval_input.ndim - 1
    else: 
        eval_kwargs["z_axis"] = None
        if img_stack_to_process.shape[0] == 1:
            actual_eval_input = img_stack_to_process.squeeze(0) 
            eval_kwargs["channel_axis"] = actual_eval_input.ndim - 1
        else: 
            actual_eval_input = [img_stack_to_process[i] for i in range(img_stack_to_process.shape[0])]
            eval_kwargs["channel_axis"] = 2 
    
    print(f"[Cellpose Plugin DEBUG] Preparing to process {len(to_segment_z)} Z-plane(s) with {'3D' if is_3D_model else '2D'} model.")
    print(f"[Cellpose Plugin DEBUG] model.eval arguments (eval_kwargs): {eval_kwargs}")
    if isinstance(actual_eval_input, np.ndarray):
        print(f"[Cellpose Plugin DEBUG] actual_eval_input shape: {actual_eval_input.shape}")
    elif isinstance(actual_eval_input, list) and actual_eval_input:
        print(f"[Cellpose Plugin DEBUG] actual_eval_input is a list of {len(actual_eval_input)} arrays, first element shape: {actual_eval_input[0].shape}")
    else:
        print(f"[Cellpose Plugin DEBUG] actual_eval_input type: {type(actual_eval_input)}")
            
    processed_mask_output = model.eval(actual_eval_input, **eval_kwargs)[0]
    print(f"[Cellpose Plugin DEBUG] model.eval output (masks) type: {type(processed_mask_output)}")
    if isinstance(processed_mask_output, np.ndarray):
        print(f"[Cellpose Plugin DEBUG] model.eval output (masks) shape: {processed_mask_output.shape}")
    elif isinstance(processed_mask_output, list) and processed_mask_output:
        print(f"[Cellpose Plugin DEBUG] model.eval output (masks) is a list of {len(processed_mask_output)} arrays, first element shape: {processed_mask_output[0].shape}")

    if isinstance(processed_mask_output, list):
        processed_mask = np.array(processed_mask_output)
    elif not is_3D_model and img_stack_to_process.shape[0] == 1:
        processed_mask = np.expand_dims(processed_mask_output, axis=0) 
    else:
        processed_mask = processed_mask_output

    full_mask_shape = (image.shape[0], image.shape[1], image.shape[2])
    if processed_mask.size > 0 : 
        final_mask_array = np.zeros(full_mask_shape, dtype=processed_mask.dtype)
    else: 
        final_mask_array = np.zeros(full_mask_shape, dtype=np.int32) 

    for idx, original_z_idx in enumerate(to_segment_z):
        final_mask_array[original_z_idx] = processed_mask[idx]
        
    print(f"[Cellpose Plugin DEBUG] --- Finished run --- Returning final_mask_array with shape: {final_mask_array.shape}")
    return final_mask_array
