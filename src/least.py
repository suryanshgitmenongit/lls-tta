# least.py - Layer selection algorithm for test-time adaptation 

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
import traceback
import datetime
import os

# Enable/disable debug prints overall
DEBUG_MODE = True

def debug_print(message, force=False):
    """Print debug message if DEBUG_MODE is enabled or force is True."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if DEBUG_MODE or force:
        print(f"[DEBUG:{timestamp}] {message}")

def remove_duplicates(seq):
    """Remove duplicate items from a sequence while preserving order."""
    seen = set()
    seen_add = seen.add
    result = [x for x in seq if not (x in seen or seen_add(x))]
    debug_print(f"remove_duplicates: input={len(seq)} items, output={len(result)} items (removed {len(seq)-len(result)} duplicates)")
    return result


def ensure_float(value):
    """Convert tensor or numeric value to plain float.
    
    This helper ensures consistent type handling across the codebase for mask values.
    
    Args:
        value: Can be torch.Tensor, Python float, or Python int
        
    Returns:
        float: Plain Python float
    """
    debug_print(f"ensure_float: input type={type(value).__name__}, value={value}")
    if isinstance(value, torch.Tensor):
        result = float(value.item())
        debug_print(f"ensure_float: converted tensor to float={result}")
        return result
    result = float(value)
    debug_print(f"ensure_float: converted to float={result}")
    return result


def ensure_tensor(value, device):
    """Convert float or tensor to tensor on specified device.
    
    This helper ensures consistent type handling when mask values need to be tensors.
    
    Args:
        value: Can be torch.Tensor, Python float, or Python int
        device: Target device for the tensor
        
    Returns:
        torch.Tensor: Tensor on specified device
    """
    debug_print(f"ensure_tensor: input type={type(value).__name__}, value={value}, target_device={device}")
    if isinstance(value, torch.Tensor):
        result = value.to(device)
        debug_print(f"ensure_tensor: tensor already exists, moved to {device}, shape={result.shape}")
        return result
    result = torch.tensor(value, device=device)
    debug_print(f"ensure_tensor: created new tensor on {device}, shape={result.shape}, value={result.item()}")
    return result


def get_all_param_name_list(algorithm, granularity):
    """Get list of parameter group names based on specified granularity.
    
    Args:
        algorithm: Model or algorithm instance
        granularity: Level of detail ('block', 'subblock', 'layer', 'model')
        
    Returns:
        list: List of parameter group names
    """
    debug_print("=" * 60)
    debug_print(f"get_all_param_name_list ENTRY: granularity='{granularity}'")
    
    granularity = granularity.lower()
    
    model = algorithm
    if hasattr(algorithm, 'featurizer'):
        model = algorithm.featurizer.network
        debug_print("get_all_param_name_list: extracted model from algorithm.featurizer.network")
    elif hasattr(algorithm, 'network'):
        model = algorithm.network
        debug_print("get_all_param_name_list: extracted model from algorithm.network")
    else:
        debug_print("get_all_param_name_list: using algorithm directly as model")
    
    total_params = len(list(model.named_parameters()))
    debug_print(f"get_all_param_name_list: model has {total_params} named parameters")
    debug_print(f"get_all_param_name_list: model type = {type(model).__name__}")
    
    if isinstance(model, torchvision.models.ResNet):
        debug_print("get_all_param_name_list: detected ResNet model")
        pre_box_list = []
        post_box_list = []
        layer_name = 'layer'
        
        if granularity == 'block':
            granularity_level = 1
            debug_print("get_all_param_name_list: using Block granularity (level=1)")
        elif granularity == 'subblock':
            granularity_level = 2
            debug_print("get_all_param_name_list: using SubBlock granularity (level=2)")
        elif granularity == 'layer':
            granularity_level = -1
            debug_print("get_all_param_name_list: using Layer granularity (level=-1, all layers)")
        elif granularity == 'model':
            debug_print("get_all_param_name_list: using Model granularity (single box)")
            return [layer_name]
        else:
            error_msg = f'Not implemented for granularity: {granularity}'
            debug_print(f"get_all_param_name_list ERROR: {error_msg}", force=True)
            raise NotImplementedError(error_msg)

    elif len(list(model.named_parameters())) == 80:
        debug_print("get_all_param_name_list: detected 80-parameter model")
        pre_box_list = []
        post_box_list = []
        layer_name = 'layer'
        
        if granularity == 'block':
            granularity_level = 1
            debug_print("get_all_param_name_list: using Block granularity (level=1)")
        elif granularity == 'subblock':
            granularity_level = 3
            debug_print("get_all_param_name_list: using SubBlock granularity (level=3)")
        elif granularity == 'layer':
            granularity_level = 4
            debug_print("get_all_param_name_list: using Layer granularity (level=4)")
        elif granularity == 'model':
            debug_print("get_all_param_name_list: using Model granularity (single box)")
            return [layer_name]
        else:
            error_msg = f'Not implemented for granularity: {granularity}'
            debug_print(f"get_all_param_name_list ERROR: {error_msg}", force=True)
            raise NotImplementedError(error_msg)
    
    elif len(list(model.named_parameters())) == 95:
        debug_print("get_all_param_name_list: detected 95-parameter model")
        pre_box_list = []
        post_box_list = []
        layer_name = 'stage'
        
        if granularity == 'block':
            granularity_level = 1
            debug_print("get_all_param_name_list: using Block granularity (level=1)")
        elif granularity == 'subblock':
            granularity_level = 2
            debug_print("get_all_param_name_list: using SubBlock granularity (level=2)")
        elif granularity == 'layer':
            granularity_level = 3
            debug_print("get_all_param_name_list: using Layer granularity (level=3)")
        elif granularity == 'model':
            debug_print("get_all_param_name_list: using Model granularity (single box)")
            return [layer_name]
        else:
            error_msg = f'Not implemented for granularity: {granularity}'
            debug_print(f"get_all_param_name_list ERROR: {error_msg}", force=True)
            raise NotImplementedError(error_msg)
        
    else:
        error_msg = f'Unrecognized model type with {len(list(model.named_parameters()))} parameters'
        debug_print(f"get_all_param_name_list ERROR: {error_msg}", force=True)
        raise NotImplementedError(error_msg)
    
    all_params = list(model.named_parameters())
    debug_print(f"get_all_param_name_list: processing {len(all_params)} parameters to extract boxes")
    
    all_box_list = []
    for name, _ in all_params:
        if layer_name in name:
            if granularity_level > 0:
                parts = name.split(".")
                if len(parts) >= granularity_level:
                    box_name = ".".join(parts[:granularity_level])
                    all_box_list.append(box_name)
            else:
                all_box_list.append(name)
    
    debug_print(f"get_all_param_name_list: extracted {len(all_box_list)} box names (before deduplication)")
    
    all_box_list = remove_duplicates(all_box_list)
    
    result = pre_box_list + all_box_list + post_box_list
    debug_print(f"get_all_param_name_list RETURN: {len(result)} boxes = {result}")
    debug_print("=" * 60)
    return result


def get_clean_layer_name(layer_name):
    """Remove common prefixes from layer names for consistent matching.
    
    Args:
        layer_name: Original layer name
        
    Returns:
        str: Cleaned layer name
    """
    remove_words = ['featurizer', 'network']
    clean_name = '.'.join([
        split_name for split_name in layer_name.split('.')
        if split_name not in remove_words
    ])
    return clean_name


def is_layer_trainable(name, trainable_layer_names):
    """Check if a layer should be trained based on its name.
    
    Args:
        name: Layer name
        trainable_layer_names: List of trainable layer names
        
    Returns:
        bool: True if layer should be trained, False otherwise
    """
    if trainable_layer_names is None:
        return True
    
    clean_name = get_clean_layer_name(name)
    is_trainable = clean_name in trainable_layer_names
    return is_trainable


# Global variables for LEAST
source_params_dict = {}
window_start_params_dict = {}
window_index = None

# Global storage for LLS
_lls_storage = {}

# Global state for early stopping
min_loss_dict, steps_no_improve_dict = {}, {}
stop_flag_dict = {}


def get_td_variant(mask_algo):
    """Parse trajectory delta (TD) hyperparameters from mask_algo string.
    
    Args:
        mask_algo: Configuration string (e.g., "LEAST_nbox:1_curr:next")
        
    Returns:
        dict: Dictionary of TD configuration options
    """
    debug_print(f"get_td_variant ENTRY: mask_algo='{mask_algo}'")
    
    td_variant = {
        'nbox'   : '1',
        'curr'   : 'next',
        'bin'    : '1',
        'trust'  : 'low',
        'lvl'    : 'Layer',
        'type'   : 'best',
        'order'  : '0',
        'sz'     : '20',
        'mthr'   : '0.75',
        'sthr'   : '0.75',
    }
    
    debug_print(f"get_td_variant: default config = {td_variant}")

    hparam_count = 0
    for hparam in mask_algo.split('_'):
        if ':' in hparam:
            key, value = hparam.split(':')
            if key not in td_variant:
                error_msg = f'Invalid mask_hparam: {key}'
                debug_print(f"get_td_variant ERROR: {error_msg}", force=True)
                raise ValueError(error_msg)
            td_variant[key] = value.lower()
            hparam_count += 1
            debug_print(f"get_td_variant: parsed {key}={value.lower()}")
    
    debug_print(f"get_td_variant RETURN: parsed {hparam_count} hyperparameters, final config = {td_variant}")
    return td_variant


def get_anchor_params_dict(before_params_dict, td_variant):
    """Get anchor parameters for computing trajectory deltas.
    
    Args:
        before_params_dict: Dictionary of parameters before update
        td_variant: LEAST configuration dictionary
        
    Returns:
        dict: Anchor parameter dictionary for trajectory computation
    """
    global source_params_dict, window_start_params_dict, window_index

    iter_idx = td_variant['iter_idx']
    debug_print(f"get_anchor_params_dict ENTRY: iter_idx={iter_idx}, window_sz={td_variant['sz']}")
    
    anchor_params_dict = {}
    
    if iter_idx == 0:
        debug_print("get_anchor_params_dict: iter_idx=0, initializing global state")
        source_params_dict = copy.deepcopy(before_params_dict)
        anchor_params_dict = copy.deepcopy(before_params_dict)
        window_start_params_dict = copy.deepcopy(before_params_dict)
        window_index = 0
        debug_print(f"get_anchor_params_dict: initialized {len(source_params_dict)} boxes")

    else:
        if td_variant['sz'] == 'n':
            debug_print("get_anchor_params_dict: using source params (infinite window)")
            if not source_params_dict:
                error_msg = 'TD Error: Source parameters are empty'
                debug_print(f"get_anchor_params_dict ERROR: {error_msg}", force=True)
                raise ValueError(error_msg)
            anchor_params_dict = copy.deepcopy(source_params_dict)
            window_index += 1
            debug_print(f"get_anchor_params_dict: window_index incremented to {window_index}")

        elif td_variant['sz'].isdigit() and int(td_variant['sz']) > 0:
            sz = int(td_variant['sz'])
            debug_print(f"get_anchor_params_dict: using windowed params (window_sz={sz})")
            
            if iter_idx % sz == 0:
                debug_print(f"get_anchor_params_dict: window boundary reached (iter_idx={iter_idx} % {sz} = 0), resetting window")
                window_start_params_dict = copy.deepcopy(before_params_dict)
                window_index = 0
            else:
                window_index += 1
            
            debug_print(f"get_anchor_params_dict: window_index={window_index}/{sz}")
            
            if not window_start_params_dict:
                error_msg = 'TD Error: Window start parameters are empty'
                debug_print(f"get_anchor_params_dict ERROR: {error_msg}", force=True)
                raise ValueError(error_msg)
            anchor_params_dict = copy.deepcopy(window_start_params_dict)

        else:
            error_msg = f'TD Error: Invalid window size: {td_variant["sz"]}'
            debug_print(f"get_anchor_params_dict ERROR: {error_msg}", force=True)
            raise ValueError(error_msg)

    debug_print(f"get_anchor_params_dict RETURN: {len(anchor_params_dict)} boxes, window_index={window_index}")
    return anchor_params_dict


def get_curr_params_dict(before_params_dict, after_params_dict, td_variant):
    """Get current parameters for computing deltas based on strategy.
    
    Args:
        before_params_dict: Parameters before update
        after_params_dict: Parameters after update
        td_variant: Configuration dictionary
        
    Returns:
        dict: Current parameters for delta computation
    """
    debug_print(f"get_curr_params_dict ENTRY: strategy='{td_variant['curr']}'")
    
    if td_variant['curr'] == 'next':
        debug_print("get_curr_params_dict: using after_params (next)")
        return copy.deepcopy(after_params_dict)
    elif td_variant['curr'] == 'prev':
        debug_print("get_curr_params_dict: using before_params (prev)")
        return copy.deepcopy(before_params_dict)        
    else:
        error_msg = f'TD Error: Get current parameters not implemented for {td_variant["curr"]}'
        debug_print(f"get_curr_params_dict ERROR: {error_msg}", force=True)
        raise ValueError(error_msg)


def compute_total_delta_tensor_of_box(curr_params, anchor_params, td_variant):
    """Compute parameter delta for a specific box.
    
    Args:
        curr_params: Current parameters
        anchor_params: Anchor parameters
        td_variant: Configuration dictionary
        
    Returns:
        torch.Tensor: Parameter delta tensor
    """
    debug_print(f"compute_total_delta_tensor_of_box: curr_shape={curr_params.shape}, anchor_shape={anchor_params.shape}")
    
    # FIX #3: Handle empty tensors explicitly
    if curr_params.numel() == 0 and anchor_params.numel() == 0:
        debug_print("compute_total_delta_tensor_of_box: both tensors empty, returning empty delta", force=True)
        return torch.tensor([], device=curr_params.device)
    
    if curr_params.numel() == 0 or anchor_params.numel() == 0:
        error_msg = f'TD Error: Mismatched empty state (curr={curr_params.numel()}, anchor={anchor_params.numel()})'
        debug_print(f"compute_total_delta_tensor_of_box ERROR: {error_msg}", force=True)
        raise ValueError(error_msg)
    
    if curr_params.shape != anchor_params.shape:
        error_msg = f'TD Error: Shape mismatch for curr ({curr_params.shape}) and anchor ({anchor_params.shape}) parameters'
        debug_print(f"compute_total_delta_tensor_of_box ERROR: {error_msg}", force=True)
        raise ValueError(error_msg)
    
    total_delta = curr_params - anchor_params
    delta_norm = total_delta.norm().item()
    debug_print(f"compute_total_delta_tensor_of_box: computed delta, norm={delta_norm:.6f}")
    
    return total_delta


def get_total_delta_tensor(anchor_params_dict, curr_params_dict, td_variant):
    """Compute parameter deltas for all boxes.
    
    Args:
        anchor_params_dict: Anchor parameters for all boxes
        curr_params_dict: Current parameters for all boxes
        td_variant: Configuration dictionary
        
    Returns:
        dict: Dictionary of parameter delta tensors for each box
    """
    debug_print(f"get_total_delta_tensor ENTRY: computing deltas for {len(anchor_params_dict)} boxes")
    
    total_delta_tensor_dict = {}
    
    for box_name in anchor_params_dict.keys():
        debug_print(f"get_total_delta_tensor: processing box '{box_name}'")
        
        # FIX #2: Guard against empty parameter lists
        curr_parts = [params.flatten() for params in curr_params_dict.get(box_name, [])]
        anchor_parts = [params.flatten() for params in anchor_params_dict.get(box_name, [])]
        
        if len(curr_parts) == 0:
            debug_print(f"get_total_delta_tensor: box '{box_name}' has no parameters - creating empty tensor", force=True)
            # Determine device from anchor if available, otherwise use CPU
            device = torch.device('cpu')
            if len(anchor_parts) > 0:
                device = anchor_parts[0].device
            total_delta_tensor_dict[box_name] = torch.tensor([], device=device)
            continue
        
        curr_params = torch.cat(curr_parts)
        anchor_params = torch.cat(anchor_parts)
        
        debug_print(f"get_total_delta_tensor: box '{box_name}' has {curr_params.numel()} parameters")

        total_delta_tensor_dict[box_name] = compute_total_delta_tensor_of_box(
            curr_params, anchor_params, td_variant)

    debug_print(f"get_total_delta_tensor RETURN: computed deltas for {len(total_delta_tensor_dict)} boxes")
    return total_delta_tensor_dict


def get_traj_td_tensor_dict(before_params_dict, after_params_dict, td_variant):
    """Compute trajectory delta tensors for all boxes.
    
    Args:
        before_params_dict: Parameters before update
        after_params_dict: Parameters after update
        td_variant: Configuration dictionary
        
    Returns:
        dict: Dictionary of trajectory delta tensors
    """
    debug_print("get_traj_td_tensor_dict ENTRY: computing trajectory deltas")
    
    anchor_params_dict = get_anchor_params_dict(before_params_dict, td_variant)
    curr_params_dict = get_curr_params_dict(before_params_dict, after_params_dict, td_variant)

    total_delta_tensor_dict = get_total_delta_tensor(
        anchor_params_dict, curr_params_dict, td_variant)
    
    debug_print(f"get_traj_td_tensor_dict RETURN: {len(total_delta_tensor_dict)} trajectory deltas")
    return total_delta_tensor_dict


def get_samp_td_tensor_dict(before_params_dict, after_params_dict, td_variant):
    """Compute sample/batch delta tensors for all boxes.
    
    Args:
        before_params_dict: Parameters before update
        after_params_dict: Parameters after update
        td_variant: Configuration dictionary
        
    Returns:
        dict: Dictionary of sample delta tensors
    """
    debug_print("get_samp_td_tensor_dict ENTRY: computing sample deltas")
    
    anchor_params_dict = copy.deepcopy(before_params_dict)
    curr_params_dict = copy.deepcopy(after_params_dict)

    total_delta_tensor_dict = get_total_delta_tensor(
        anchor_params_dict, curr_params_dict, td_variant)
    
    debug_print(f"get_samp_td_tensor_dict RETURN: {len(total_delta_tensor_dict)} sample deltas")
    return total_delta_tensor_dict


def compute_metric_values(td_traj_tensor_dict, td_samp_tensor_dict, td_variant):
    """Compute alignment metrics between trajectory and sample deltas.
    
    Args:
        td_traj_tensor_dict: Trajectory delta tensors
        td_samp_tensor_dict: Sample delta tensors
        td_variant: Configuration dictionary
        
    Returns:
        dict: Dictionary of alignment metrics for each box
    """
    debug_print(f"compute_metric_values ENTRY: computing cosine similarity for {len(td_traj_tensor_dict)} boxes")
    
    metric_values = {}
    
    for box_name in td_traj_tensor_dict.keys():
        td_traj_tensor = td_traj_tensor_dict[box_name]
        td_samp_tensor = td_samp_tensor_dict[box_name]
        
        traj_norm = td_traj_tensor.norm().item()
        samp_norm = td_samp_tensor.norm().item()
        
        # FIX #4: CRITICAL - Check for zero norms before cosine similarity
        if traj_norm < 1e-10 or samp_norm < 1e-10:
            metric_value = torch.tensor(0.0, device=td_traj_tensor.device)
            debug_print(f"compute_metric_values: box '{box_name}' has zero norm (traj={traj_norm:.2e}, samp={samp_norm:.2e}), setting metric=0", force=True)
        else:
            metric_value = torch.cosine_similarity(td_traj_tensor, td_samp_tensor, dim=0)
            debug_print(f"compute_metric_values: box '{box_name}' cosine={metric_value.item():.6f} (traj_norm={traj_norm:.6f}, samp_norm={samp_norm:.6f})")
        
        metric_values[box_name] = metric_value

    debug_print(f"compute_metric_values RETURN: {len(metric_values)} metric values computed")
    return metric_values


def get_mistrust_window_sz(td_variant):
    """Get window size for mistrust calculation.
    
    Args:
        td_variant: Configuration dictionary
        
    Returns:
        int: Window size for mistrust calculation
    """
    if td_variant['sz'] == 'n':
        sz = 50
        debug_print(f"get_mistrust_window_sz: infinite window, using default sz=50")
    else:
        sz = int(td_variant['sz'])
        debug_print(f"get_mistrust_window_sz: using configured sz={sz}")
    
    sthr = float(td_variant['sthr'])
    mistrust_window_sz = int(sz * sthr)
    debug_print(f"get_mistrust_window_sz RETURN: mistrust_window_sz={mistrust_window_sz} (sz={sz} * sthr={sthr})")
    return mistrust_window_sz


def compute_mask(metric_values_dict, td_variant):
    """Compute masks from metric values based on configuration.
    
    FIX: Derive device consistently from metric_values_dict at start,
    rather than using conditional logic with 'cpu' fallback which could cause
    device mismatches in multi-GPU setups.
    
    Args:
        metric_values_dict: Dictionary of metrics for each box
        td_variant: Configuration dictionary
        
    Returns:
        dict: Dictionary of mask values for each box
    """
    global window_index
    
    debug_print("=" * 50)
    debug_print(f"compute_mask ENTRY: computing masks for {len(metric_values_dict)} boxes")
    debug_print(f"compute_mask: config type={td_variant['type']}, nbox={td_variant['nbox']}, bin={td_variant['bin']}, trust={td_variant['trust']}")
    
    # FIX #5: Derive device consistently from metric values
    if len(metric_values_dict) == 0:
        debug_print("compute_mask ERROR: Empty metric_values_dict", force=True)
        raise ValueError("compute_mask called with empty metric_values_dict")
    
    device = next(iter(metric_values_dict.values())).device
    debug_print(f"compute_mask: using device={device} (derived from metric values)")
    
    type = td_variant['type']
    if type == 'worst':
        debug_print("compute_mask: using 'worst' strategy")
        return compute_worst_mask(metric_values_dict, td_variant)
    if type == 'notbest':
        debug_print("compute_mask: using 'notbest' strategy")
        return compute_not_best_mask(metric_values_dict, td_variant)
    elif type == 'random':
        debug_print("compute_mask: using 'random' strategy")
        return compute_random_mask(metric_values_dict, td_variant)

    mask_dict = copy.deepcopy(metric_values_dict)
    debug_print("compute_mask: starting with raw metric values as initial masks")

    # Step 1: Apply nbox constraint
    nbox = td_variant['nbox']
    if nbox == '1':
        debug_print("compute_mask: applying nbox=1 (keep only best box)")
        max_mask_value = max(mask_dict.values())
        debug_print(f"compute_mask: max metric value = {max_mask_value.item():.6f}")
        
        for box_name, mask_value in mask_dict.items():
            if mask_value == max_mask_value:
                mask_dict[box_name] = mask_value
                debug_print(f"compute_mask: box '{box_name}' KEPT (value={mask_value.item():.6f})")
            else:
                # FIX #5: Use consistent device
                mask_dict[box_name] = torch.tensor(0.0, device=device)
                debug_print(f"compute_mask: box '{box_name}' ZEROED (was {mask_value.item():.6f})")
    elif nbox == 'n':
        debug_print("compute_mask: nbox='n', keeping all boxes")
        pass
    else:
        error_msg = f'TD Error: Invalid number of boxes: {nbox}'
        debug_print(f"compute_mask ERROR: {error_msg}", force=True)
        raise ValueError(error_msg)

    # Step 2: Apply binarization
    bin_mask = td_variant['bin']
    mthr = float(td_variant['mthr'])
    if bin_mask == '1':
        debug_print(f"compute_mask: applying binarization (threshold={mthr})")
        for box_name, mask_value in mask_dict.items():
            original = mask_value.item() if isinstance(mask_value, torch.Tensor) else mask_value
            if mask_value > mthr:
                # FIX #5: Use consistent device
                mask_dict[box_name] = torch.tensor(1.0, device=device)
                debug_print(f"compute_mask: box '{box_name}' -> 1.0 (was {original:.6f} > {mthr})")
            else:
                # FIX #5: Use consistent device
                mask_dict[box_name] = torch.tensor(0.0, device=device)
                debug_print(f"compute_mask: box '{box_name}' -> 0.0 (was {original:.6f} <= {mthr})")
    elif bin_mask == '0':
        debug_print("compute_mask: binarization disabled, keeping continuous values")
        pass
    else:
        error_msg = f'TD Error: Invalid binarize mask: {bin_mask}'
        debug_print(f"compute_mask ERROR: {error_msg}", force=True)
        raise ValueError(error_msg)

    # Step 3: Apply trust factor
    trust = td_variant['trust']
    if trust == 'low':
        mistrust_window_sz = get_mistrust_window_sz(td_variant)
        debug_print(f"compute_mask: applying low trust (mistrust_window={mistrust_window_sz}, current_idx={window_index})")
        
        if window_index+1 < mistrust_window_sz:
            mistrust_factor = (window_index+1) / mistrust_window_sz
            debug_print(f"compute_mask: within mistrust window, applying factor={mistrust_factor:.4f}")
            
            for box_name, mask_value in mask_dict.items():
                original = mask_value.item() if isinstance(mask_value, torch.Tensor) else mask_value
                mask_dict[box_name] *= mistrust_factor
                new_val = mask_dict[box_name].item() if isinstance(mask_dict[box_name], torch.Tensor) else mask_dict[box_name]
                debug_print(f"compute_mask: box '{box_name}' scaled: {original:.6f} -> {new_val:.6f}")
        else:
            debug_print(f"compute_mask: outside mistrust window (idx={window_index+1} >= {mistrust_window_sz}), no scaling")
    elif trust == 'high':
        debug_print("compute_mask: high trust mode, no scaling applied")
        pass
    else:
        error_msg = f'TD Error: Invalid trust on init samples: {trust}'
        debug_print(f"compute_mask ERROR: {error_msg}", force=True)
        raise ValueError(error_msg)

    debug_print("compute_mask RETURN: final mask values:")
    for box_name, mask_value in mask_dict.items():
        val = mask_value.item() if isinstance(mask_value, torch.Tensor) else mask_value
        debug_print(f"  {box_name}: {val:.6f}")
    debug_print("=" * 50)
    
    return mask_dict

def get_mask_values(trainable_lr_dict, model, trainable_layer_names):
    """Convert mask dictionary to list of mask values for each parameter.
    
    Args:
        trainable_lr_dict: Dictionary of mask values for each box
        model: Neural network model
        trainable_layer_names: List of trainable layer names
        
    Returns:
        list: List of mask values for each parameter
    """
    debug_print(f"get_mask_values ENTRY: converting {len(trainable_lr_dict)} box masks to parameter masks")
    
    lr_values = []
    total_params = 0
    trainable_params = 0
    matched_params = 0
    
    for idx, (name, param) in enumerate(model.named_parameters()):
        total_params += 1
        trainable_layer = is_layer_trainable(name, trainable_layer_names)
        
        if param.requires_grad and trainable_layer:
            trainable_params += 1
            lr_value = 0
            matched = False
            
            for dname in trainable_lr_dict.keys():
                if dname in name:
                    lr_value = trainable_lr_dict[dname]
                    matched = True
                    matched_params += 1
                    break
            
            # Use ensure_float helper for consistent type handling
            lr_values.append(ensure_float(lr_value))

    debug_print(f"get_mask_values: processed {total_params} total params, {trainable_params} trainable, {matched_params} matched to boxes")
    
    if sum(lr_values) <= 0:
        debug_print("WARNING: No trainable parameters found (all masks are zero)", force=True)
    else:
        active_count = sum(1 for v in lr_values if v > 0)
        inactive_count = sum(1 for v in lr_values if v == 0)
        debug_print(f"get_mask_values RETURN: {len(lr_values)} masks ({active_count} active, {inactive_count} inactive, sum={sum(lr_values):.4f})")
    
    return lr_values


def get_LEAST_masks(mask_algo, model, base_optimizer, iter_idx, trainable_layer_names, track_flag=False, **kwargs):
    """Compute LEAST masks for layer selection.
    
    Args:
        mask_algo: Configuration string
        model: Neural network model
        base_optimizer: Optimizer used for adaptation
        iter_idx: Current iteration index
        trainable_layer_names: List of trainable layer names
        track_flag: If True, return info dictionary for tracking
        **kwargs: Additional arguments
        
    Returns:
        tuple: (mask values, info dictionary)
    """
    debug_print("=" * 80)
    debug_print("GET_LEAST_MASKS ENTRY")
    debug_print(f"  mask_algo: '{mask_algo}'")
    debug_print(f"  iter_idx: {iter_idx}")
    debug_print(f"  trainable_layer_names: {trainable_layer_names}")
    debug_print(f"  track_flag: {track_flag}")
    
    try:
        td_variant = get_td_variant(mask_algo)
        td_variant['iter_idx'] = iter_idx
        debug_print(f"  Parsed TD config: {td_variant}")
        
        granularity = td_variant['lvl']
        box_names = get_all_param_name_list(model, granularity)
        debug_print(f"  Found {len(box_names)} boxes at '{granularity}' level")

        # Step 1: Save current state
        debug_print("  STEP 1: Saving current model state")
        all_curr_params_dict, all_curr_grads_dict = {}, {}
        params_with_grad = 0
        params_without_grad = 0
        
        for name, params in model.named_parameters():
            all_curr_params_dict[name] = params.data.clone()
            if params.grad is None:
                all_curr_grads_dict[name] = None
                params_without_grad += 1
            else:
                all_curr_grads_dict[name] = params.grad.clone()
                params_with_grad += 1
        
        debug_print(f"  Saved state for {len(all_curr_params_dict)} params ({params_with_grad} with grad, {params_without_grad} without)")
        
        # Step 2: Simulate optimizer step
        debug_print("  STEP 2: Simulating optimizer step to measure deltas")
        base_opt_params = copy.deepcopy(base_optimizer.defaults)
        debug_print(f"  Base optimizer params: {base_opt_params}")
        
        temp_params_list = []
        before_params_dict = {}
        
        for box_name in box_names:
            before_params_dict[box_name] = []
            box_param_count = 0
            
            for idx, (name, params) in enumerate(model.named_parameters()):
                trainable_layer = is_layer_trainable(name, trainable_layer_names)
                if params.requires_grad and trainable_layer:
                    if box_name in name:
                        temp_params_list.append(params)
                        before_params_dict[box_name].append(params.data.clone())
                        box_param_count += 1
            
            if box_param_count > 0:
                debug_print(f"  Box '{box_name}': {box_param_count} parameters")

        debug_print(f"  Total trainable params for optimizer: {len(temp_params_list)}")
        temp_optimizer = type(base_optimizer)(temp_params_list, **base_opt_params)
        debug_print(f"  Created temporary optimizer: {type(temp_optimizer).__name__}")
        
        # FIX #1: Guard against missing gradients
        has_grad = any(p.grad is not None for p in temp_params_list)
        if not has_grad:
            debug_print("LEAST ERROR: No gradients on parameters. Call backward() before layer selection.", force=True)
            raise RuntimeError(
                "LEAST requires gradients to compute parameter deltas. "
                "Ensure loss.backward() is called before get_new_optimizer()."
            )
        
        temp_optimizer.step()
        debug_print("  Executed optimizer step")

        # Step 3: Capture after state
        debug_print("  STEP 3: Capturing state after optimizer step")
        after_params_dict = {}
        
        for box_name in box_names:
            after_params_dict[box_name] = []
            
            for idx, (name, params) in enumerate(model.named_parameters()):
                trainable_layer = is_layer_trainable(name, trainable_layer_names)
                if params.requires_grad and trainable_layer:
                    if box_name in name:
                        after_params_dict[box_name].append(params.data.clone())

        # Step 4: Compute trajectory deltas
        debug_print("  STEP 4: Computing trajectory deltas")
        td_traj_tensor_dict = get_traj_td_tensor_dict(before_params_dict, after_params_dict, td_variant)

        # Step 5: Compute sample deltas
        debug_print("  STEP 5: Computing sample deltas")
        td_samp_tensor_dict = get_samp_td_tensor_dict(before_params_dict, after_params_dict, td_variant)
        
        # Step 6: Compute alignment metrics
        debug_print("  STEP 6: Computing alignment metrics")
        metric_values_dict = compute_metric_values(td_traj_tensor_dict, td_samp_tensor_dict, td_variant)

        # Step 7: Compute final masks
        debug_print("  STEP 7: Computing final masks from metrics")
        mask_dict = compute_mask(metric_values_dict, td_variant)

        # FIX #6 + FIX #4: Normalize mask_dict to plain floats using helper
        debug_print("  Converting mask tensors to plain floats")
        mask_dict = {k: ensure_float(v) for k, v in mask_dict.items()}

        # Step 8: Restore original state
        debug_print("  STEP 8: Restoring original model state")
        restored_params = 0
        restored_grads = 0
        
        for name, params in model.named_parameters():
            params.data = all_curr_params_dict[name]
            restored_params += 1
            params.grad = all_curr_grads_dict[name]
            if params.grad is not None:
                restored_grads += 1
        
        debug_print(f"  Restored {restored_params} params and {restored_grads} gradients")

        # Step 9: Convert to parameter-level masks
        debug_print("  STEP 9: Converting box masks to parameter-level masks")
        mask_values = get_mask_values(mask_dict, model, trainable_layer_names)

        info = {
            'metric_values_dict': metric_values_dict,
            'mask_dict': mask_dict
        }

        debug_print("GET_LEAST_MASKS RETURN:")
        debug_print(f"  Returning {len(mask_values)} parameter masks")
        debug_print(f"  Info dict keys: {list(info.keys())}")
        debug_print("=" * 80)
        
        return mask_values, info
        
    except Exception as e:
        debug_print(f"ERROR in get_LEAST_masks: {str(e)}", force=True)
        debug_print(traceback.format_exc(), force=True)
        raise


def get_lls_variant(mask_algo):
    """Parse LLS configuration from mask_algo string.
    
    Args:
        mask_algo: Configuration string (e.g., "LLS_init:rgn_lr:0.01")
        
    Returns:
        dict: Dictionary of LLS configuration options
    """
    debug_print(f"get_lls_variant ENTRY: mask_algo='{mask_algo}'")
    
    mtype = False
    if 'update' in mask_algo.lower():
        mtype = True
        debug_print("get_lls_variant: detected 'update' keyword, enabling model interpolation mode")
    
    lls_variant = {
        'init'    : 'least',
        'finit'   : '0.5',
        'emalr'   : '0.5',
        'lr'      : '0.01',
        'steps'   : '10',
        'patience': '3',
        'loss'    : 'curr',
        'reg'     : 'l1',
        'regstr'  : '0.01',
        'bnd'     : 'clip',
        'bin'     : '0',
        'appl'    : 'avg',
        'update'  : mtype,
        'lvl'     : 'Layer',
        'nbox'    : '1',
        'mthr'    : '0.75',
    }
    
    debug_print(f"get_lls_variant: default config = {lls_variant}")
    
    hparam_count = 0
    for hparam in mask_algo.split('_'):
        if ':' in hparam:
            key, value = hparam.split(':')
            if key not in lls_variant:
                debug_print(f"WARNING: Invalid LLS hyperparameter: {key}, ignoring")
            else:
                lls_variant[key] = value
                hparam_count += 1
                debug_print(f"get_lls_variant: parsed {key}={value}")
    
    debug_print(f"get_lls_variant RETURN: parsed {hparam_count} hyperparameters, final config = {lls_variant}")
    return lls_variant


def tta_loss(outputs):
    """Compute test-time adaptation loss (entropy minimization).
    
    Args:
        outputs: Model output logits
        
    Returns:
        torch.Tensor: Mean entropy loss
    """
    # FIX #5: Use .mean() for clarity (functionally identical to .mean(0) for 1D)
    loss = softmax_entropy(outputs).mean()
    debug_print(f"tta_loss: computed entropy loss = {loss.item():.6f} from outputs shape {outputs.shape}")
    return loss


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def is_numeric(string):
    """Check if a string represents a numeric value.
    
    Args:
        string: String to check
        
    Returns:
        bool: True if string is numeric, False otherwise
    """
    if string is None:
        return False
    try:
        for char in string:
            if not (char.isdigit() or char == '.'):
                return False
        return True
    except:
        return False


def set_parameter(current_module, name, parameters):
    """Set a parameter in a module, handling nested names.
    
    IMPORTANT ARCHITECTURAL NOTE:
    This function assigns plain Tensors (not nn.Parameter) to _parameters dict.
    This is intentionally non-standard to preserve gradient flow to mask variables
    during LLS optimization. Standard approach would wrap in nn.Parameter(), but that
    creates a leaf node and severs the grad_fn chain back to the mask.
    
    This approach works because:
    1. The modified model (new_model) is only used temporarily for one forward pass
    2. We never call optimizer methods on new_model
    3. We never call .parameters() or .named_parameters() on new_model
    4. Gradient flow from outputs -> mask is preserved via the tensor computation graph
    
    Trade-offs accepted:
    - new_model.parameters() would return incomplete/incorrect results
    - Optimizer created from new_model would fail
    - This violates PyTorch's module parameter contract
    
    These are acceptable because new_model is ephemeral (used once then discarded).
    
    Args:
        current_module: The module to modify
        name: Parameter name (may be nested like 'layer1.0.conv1')
        parameters: New parameter tensor (plain Tensor, maintains gradient flow to mask)
        
    Raises:
        KeyError: If nested module path cannot be resolved
    """
    debug_print(f"set_parameter ENTRY: name='{name}', param_shape={parameters.shape}, device={parameters.device}")
    
    if '.' in name:
        # Handle nested module path
        name_split = name.split('.')
        module_name = name_split[0]
        rest_name = '.'.join(name_split[1:])
        
        debug_print(f"set_parameter: nested path detected, looking for module '{module_name}'")
        
        # FIX #2: Add error handling for missing modules
        found = False
        for children_name, children in current_module.named_children():
            if module_name == children_name:
                debug_print(f"set_parameter: found module '{module_name}', recursing with rest_name='{rest_name}'")
                set_parameter(children, rest_name, parameters)
                found = True
                break
        
        if not found:
            available_children = [n for n, _ in current_module.named_children()]
            debug_print(f"set_parameter ERROR: module '{module_name}' not found. Available: {available_children}", force=True)
            raise KeyError(
                f"set_parameter: module '{module_name}' not found in {type(current_module).__name__}. "
                f"Full path attempted: {name}. Available children: {available_children}"
            )
    else:
        # FIX #4: Add device handling - ensure tensor is on correct device
        if len(list(current_module.parameters())) > 0:
            target_device = next(current_module.parameters()).device
            if parameters.device != target_device:
                debug_print(f"set_parameter: moving parameter from {parameters.device} to {target_device}")
                parameters = parameters.to(target_device)
        
        debug_print(f"set_parameter: assigning parameter '{name}' (shape={parameters.shape})")
        # Assign plain Tensor (not nn.Parameter) to preserve gradient flow
        # See docstring for detailed explanation of why this is intentional
        current_module._parameters[name] = parameters
        debug_print(f"set_parameter: successfully assigned parameter '{name}'")


def get_lls_masks(mask_algo, model, base_optimizer, iter_idx, trainable_layer_names, x=None, **kwargs):
    """Implement learnable mask-based layer selection for test-time adaptation.
    
    This function learns which layers should be adapted during test-time adaptation
    by optimizing layer-specific masks that control gradient flow or parameter updates.
    
    Args:
        mask_algo: String configuration for LLS (e.g., "LLS_init:rgn_lr:0.01")
        model: The neural network model being adapted
        base_optimizer: Optimizer used for model adaptation
        iter_idx: Current iteration index
        trainable_layer_names: List of layer names that can be trained
        x: Input tensor for forward pass and loss computation
        **kwargs: Additional keyword arguments
        
    Returns:
        mask_values: Learning rate scaling values for each parameter
        info: Dictionary with diagnostic information
    """
    global _lls_storage
    
    debug_print("=" * 80)
    debug_print("LLS ENTRY POINT")
    debug_print(f"Iteration: {iter_idx}, Input shape: {x.shape if x is not None else 'None'}")
    debug_print(f"mask_algo: '{mask_algo}'")
    debug_print(f"trainable_layer_names: {trainable_layer_names}")
    debug_print(f"kwargs keys: {list(kwargs.keys())}")
    
    try:
        # ==================== INITIALIZATION AND VALIDATION ====================
        
        if x is None:
            error_msg = "[LLS] ERROR: Input tensor required for LLS but not provided"
            print(error_msg)
            raise ValueError(error_msg)
        
        debug_print(f"Input tensor: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        debug_print(f"Input stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
        
        if trainable_layer_names is not None:
            original_count = len(trainable_layer_names)
            trainable_layer_names = [get_clean_layer_name(name) for name in trainable_layer_names]
            debug_print(f"Cleaned {original_count} trainable layer names")
        
        device = next(model.parameters()).device
        debug_print(f"Model device: {device}")
        
        # ==================== MODEL STORAGE MANAGEMENT ====================
        
        model_id = id(model)
        debug_print(f"Model ID: {model_id}")
        
        if model_id not in _lls_storage:
            _lls_storage[model_id] = {}
            debug_print(f"Initialized storage for model {model_id}")
        storage = _lls_storage[model_id]
        
        debug_print(f"Current storage keys: {list(storage.keys())}")
        
        if iter_idx == 0 and 'prev_mask' in storage:
            debug_print("iter_idx=0 and storage exists, clearing storage")
            storage.clear()
        
        # ==================== CONFIGURATION PARSING ====================
        
        lls_variant = get_lls_variant(mask_algo)
        
        # Print configuration
        debug_print("LLS CONFIGURATION:")
        debug_print(f"  Strategy: {'MODEL_INTERPOLATION' if lls_variant['update'] else 'GRADIENT_MASKING'}")
        debug_print(f"  Init: {lls_variant['init']}, Steps: {lls_variant['steps']}, LR: {lls_variant['lr']}")
        debug_print(f"  Granularity: {lls_variant['lvl']}, nbox: {lls_variant['nbox']}, bin: {lls_variant['bin']}")
        debug_print(f"  Application: {lls_variant['appl']}, Regularization: {lls_variant['reg']} (strength={lls_variant['regstr']})")
        debug_print(f"  Loss: {lls_variant['loss']}, Boundary: {lls_variant['bnd']}, Patience: {lls_variant['patience']}")
        
        # ==================== CONTINUE WITH NORMAL LLS LOGIC ====================
        
        debug_print("Continuing with normal LLS execution")
        
        init_value = str(lls_variant['init']).strip().lower()
        used_least_init = (init_value == 'least')
        debug_print(f"Initialization method: '{init_value}', used_least_init={used_least_init}")

        granularity = lls_variant['lvl']
        box_names = get_all_param_name_list(model, granularity)
        debug_print(f"Model boxes ({len(box_names)} at {granularity} level): {box_names}")
        
        # ==================== PARAMETER MAPPING ====================
        # FIX: Only track trainable params for efficiency
        debug_print("Building parameter to box mapping...")
        
        param_to_box_idx = {}
        trainable_params_list = []
        trainable_param_names = []
        
        total_params = 0
        trainable_count = 0
        mapped_count = 0
        
        for idx, (name, param) in enumerate(model.named_parameters()):
            total_params += 1
            if param.requires_grad and (trainable_layer_names is None or 
                                       get_clean_layer_name(name) in trainable_layer_names):
                trainable_count += 1
                trainable_params_list.append(param)
                trainable_param_names.append(name)
                
                for box_idx, box_name in enumerate(box_names):
                    if box_name in name:
                        param_to_box_idx[name] = box_idx
                        mapped_count += 1
                        break
                else:
                    param_to_box_idx[name] = -1
        
        debug_print(f"Parameter mapping: {total_params} total, {trainable_count} trainable, {mapped_count} mapped to boxes")
        debug_print(f"Trainable params list: {len(trainable_params_list)} parameters")
        
        # ==================== MASK INITIALIZATION ====================
        
        # PRESERVED FIX: Keep duplicate init_value assignment (from revert)
        init_value = str(lls_variant['init']).strip().lower()
        
        debug_print("=" * 50)
        debug_print(f"MASK INITIALIZATION: Using '{init_value}' method")
        debug_print("=" * 50)
        
        if init_value == 'least':
            # Use LEAST's defaults, not LLS's defaults
            # Only pass 'lvl' to LEAST and let it use all its own defaults:
            least_mask_algo = f"LEAST_lvl:{lls_variant['lvl']}"
            
            debug_print(f"Calling LEAST with: {least_mask_algo}")
            debug_print("IMPORTANT: LEAST will use ALL its own defaults (bin=1, nbox=1, mthr=0.75, etc.)")
            
            try:
                _, least_info = get_LEAST_masks(least_mask_algo, model, base_optimizer, iter_idx, 
                                              trainable_layer_names, track_flag=True, x=x, **kwargs)
                
                if 'mask_dict' not in least_info:
                    error_msg = "LEAST didn't return 'mask_dict'"
                    print(error_msg)
                    raise KeyError(error_msg)
                
                # Extract raw metrics and final masks from LEAST
                metric_values_dict = least_info.get('metric_values_dict', {})
                box_masks = least_info['mask_dict']
                
                # Log LEAST results
                debug_print("LEAST RETURNED:")
                debug_print("  Raw metrics (cosine similarities before processing):")
                for box_name in box_names:
                    if box_name in metric_values_dict:
                        val = metric_values_dict[box_name]
                        # FIX #4: Use ensure_float helper for consistent type handling
                        val_float = ensure_float(val)
                        debug_print(f"    {box_name}: {val_float:.6f}")
                
                debug_print("  Final masks (after nbox + bin + trust processing):")
                for box_name in box_names:
                    if box_name in box_masks:
                        val = box_masks[box_name]
                        # FIX #4: Use ensure_float helper
                        val_float = ensure_float(val)
                        debug_print(f"    {box_name}: {val_float:.6f}")
                
                # Check box name matching
                boxes_found = sum(1 for b in box_names if b in box_masks)
                boxes_defaulted = sum(1 for b in box_names if b not in box_masks)
                debug_print(f"  Box matching: {boxes_found}/{len(box_names)} found, {boxes_defaulted} defaulted to 0.5")
                if boxes_defaulted > 0:
                    defaulted_list = [b for b in box_names if b not in box_masks]
                    debug_print(f"  WARNING: Defaulted boxes: {defaulted_list}")
                
                # Create initial masks
                debug_print("Creating initial mask ParameterList...")
                init_mask = nn.ParameterList()
                box_metrics = {}
                
                for box_name in box_names:
                    if box_name in box_masks:
                        # FIX #4: Use ensure_float helper
                        value = ensure_float(box_masks[box_name])
                        value = float(max(0.0, min(1.0, value)))
                        init_mask.append(nn.Parameter(torch.tensor([value], device=device), requires_grad=True))
                        box_metrics[box_name] = value
                        debug_print(f"  Created mask for '{box_name}': {value:.6f}")
                    else:
                        value = 0.5
                        init_mask.append(nn.Parameter(torch.tensor([value], device=device), requires_grad=True))
                        box_metrics[box_name] = value
                        debug_print(f"  Created default mask for '{box_name}': {value:.6f}")
                
                debug_print("INITIAL MASKS CREATED:")
                for i, box_name in enumerate(box_names):
                    debug_print(f"  {box_name}: {init_mask[i].item():.6f} (requires_grad={init_mask[i].requires_grad})")
                
            except Exception as e:
                error_msg = f"ERROR in LEAST initialization: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                raise ValueError(error_msg)
        
        elif init_value in ['rgn', 'rgnbin']:
            # ==================== FIX #2: USE torch.autograd.grad FOR RGN ====================
            # RGN initialization needs entropy gradients to calculate relative gradient norms.
            # Using torch.autograd.grad instead of .backward() ensures no mutation of model.grad
            debug_print("RGN initialization using torch.autograd.grad (no .grad mutation)")
            
            # Use RGN (Relative Gradient Norm) for initialization
            debug_print("Computing forward pass for RGN...")
            outputs = model(x)
            debug_print(f"  Outputs shape: {outputs.shape}, device: {outputs.device}")
            
            loss = tta_loss(outputs)
            debug_print(f"  Entropy loss for RGN: {loss.item():.6f}")
            
            # Compute gradients as tensors (NO mutation of .grad)
            debug_print("Computing entropy gradients via torch.autograd.grad...")
            entropy_grads_tuple = torch.autograd.grad(
                outputs=loss,
                inputs=trainable_params_list,
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            
            debug_print(f"  Received {len(entropy_grads_tuple)} gradient tensors")
            
            # Map gradients to parameter names
            entropy_grads = {}
            none_count = 0
            for name, grad_tensor in zip(trainable_param_names, entropy_grads_tuple):
                if grad_tensor is not None:
                    entropy_grads[name] = grad_tensor
                else:
                    # Parameter wasn't used in loss computation
                    entropy_grads[name] = torch.zeros_like(
                        dict(model.named_parameters())[name]
                    )
                    none_count += 1
            
            debug_print(f"  Computed entropy gradients for {len(entropy_grads)} params ({none_count} were unused)")
            debug_print("  CRITICAL: Caller's model.grad UNCHANGED (no mutation)")
            
            # Compute RGN per box
            debug_print("Computing RGN per box...")
            box_rgn = {}
            for box_idx in range(len(box_names)):
                box_rgn[box_idx] = []
            
            # PRESERVED FIX: Keep detailed RGN tracking with param_rgn_details (from revert)
            param_rgn_details = []
            for name in trainable_param_names:
                if name in param_to_box_idx and param_to_box_idx[name] >= 0:
                    box_idx = param_to_box_idx[name]
                    param = dict(model.named_parameters())[name]
                    grad = entropy_grads[name]
                    
                    if grad.norm() > 0 and param.data.norm() > 0:
                        grad_norm = grad.norm().item()
                        param_norm = param.data.norm().item()
                        rgn = grad.norm() / param.data.norm()
                        box_rgn[box_idx].append(rgn)
                        param_rgn_details.append((name, grad_norm, param_norm, rgn.item()))
            
            debug_print(f"Calculated RGN for {len(param_rgn_details)} parameters")
            if len(param_rgn_details) > 0:
                debug_print(f"Sample RGN: {param_rgn_details[0][0]} = {param_rgn_details[0][3]:.6f}")
                debug_print(f"  (grad_norm={param_rgn_details[0][1]:.6f}, param_norm={param_rgn_details[0][2]:.6f})")
            
            # Compute average RGN per box
            debug_print("Creating initial masks from RGN values...")
            init_mask = nn.ParameterList()
            box_metrics = {}
            
            debug_print("RGN per box (before normalization):")
            for box_idx in range(len(box_names)):
                if box_idx in box_rgn and len(box_rgn[box_idx]) > 0:
                    rgn_list = [r.item() for r in box_rgn[box_idx]]
                    avg_rgn = sum(rgn_list) / len(rgn_list)
                    debug_print(f"  {box_names[box_idx]}: {avg_rgn:.6f} (from {len(rgn_list)} params)")
                    value = torch.tensor(avg_rgn, device=device)
                    box_metrics[box_names[box_idx]] = avg_rgn
                else:
                    debug_print(f"  {box_names[box_idx]}: 0.5 (default, no valid params)")
                    value = torch.tensor(0.5, device=device)
                    box_metrics[box_names[box_idx]] = 0.5
                
                init_mask.append(nn.Parameter(value.view(1), requires_grad=True))
            
            # Normalize RGN values
            debug_print("Normalizing RGN masks...")
            mask_values = [m.item() for m in init_mask]
            if any(v > 0 for v in mask_values):
                max_value = max(mask_values)
                debug_print(f"  Max RGN value: {max_value:.6f}")
                if max_value > 0:
                    for i, m in enumerate(init_mask):
                        init_mask[i].data = m.data / max_value
            
            # Binarize if requested
            if init_value == 'rgnbin':
                debug_print("Binarizing RGN masks (threshold=0.5)")
                bin_count_one = 0
                bin_count_zero = 0
                for i, m in enumerate(init_mask):
                    original = m.item()
                    init_mask[i].data = (m.data > 0.5).float()
                    if init_mask[i].item() > 0.5:
                        bin_count_one += 1
                    else:
                        bin_count_zero += 1
                    debug_print(f"  {box_names[i]}: {original:.6f} -> {init_mask[i].item():.6f}")
                debug_print(f"  Binarization result: {bin_count_one} boxes set to 1.0, {bin_count_zero} set to 0.0")
            
            debug_print("INITIAL MASKS AFTER NORMALIZATION:")
            for i, box_name in enumerate(box_names):
                debug_print(f"  {box_name}: {init_mask[i].item():.6f}")
        
        elif init_value == 'prev' and 'prev_mask' in storage:
            # Use previous iteration's masks
            debug_print("Using previous iteration masks from storage")
            # FIX #5: Load from detached storage (will be list of floats, not Parameters)
            prev_mask_values = storage['prev_mask']
            debug_print(f"  Loaded {len(prev_mask_values)} previous mask values")
            
            init_mask = nn.ParameterList()
            box_metrics = {}
            
            for i, box_name in enumerate(box_names):
                if i < len(prev_mask_values):
                    value = prev_mask_values[i]
                    init_mask.append(nn.Parameter(torch.tensor([value], device=device), requires_grad=True))
                    box_metrics[box_name] = value
                    debug_print(f"  Restored mask for '{box_name}': {value:.6f}")
                else:
                    # New box not in previous mask
                    value = 0.5
                    init_mask.append(nn.Parameter(torch.tensor([value], device=device), requires_grad=True))
                    box_metrics[box_name] = value
                    debug_print(f"  Adding new box '{box_name}' with default 0.5")
        
        else:
            # Default initialization
            debug_print(f"Using default initialization (method='{init_value}')")
            init_mask = nn.ParameterList()
            box_metrics = {}
            
            if is_numeric(lls_variant['finit']):
                init_value_num = float(lls_variant['finit'])
                debug_print(f"Using constant initialization: {init_value_num}")
            elif lls_variant['finit'] == 'rand':
                init_value_num = None
                debug_print("Using random initialization")
            else:
                init_value_num = 0.5
                debug_print("Using default initialization: 0.5")
            
            for i, box_name in enumerate(box_names):
                if init_value_num is None:
                    value = torch.rand(1, device=device).item()
                    init_mask.append(nn.Parameter(torch.tensor([value], device=device), requires_grad=True))
                    box_metrics[box_name] = value
                    debug_print(f"  Random mask for '{box_name}': {value:.6f}")
                else:
                    init_mask.append(nn.Parameter(torch.tensor([init_value_num], device=device), requires_grad=True))
                    box_metrics[box_name] = init_value_num
                    debug_print(f"  Constant mask for '{box_name}': {init_value_num:.6f}")
        
        debug_print(f"Initialization complete: created {len(init_mask)} mask parameters")
        
        # ==================== MODEL PREPARATION ====================
        
        debug_print("=" * 50)
        debug_print("MODEL PREPARATION")
        debug_print("=" * 50)
        
        debug_print("Creating deep copy of model as pre_model...")
        pre_model = copy.deepcopy(model)
        debug_print("  pre_model created successfully")
        
        if lls_variant['update'] == True:
            # Model interpolation: create finetuned model
            debug_print("MODEL INTERPOLATION: Creating finetuned model")
            
            debug_print("  Copying model for finetuning...")
            ft_model = copy.deepcopy(model)
            
            debug_print("  Setting up finetuning optimizer...")
            base_opt_params = copy.deepcopy(base_optimizer.defaults)
            debug_print(f"  Base optimizer params: {base_opt_params}")
            
            ft_opt = type(base_optimizer)(ft_model.parameters(), **base_opt_params)
            debug_print(f"  Created finetuning optimizer: {type(ft_opt).__name__}")
            
            # PRESERVED FIX: Use original ft_loss structure (from revert)
            debug_print("  Computing loss for finetuning...")
            ft_outputs = ft_model(x)
            debug_print(f"  ft_outputs shape: {ft_outputs.shape}")
            
            ft_loss = tta_loss(ft_outputs)
            debug_print(f"  Initial finetuned loss: {ft_loss.item():.6f}")
            
            debug_print("  Performing gradient descent step...")
            ft_opt.zero_grad()
            ft_loss.backward()
            ft_opt.step()
            
            debug_print("  Measuring loss after finetuning step...")
            with torch.no_grad():
                ft_outputs_after = ft_model(x)
                ft_loss_after = tta_loss(ft_outputs_after).item()
            
            debug_print(f"  Loss after SGD step: {ft_loss_after:.6f}, delta: {ft_loss_after - ft_loss.item():.6f}")
            
            if ft_loss_after < ft_loss.item():
                debug_print("  SUCCESS: Finetuning reduced loss")
            else:
                debug_print("  WARNING: Finetuning increased loss")
        
        else:
            # ==================== FIX #3: USE torch.autograd.grad FOR ENTROPY GRADS ====================
            # Gradient masking path needs entropy gradients to simulate masked updates.
            # Using torch.autograd.grad instead of .backward() ensures no mutation of model.grad
            debug_print("GRADIENT MASKING: Computing entropy gradients via torch.autograd.grad")
            
            debug_print("  Computing forward pass...")
            outputs = model(x)
            debug_print(f"  Outputs shape: {outputs.shape}")
            
            loss = tta_loss(outputs)
            debug_print(f"  Entropy loss: {loss.item():.6f}")
            
            debug_print("  Computing gradients via torch.autograd.grad...")
            # Compute gradients as tensors (NO mutation of .grad)
            entropy_grads_tuple = torch.autograd.grad(
                outputs=loss,
                inputs=trainable_params_list,
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            
            debug_print(f"  Received {len(entropy_grads_tuple)} gradient tensors")
            
            # Map to dictionary
            grads = {}
            grad_count = 0
            none_count = 0
            for name, grad_tensor in zip(trainable_param_names, entropy_grads_tuple):
                if grad_tensor is not None:
                    grads[name] = grad_tensor
                    grad_count += 1
                else:
                    grads[name] = torch.zeros_like(
                        dict(model.named_parameters())[name]
                    )
                    none_count += 1
            
            debug_print(f"  Collected {grad_count} non-None gradients, {none_count} unused parameters")
            debug_print("  CRITICAL: Caller's model.grad UNCHANGED (no mutation)")
        
        # ==================== MASK OPTIMIZATION ====================
        
        debug_print("=" * 50)
        debug_print("MASK OPTIMIZATION")
        debug_print("=" * 50)
        
        debug_print("MASK OPTIMIZATION SETUP:")
        debug_print(f"  Max steps: {lls_variant['steps']}, Patience: {lls_variant['patience']}, Mask LR: {lls_variant['lr']}")
        debug_print(f"  Regularization: {lls_variant['reg']} (strength={lls_variant['regstr']})")
        debug_print(f"  Boundary handling: {lls_variant['bnd']}")
        
        best_mask = copy.deepcopy(init_mask)
        best_loss = float('inf')
        debug_print(f"  Initialized best_loss to infinity")
        
        patience = int(lls_variant['patience'])
        patience_counter = patience
        debug_print(f"  Patience counter initialized to {patience}")
        
        mask_lr = float(lls_variant['lr'])
        # FIX: Use list(init_mask) instead of init_mask directly
        debug_print(f"  Creating Adam optimizer for masks with lr={mask_lr}")
        mask_opt = torch.optim.Adam(list(init_mask), lr=mask_lr)
        debug_print(f"  Mask optimizer created: {len(list(init_mask))} parameters")
        
        max_steps = int(lls_variant['steps'])
        debug_print(f"  Will run for maximum {max_steps} optimization steps")
        
        # Print initial masks before optimization
        debug_print("Initial masks (before optimization):")
        for i, box_name in enumerate(box_names):
            debug_print(f"  {box_name}: {init_mask[i].item():.6f}")
        
        # Optimization loop - only print for first 2 steps and last step
        debug_print("Starting optimization loop...")
        
        for step in range(max_steps):
            if step <= 1 or step == max_steps - 1:
                debug_print(f"--- Step {step}/{max_steps} ---")
            
            # Retain gradients for inspection
            for param in init_mask:
                param.retain_grad()
            
            # Apply mask to create modified model
            if lls_variant['update'] == True:
                # Model interpolation
                if step <= 1 or step == max_steps - 1:
                    debug_print(f"  Creating interpolated model...")
                
                new_model = copy.deepcopy(pre_model)
                
                pre_params_dict = dict(pre_model.named_parameters())
                ft_params_dict = dict(ft_model.named_parameters())
                
                interpolated_params = 0
                for name in param_to_box_idx.keys():
                    if param_to_box_idx[name] >= 0:
                        box_idx = param_to_box_idx[name]
                        mask = init_mask[box_idx]
                        
                        pre_param = pre_params_dict[name]
                        ft_param = ft_params_dict[name]
                        
                        pre_param_detached = pre_param.detach()
                        ft_param_detached = ft_param.detach()
                        
                        new_param_value = (1 - mask) * pre_param_detached + mask * ft_param_detached
                        
                        set_parameter(new_model, name, new_param_value)
                        interpolated_params += 1
                
                if step <= 1 or step == max_steps - 1:
                    debug_print(f"  Interpolated {interpolated_params} parameters")
                    
            else:
                # Gradient masking
                if step <= 1 or step == max_steps - 1:
                    debug_print(f"  Creating gradient-masked model...")
                
                new_model = copy.deepcopy(pre_model)
                
                pre_params_dict = dict(pre_model.named_parameters())
                base_lr = base_optimizer.defaults['lr']
                
                if step <= 1 or step == max_steps - 1:
                    debug_print(f"  Using base_lr={base_lr} for gradient masking")
                
                masked_params = 0
                for name in param_to_box_idx.keys():
                    if param_to_box_idx[name] >= 0 and name in grads:
                        box_idx = param_to_box_idx[name]
                        mask = init_mask[box_idx]
                        
                        param = pre_params_dict[name]
                        grad = grads[name]
                        
                        param_detached = param.detach()
                        grad_detached = grad.detach()
                        
                        new_param_value = param_detached - mask * base_lr * grad_detached
                        
                        set_parameter(new_model, name, new_param_value)
                        masked_params += 1
                
                if step <= 1 or step == max_steps - 1:
                    debug_print(f"  Masked {masked_params} parameters")
            
            # Calculate loss for the current mask
            if lls_variant['loss'] == 'prev' and 'prev_x' in storage:
                if step <= 1 or step == max_steps - 1:
                    debug_print(f"  Using previous input from storage")
                outputs = new_model(storage['prev_x'])
            else:
                outputs = new_model(x)
                
            mask_loss = tta_loss(outputs)
            
            # Add regularization
            reg_loss = 0
            if lls_variant['reg'] == 'l1':
                reg_strength = float(lls_variant['regstr'])
                reg_loss = reg_strength * sum(param.abs().sum() for param in init_mask)
            elif lls_variant['reg'] == 'l2':
                reg_strength = float(lls_variant['regstr'])
                reg_loss = reg_strength * sum((param ** 2).sum() for param in init_mask)
            
            total_loss = mask_loss + reg_loss
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                if step <= 1 or step == max_steps - 1:
                    debug_print(f"  WARNING: Invalid loss (NaN or Inf), skipping step")
                continue
            
            # Update mask
            mask_opt.zero_grad()
            total_loss.backward()
            
            # Check for NaN or Inf gradients
            has_bad_grad = False
            if step <= 1 or step == max_steps - 1:
                for idx, param in enumerate(init_mask):
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            debug_print(f"  WARNING: Bad gradient for mask {idx}, zeroing it")
                            param.grad.zero_()
                            has_bad_grad = True
            else:
                for idx, param in enumerate(init_mask):
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            param.grad.zero_()
                            has_bad_grad = True
            
            if not has_bad_grad:
                mask_opt.step()
            elif step <= 1 or step == max_steps - 1:
                debug_print(f"  Skipped optimizer step due to bad gradients")
            
            # Print detailed info for first 2 steps and last step
            if step <= 1 or step == max_steps - 1:
                debug_print(f"Step {step}:")
                debug_print(f"  Mask loss (entropy): {mask_loss.item():.6f}, Reg loss: {reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss:.6f}, Total: {total_loss.item():.6f}")
                debug_print(f"  Mask values: {[f'{m.item():.4f}' for m in init_mask]}")
                if step <= 1:
                    debug_print(f"  Mask gradients: {[f'{m.grad.item():.6f}' if m.grad is not None else 'None' for m in init_mask]}")
            
            # Apply bounds to mask values
            if lls_variant['bnd'] == 'clip':
                for idx, param in enumerate(init_mask):
                    init_mask[idx].data.clamp_(0, 1)
            elif lls_variant['bnd'] == 'scale':
                all_values = torch.cat([param.view(-1) for param in init_mask])
                min_val = all_values.min()
                max_val = all_values.max()
                if max_val > min_val:
                    for idx, param in enumerate(init_mask):
                        scaled_value = (param.data - min_val) / (max_val - min_val)
                        init_mask[idx].data = scaled_value
                    if step <= 1 or step == max_steps - 1:
                        debug_print(f"  Applied scaling: min={min_val.item():.6f}, max={max_val.item():.6f}")
            
            # Update best mask if loss improved
            if mask_loss.item() < best_loss:
                best_mask = copy.deepcopy(init_mask)
                best_loss = mask_loss.item()
                patience_counter = patience
                
                if step <= 1 or step == max_steps - 1:
                    debug_print(f"  NEW BEST LOSS: {best_loss:.6f}")
            else:
                patience_counter -= 1
                
                if patience_counter <= 0:
                    debug_print(f"Early stopping at step {step}, best loss: {best_loss:.6f}")
                    break
        
        debug_print(f"Optimization complete - Final best loss: {best_loss:.6f}")
        
        # ==================== POST-PROCESSING ====================
        
        debug_print("=" * 50)
        debug_print("POST-PROCESSING")
        debug_print("=" * 50)
        
        skip_nbox_bin = (used_least_init and max_steps == 0)
        
        if skip_nbox_bin:
            debug_print("POST-PROCESSING: Skipping nbox/bin (LEAST init with no optimization)")
        else:
            debug_print("POST-PROCESSING:")
            
            if lls_variant['nbox'] == '1':
                debug_print("  Applying nbox=1 constraint...")
                max_idx = -1
                max_value = -float('inf')
                for i, param in enumerate(best_mask):
                    if param.item() > max_value:
                        max_value = param.item()
                        max_idx = i
                
                if max_idx >= 0:
                    debug_print(f"  nbox=1: Keeping only box {max_idx} ({box_names[max_idx]}) = {max_value:.6f}")
                    for i, param in enumerate(best_mask):
                        if i != max_idx:
                            best_mask[i].data = torch.tensor([0.0], device=device)
                            debug_print(f"    Zeroed box {i} ({box_names[i]})")
            
            if lls_variant['bin'] == '1':
                threshold = float(lls_variant['mthr']) if 'mthr' in lls_variant else 0.5
                debug_print(f"  Binarizing masks (threshold={threshold})")
                one_count = 0
                zero_count = 0
                for i, param in enumerate(best_mask):
                    original = param.item()
                    binary_value = 1.0 if param.item() >= threshold else 0.0
                    best_mask[i].data = torch.tensor([binary_value], device=device)
                    if binary_value > 0.5:
                        one_count += 1
                    else:
                        zero_count += 1
                    debug_print(f"    {box_names[i]}: {original:.6f} -> {binary_value:.6f}")
                debug_print(f"  Binarization result: {one_count} set to 1.0, {zero_count} set to 0.0")
        
        # ==================== MASK APPLICATION ====================
        
        debug_print("=" * 50)
        debug_print("MASK APPLICATION")
        debug_print("=" * 50)
        
        debug_print(f"MASK APPLICATION: Strategy={lls_variant['appl']}")
        
        if lls_variant['appl'] == 'curr' or 'prev_mask' not in storage:
            final_mask = best_mask
            debug_print("  Using current masks (no temporal blending)")
        elif lls_variant['appl'] == 'avg':
            count = storage.get('count', 0)
            # FIX #5: Load from detached storage (list of floats)
            prev_mask_values = storage['prev_mask']
            final_mask = copy.deepcopy(best_mask)
            debug_print(f"  Averaging with previous (count={count})")
            
            for i in range(len(best_mask)):
                if i < len(box_names) and i < len(prev_mask_values):
                    curr_val = best_mask[i].item()
                    prev_val = prev_mask_values[i]
                    avg_value = (count * prev_val + curr_val) / (count + 1)
                    final_mask[i].data = torch.tensor([avg_value], device=device)
                    debug_print(f"    {box_names[i]}: curr={curr_val:.6f}, prev={prev_val:.6f}, avg={avg_value:.6f}")
        elif lls_variant['appl'] == 'prod':
            # FIX #5: Load from detached storage (list of floats)
            prev_mask_values = storage['prev_mask']
            final_mask = copy.deepcopy(best_mask)
            debug_print("  Computing product with previous")
            
            for i in range(len(best_mask)):
                if i < len(box_names) and i < len(prev_mask_values):
                    curr_val = best_mask[i].item()
                    prev_val = prev_mask_values[i]
                    prod_value = curr_val * prev_val
                    final_mask[i].data = torch.tensor([prod_value], device=device)
                    debug_print(f"    {box_names[i]}: curr={curr_val:.6f}, prev={prev_val:.6f}, prod={prod_value:.6f}")
        elif lls_variant['appl'] == 'ema':
            # FIX #5: Load from detached storage (list of floats)
            prev_mask_values = storage['prev_mask']
            alpha = float(lls_variant['emalr'])
            final_mask = copy.deepcopy(best_mask)
            debug_print(f"  Computing EMA (alpha={alpha})")
            
            for i in range(len(best_mask)):
                if i < len(box_names) and i < len(prev_mask_values):
                    curr_val = best_mask[i].item()
                    prev_val = prev_mask_values[i]
                    ema_value = (1 - alpha) * prev_val + alpha * curr_val
                    final_mask[i].data = torch.tensor([ema_value], device=device)
                    debug_print(f"    {box_names[i]}: curr={curr_val:.6f}, prev={prev_val:.6f}, ema={ema_value:.6f}")
        else:
            final_mask = best_mask
            debug_print("  Using best mask directly (unknown strategy)")
        
        # FIX #5: Store masks as plain floats (not ParameterList) using helper
        debug_print("Saving final masks to storage...")
        storage['prev_mask'] = [ensure_float(m) for m in final_mask]
        storage['count'] = storage.get('count', 0) + 1
        storage['prev_x'] = x.detach().clone()
        debug_print(f"  Stored {len(storage['prev_mask'])} mask values, count={storage['count']}")
        
        # ==================== RETURN VALUES ====================
        
        debug_print("=" * 50)
        debug_print("RETURN VALUE PREPARATION")
        debug_print("=" * 50)
        
        debug_print("Converting box masks to parameter-level masks...")
        lr_values = []
        for name, param in model.named_parameters():
            if param.requires_grad and (trainable_layer_names is None or 
                                       get_clean_layer_name(name) in trainable_layer_names):
                box_idx = param_to_box_idx.get(name, -1)
                if box_idx >= 0 and box_idx < len(final_mask):
                    lr_value = final_mask[box_idx].item()
                    lr_values.append(lr_value)
                else:
                    lr_values.append(0.0)
        
        debug_print(f"Created {len(lr_values)} parameter-level mask values")
        
        info = {
            'best_loss': best_loss,
            # FIX #4: Use ensure_float for consistent type in info dict
            'mask_dict': {box_names[i]: ensure_float(final_mask[i]) 
                         for i in range(min(len(box_names), len(final_mask)))},
            'init_metrics': box_metrics,
            'strategy': 'model_interpolation' if lls_variant['update'] else 'gradient_masking',
        }
        
        debug_print("Info dictionary created:")
        debug_print(f"  best_loss: {info['best_loss']:.6f}")
        debug_print(f"  mask_dict: {len(info['mask_dict'])} boxes")
        debug_print(f"  init_metrics: {len(info['init_metrics'])} boxes")
        debug_print(f"  strategy: {info['strategy']}")
        
        debug_print("FINAL MASKS:")
        for i, box_name in enumerate(box_names):
            if i < len(final_mask):
                debug_print(f"  {box_name}: {final_mask[i].item():.6f}")
        
        active_params = sum(1 for v in lr_values if v > 0)
        inactive_params = sum(1 for v in lr_values if v == 0)
        debug_print(f"RETURN: {len(lr_values)} parameter masks, Active: {active_params}, Inactive: {inactive_params}")
        
        debug_print("=" * 80)
        
        return lr_values, info
        
    except Exception as e:
        print(f"[LLS_ERROR] CRITICAL ERROR in get_lls_masks: {str(e)}")
        print(traceback.format_exc())
        raise

def get_auto_rgn_values(param_groups):
    """Compute automatic Relative Gradient Norm (RGN) values.
    FIX: Handle case where no parameters have gradients (empty list).
    If rgn_values is empty, torch.tensor([]).max() raises RuntimeError.
    Args:
        param_groups: Optimizer parameter groups
    Returns:
        torch.Tensor: RGN values for each parameter (empty tensor if no grads)
    """
    debug_print(f"get_auto_rgn_values ENTRY: {len(param_groups)} parameter groups")
    
    rgn_values = []
    
    for group_idx, group in enumerate(param_groups):
        for p_idx, p in enumerate(group['params']):
            if p.grad is None:
                continue
            grad_norm = p.grad.norm()
            param_norm = p.norm()
            rgn = grad_norm / (param_norm + 1e-8)
            rgn_values.append(rgn)

    # CRITICAL FIX #3: Handle empty list case
    if len(rgn_values) == 0:
        debug_print("get_auto_rgn_values: No parameters with gradients, returning empty tensor", force=True)
        return torch.tensor([], dtype=torch.float32)
    
    debug_print(f"  Computed {len(rgn_values)} RGN values")
    
    # FIX #6: Use torch.stack for semantic correctness
    rgn_values = torch.stack(rgn_values)
    
    # Normalize by max - safe because tensor is non-empty
    max_val = rgn_values.max()
    if max_val > 0:
        rgn_values = (rgn_values) / (max_val + 1e-8)
        debug_print(f"  Normalized by max={max_val.item():.6f}")
    else:
        debug_print("  Warning: All RGN values are zero, no normalization applied", force=True)
    
    debug_print(f"get_auto_rgn_values RETURN: {len(rgn_values)} values")
    return rgn_values

def get_auto_snr_values(param_groups):
    """Compute automatic Signal-to-Noise Ratio (SNR) values.
    FIX: Handle case where no parameters have gradients (empty list).
    If snr_values is empty, subsequent operations will fail.
    Args:
        param_groups: Optimizer parameter groups
    Returns:
        torch.Tensor: SNR values for each parameter (empty tensor if no grads)
    """
    debug_print(f"get_auto_snr_values ENTRY: {len(param_groups)} parameter groups")
    
    snr_values = []
    
    for group in param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            snr = (grad*grad) / (torch.var(grad, dim=0, keepdim=True)+1e-8)
            snr = snr.mean()
            snr_values.append(snr)

    # FIX #3: Handle empty list case
    if len(snr_values) == 0:
        debug_print("get_auto_snr_values: No parameters with gradients, returning empty tensor", force=True)
        return torch.tensor([], dtype=torch.float32)
    
    debug_print(f"  Computed {len(snr_values)} SNR values")
    
    # FIX #6: Use torch.stack for semantic correctness
    snr_values = torch.stack(snr_values)
    
    # Apply binarization threshold
    snr_values = torch.where(snr_values < 0.95,
                             torch.zeros_like(snr_values),
                             torch.ones_like(snr_values))
    
    ones_count = (snr_values == 1).sum().item()
    zeros_count = (snr_values == 0).sum().item()
    debug_print(f"  Binarized: {ones_count} ones, {zeros_count} zeros")
    
    debug_print(f"get_auto_snr_values RETURN: {len(snr_values)} values")
    return snr_values

def diff_lr_optimizer(base_optimizer):
    """Create a new optimizer with separate learning rate for each parameter.
    Args:
        base_optimizer: Original optimizer
    Returns:
        Optimizer: New optimizer with per-parameter learning rates
    """
    debug_print("diff_lr_optimizer ENTRY")
    
    base_opt_params = base_optimizer.defaults
    base_lr = base_opt_params['lr']
    debug_print(f"  Base optimizer: {type(base_optimizer).__name__}, base_lr={base_lr}")

    params = []
    total_params = 0
    for group_idx, group in enumerate(base_optimizer.param_groups):
        for p_idx, p in enumerate(group['params']):
            # FIX: Wrap p in list to comply with PyTorch API
            params.append({'params': [p], 'lr': base_lr})
            total_params += 1

    debug_print(f"  Created {len(params)} parameter groups for {total_params} parameters")
    
    new_optimizer = type(base_optimizer)(params, **base_opt_params)
    
    # Verification: Ensure parameter count is preserved
    new_param_count = sum(len(g['params']) for g in new_optimizer.param_groups)
    if new_param_count != total_params:
        error_msg = f"Parameter count mismatch: expected {total_params}, got {new_param_count}"
        debug_print(f"diff_lr_optimizer ERROR: {error_msg}", force=True)
        raise RuntimeError(error_msg)
    
    debug_print(f"diff_lr_optimizer RETURN: {type(new_optimizer).__name__} with {len(params)} param groups")
    return new_optimizer

def diff_lr(base_optimizer, lr_factor_values):
    """Apply layer-specific learning rates to optimizer.
    
    Args:
        base_optimizer: Optimizer to modify
        lr_factor_values: List of learning rate scaling factors
    """
    debug_print(f"diff_lr ENTRY: {len(base_optimizer.param_groups)} param groups, {len(lr_factor_values)} lr factors")
    
    if len(base_optimizer.param_groups) != len(lr_factor_values):
        error_msg = f'Length mismatch: {len(lr_factor_values)} lr factors, {len(base_optimizer.param_groups)} param groups'
        debug_print(f"diff_lr ERROR: {error_msg}", force=True)
        raise ValueError(error_msg)
    
    base_lr = base_optimizer.defaults['lr']
    debug_print(f"  Base learning rate: {base_lr}")

    modified_count = 0
    for idx, group in enumerate(base_optimizer.param_groups):
        lr_layer = base_lr * lr_factor_values[idx]
        group['lr'] = lr_layer
        modified_count += 1
        if idx < 3 or idx >= len(base_optimizer.param_groups) - 3:
            debug_print(f"  Group {idx}: lr = {base_lr} * {lr_factor_values[idx]:.6f} = {lr_layer:.6f}")
    
    debug_print(f"diff_lr: Modified {modified_count} parameter groups")


def compute_worst_mask(metric_values_dict, td_variant):
    """Compute masks that select the worst box.
    
    Args:
        metric_values_dict: Dictionary of metrics for each box
        td_variant: Configuration dictionary
        
    Returns:
        dict: Dictionary of mask values for each box
    """
    debug_print(f"compute_worst_mask ENTRY: {len(metric_values_dict)} boxes")
    
    mask_dict = copy.deepcopy(metric_values_dict)

    min_mask_value = min(mask_dict.values())
    debug_print(f"  Minimum metric value: {min_mask_value.item():.6f}")
    
    selected_boxes = []
    for box_name, mask_value in mask_dict.items():
        if mask_value == min_mask_value:
            mask_dict[box_name] = torch.tensor(1.0, device=mask_value.device)
            selected_boxes.append(box_name)
        else:
            mask_dict[box_name] = torch.tensor(0.0, device=mask_value.device)
    
    debug_print(f"  Selected worst boxes: {selected_boxes}")
    debug_print(f"compute_worst_mask RETURN: {len(mask_dict)} masks")
    return mask_dict


def compute_not_best_mask(metric_values_dict, td_variant):
    """Compute masks that select all boxes except the best one.
    
    Args:
        metric_values_dict: Dictionary of metrics for each box
        td_variant: Configuration dictionary
        
    Returns:
        dict: Dictionary of mask values for each box
    """
    debug_print(f"compute_not_best_mask ENTRY: {len(metric_values_dict)} boxes")
    
    mask_dict = copy.deepcopy(metric_values_dict)

    max_mask_value = max(mask_dict.values())
    debug_print(f"  Maximum metric value: {max_mask_value.item():.6f}")
    
    metrics_sum = sum(v.item() for v in mask_dict.values())
    first_sample_in_window = abs(metrics_sum - len(mask_dict)) < 1e-6
    
    if first_sample_in_window:
        debug_print("  First sample in window detected, keeping all boxes")
    
    selected_boxes = []
    excluded_boxes = []
    for box_name, mask_value in mask_dict.items():
        if mask_value != max_mask_value or first_sample_in_window:
            mask_dict[box_name] = torch.tensor(1.0, device=mask_value.device)
            selected_boxes.append(box_name)
        else:
            mask_dict[box_name] = torch.tensor(0.0, device=mask_value.device)
            excluded_boxes.append(box_name)
    
    debug_print(f"  Selected boxes (not best): {selected_boxes}")
    debug_print(f"  Excluded boxes (best): {excluded_boxes}")
    debug_print(f"compute_not_best_mask RETURN: {len(mask_dict)} masks")
    return mask_dict


def compute_random_mask(metric_values_dict, td_variant):
    """Compute masks that select a random box.
    
    Args:
        metric_values_dict: Dictionary of metrics for each box
        td_variant: Configuration dictionary
        
    Returns:
        dict: Dictionary of mask values for each box
    """
    debug_print(f"compute_random_mask ENTRY: {len(metric_values_dict)} boxes")
    
    mask_dict = copy.deepcopy(metric_values_dict)

    box_names = list(mask_dict.keys())
    random_box = random.choice(box_names)
    debug_print(f"  Randomly selected box: {random_box}")

    for box_name, mask_value in mask_dict.items():
        if box_name == random_box:
            mask_dict[box_name] = torch.tensor(1.0, device=mask_value.device)
        else:
            mask_dict[box_name] = torch.tensor(0.0, device=mask_value.device)

    debug_print(f"compute_random_mask RETURN: {len(mask_dict)} masks")
    return mask_dict


def get_box_masks(mask_algo, model, trainable_layer_names):
    """Get masks for explicit layer selection.
    
    Args:
        mask_algo: Configuration string
        model: Neural network model
        trainable_layer_names: List of trainable layer names
        
    Returns:
        list: List of mask values for each parameter
    """
    debug_print(f"get_box_masks ENTRY: mask_algo='{mask_algo}'")
    
    parts = mask_algo.lower().split('_')
    box_parts = [part for part in parts if 'layer' in part or 'stage' in part]
    
    debug_print(f"  Extracted box parts: {box_parts}")
    
    mask_dict = {box_name: 1 for box_name in box_parts}
    debug_print(f"  Created mask_dict with {len(mask_dict)} boxes set to 1")

    mask_values = get_mask_values(mask_dict, model, trainable_layer_names)
    debug_print(f"get_box_masks RETURN: {len(mask_values)} mask values")
    return mask_values


def get_es_variant(mask_algo):
    """Parse early stopping configuration from mask_algo string.
    
    Args:
        mask_algo: Configuration string
        
    Returns:
        dict: Dictionary of early stopping configuration options
    """
    debug_print(f"get_es_variant ENTRY: mask_algo='{mask_algo}'")
    
    td_variant = {
        'pat'       : '5',
        'type'      : 'hard',
        'dataset'   : 'ood',
        'lvl'       : 'Block',
    }
    
    debug_print(f"  Default config: {td_variant}")

    hparam_count = 0
    for hparam in mask_algo.split('_'):
        if ':' in hparam:
            key, value = hparam.split(':')
            if key not in td_variant:
                error_msg = f'Invalid mask_hparam: {key}'
                debug_print(f"get_es_variant ERROR: {error_msg}", force=True)
                raise ValueError(error_msg)
            td_variant[key] = value.lower()
            hparam_count += 1
            debug_print(f"  Parsed {key}={value.lower()}")
    
    debug_print(f"get_es_variant RETURN: parsed {hparam_count} hyperparameters, final config = {td_variant}")
    return td_variant


def get_valid_mask(mask_algo, model, base_optimizer, iter_idx, trainable_layer_names, **kwargs):
    """Compute masks based on validation loss for each layer.
    
    Args:
        mask_algo: Configuration string
        model: Neural network model
        base_optimizer: Optimizer used for adaptation
        iter_idx: Current iteration index
        trainable_layer_names: List of trainable layer names
        **kwargs: Additional arguments (including validation loaders)
        
    Returns:
        tuple: (mask values, info dictionary)
    """
    global stop_flag_dict, min_loss_dict, steps_no_improve_dict

    debug_print("=" * 60)
    debug_print("GET_VALID_MASK ENTRY")
    debug_print(f"  mask_algo: '{mask_algo}', iter_idx: {iter_idx}")

    es_variant = get_es_variant(mask_algo)
    es_variant['iter_idx'] = iter_idx
    
    granularity = es_variant['lvl']
    box_names = get_all_param_name_list(model, granularity)
    debug_print(f"  Found {len(box_names)} boxes at '{granularity}' level")

    if iter_idx == 0:
        stop_flag_dict = {box_name: False for box_name in box_names}
        debug_print("  Initialized stop_flag_dict (iter_idx=0)")

    if es_variant['dataset'] == 'iid':
        if 'iidte_loader' not in kwargs:
            error_msg = "iidte_loader not found in kwargs"
            debug_print(f"get_valid_mask ERROR: {error_msg}", force=True)
            raise ValueError(error_msg)
        loader = kwargs['iidte_loader']
        debug_print("  Using IID validation loader")
    elif es_variant['dataset'] == 'ood':
        if 'oodte_loader' not in kwargs:
            error_msg = "oodte_loader not found in kwargs"
            debug_print(f"get_valid_mask ERROR: {error_msg}", force=True)
            raise ValueError(error_msg)
        loader = kwargs['oodte_loader']
        debug_print("  Using OOD validation loader")

    info = {
        'valid_loss_dict': {}
    }

    debug_print("  Saving current model state...")
    all_curr_params_dict, all_curr_grads_dict = {}, {}
    for name, params in model.named_parameters():
        all_curr_params_dict[name] = params.data.clone()
        if params.grad is None:
            all_curr_grads_dict[name] = None
        else:
            all_curr_grads_dict[name] = params.grad.clone()
    
    mask_dict = {}
    
    for box_idx, box_name in enumerate(box_names):
        debug_print(f"  Processing box {box_idx+1}/{len(box_names)}: '{box_name}'")
        
        if es_variant['type'] == 'hard' and stop_flag_dict[box_name] == True:
            mask_dict[box_name] = 0
            debug_print(f"    Box already stopped, mask=0")
            continue

        temp_params_list = []
        
        for idx, (name, params) in enumerate(model.named_parameters()):
            trainable_layer = is_layer_trainable(name, trainable_layer_names)
            if params.requires_grad == True and trainable_layer:
                if box_name in name:
                    temp_params_list.append(params)

        debug_print(f"    Found {len(temp_params_list)} parameters for this box")

        base_opt_params = copy.deepcopy(base_optimizer.defaults)

        temp_optimizer = type(base_optimizer)(temp_params_list, **base_opt_params)
        
        # FIX #1: Guard against missing gradients (same as get_LEAST_masks)
        has_grad = any(p.grad is not None for p in temp_params_list)
        if not has_grad:
            debug_print("VALID ERROR: No gradients on parameters. Call backward() before layer selection.", force=True)
            raise RuntimeError(
                "Validation-based layer selection requires gradients. "
                "Ensure loss.backward() is called before get_new_optimizer()."
            )
        
        temp_optimizer.step()

        sup_loss = 0
        device = next(model.parameters()).device
        
        count = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                p = model(x)
                batch_loss = F.cross_entropy(p, y).item()
                sup_loss += batch_loss
                count += 1
        
        if count > 0:
            sup_loss /= count
        
        debug_print(f"    Validation loss: {sup_loss:.6f} (from {count} batches)")

        stop_flag = early_stopping_with_loss(
            sup_loss, int(es_variant['pat']), iter_idx, box_name)
        
        stop_flag_dict[box_name] = stop_flag
        info['valid_loss_dict'][box_name] = sup_loss

        mask_dict[box_name] = 1 if stop_flag == False else 0
        debug_print(f"    Stop flag: {stop_flag}, Mask: {mask_dict[box_name]}")

        for name, params in model.named_parameters():
            params.data = all_curr_params_dict[name]
            params.grad = all_curr_grads_dict[name]

    mask_values = get_mask_values(mask_dict, model, trainable_layer_names)
    info['mask_dict'] = mask_dict

    debug_print(f"GET_VALID_MASK RETURN: {len(mask_values)} mask values")
    debug_print("=" * 60)
    return mask_values, info


def early_stopping_with_loss(loss, patience, iter_idx, box_name):
    """Check for early stopping based on validation loss.
    
    Args:
        loss: Current validation loss
        patience: Number of iterations to wait for improvement
        iter_idx: Current iteration index
        box_name: Name of the current box
        
    Returns:
        bool: True if should stop, False otherwise
    """
    global min_loss_dict, steps_no_improve_dict

    if iter_idx == 0:
        min_loss_dict[box_name] = float('inf')
        steps_no_improve_dict[box_name] = 0
        debug_print(f"early_stopping_with_loss: Initialized for '{box_name}'")

    if loss <= min_loss_dict[box_name]:
        min_loss_dict[box_name] = loss
        steps_no_improve_dict[box_name] = 0
        debug_print(f"early_stopping_with_loss: '{box_name}' improved to {loss:.6f}")
    else:
        steps_no_improve_dict[box_name] += 1
        debug_print(f"early_stopping_with_loss: '{box_name}' no improvement ({steps_no_improve_dict[box_name]}/{patience})")
        if steps_no_improve_dict[box_name] > patience:
            debug_print(f"early_stopping_with_loss: '{box_name}' STOPPING (patience exceeded)")
            return True
        
    return False


def get_new_optimizer(mask_algo, model, optimizer, iter_idx, trainable_layer_names=None, track_flag=False, **kwargs):
    """Main entry point for layer selection methods.
    
    Args:
        mask_algo: String specifying layer selection method
        model: Neural network model
        optimizer: Optimizer used for adaptation
        iter_idx: Current iteration index
        trainable_layer_names: List of layers that can be adapted
        track_flag: If True, return info dictionary for tracking
        **kwargs: Additional arguments for specific methods
        
    Returns:
        optimizer: Modified optimizer with layer-specific learning rates
        info: (optional) Dictionary with diagnostic information
    """
    debug_print("=" * 80)
    debug_print("GET_NEW_OPTIMIZER ENTRY")
    debug_print(f"  mask_algo: '{mask_algo}'")
    debug_print(f"  iter_idx: {iter_idx}")
    debug_print(f"  track_flag: {track_flag}")
    debug_print(f"  trainable_layer_names: {trainable_layer_names}")
    debug_print(f"  kwargs keys: {list(kwargs.keys())}")
    
    try:
        info = {}
        new_opt_flag = False

        if trainable_layer_names is not None:
            original_count = len(trainable_layer_names)
            trainable_layer_names = [get_clean_layer_name(name) for name in trainable_layer_names]
            debug_print(f"  Cleaned {original_count} trainable layer names")
        
        # Determine which method to use based on mask_algo
        mask_algo_lower = mask_algo.lower()
        debug_print(f"  Determining method from: '{mask_algo_lower}'")
        
        if mask_algo_lower.startswith('lls'):
            # LLS method
            debug_print("  Detected LLS method")
            if 'x' not in kwargs:
                error_msg = "Input tensor 'x' is required for LLS but not provided"
                print(error_msg)
                raise ValueError(error_msg)
            mask_values, info = get_lls_masks(mask_algo, model, optimizer, iter_idx, 
                                              trainable_layer_names, **kwargs)
            new_opt_flag = True
        elif mask_algo_lower.startswith('least'):
            # LEAST method
            debug_print("  Detected LEAST method")
            mask_values, info = get_LEAST_masks(mask_algo, model, optimizer, iter_idx, 
                                                trainable_layer_names, track_flag=True, **kwargs)
            new_opt_flag = True
        elif 'layer' in mask_algo_lower or 'stage' in mask_algo_lower:
            # Explicit layer selection
            debug_print("  Detected explicit layer selection")
            mask_values = get_box_masks(mask_algo, model, trainable_layer_names)
            new_opt_flag = True
        elif 'valid' in mask_algo_lower:
            # Validation-based selection
            debug_print("  Detected validation-based selection")
            mask_values, info = get_valid_mask(mask_algo, model, optimizer, iter_idx, 
                                               trainable_layer_names, **kwargs)
            new_opt_flag = True
        elif 'autorgn' in mask_algo_lower:
            # Automatic RGN-based selection
            debug_print("  Detected AutoRGN method")
            mask_values = get_auto_rgn_values(optimizer.param_groups)
            new_opt_flag = True
        elif 'autosnr' in mask_algo_lower:
            # Automatic SNR-based selection
            debug_print("  Detected AutoSNR method")
            mask_values = get_auto_snr_values(optimizer.param_groups)
            new_opt_flag = True
        else:
            # No layer selection method identified
            debug_print("  No layer selection method detected, returning original optimizer")
            if track_flag:
                return optimizer, info
            else:
                return optimizer

        # Apply layer-specific learning rates
        if new_opt_flag:
            debug_print(f"  Applying {len(mask_values)} layer-specific learning rates")
            
            if len(optimizer.param_groups) == 1:
                debug_print("  Optimizer has single param group, creating diff_lr_optimizer")
                optimizer = diff_lr_optimizer(optimizer)
            else:
                debug_print(f"  Optimizer has {len(optimizer.param_groups)} param groups, using as-is")
            
            diff_lr(optimizer, mask_values)
            debug_print("  Successfully applied differential learning rates")

        if track_flag:
            debug_print(f"GET_NEW_OPTIMIZER RETURN: optimizer + info (keys: {list(info.keys())})")
            debug_print("=" * 80)
            return optimizer, info
        else:
            debug_print("GET_NEW_OPTIMIZER RETURN: optimizer only")
            debug_print("=" * 80)
            return optimizer
            
    except Exception as e:
        print(f"[ERROR] CRITICAL ERROR in get_new_optimizer: {str(e)}")
        print(traceback.format_exc())
        raise
