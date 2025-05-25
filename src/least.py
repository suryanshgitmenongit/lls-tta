# least.py - Layer selection algorithm for test-time adaptation 
# Super verbose version, with debug prints for both init:LEAST and init:RGN 
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
    seen = set()  # Set to track seen items
    seen_add = seen.add  # Store method reference for speed
    # Return only items that haven't been seen before
    result = [x for x in seq if not (x in seen or seen_add(x))]
    return result


def get_all_param_name_list(algorithm, granularity):
    """Get list of parameter group names based on specified granularity.
    
    Args:
        algorithm: Model or algorithm instance
        granularity: Level of detail ('block', 'subblock', 'layer', 'model')
        
    Returns:
        list: List of parameter group names
    """
    granularity = granularity.lower()  # Convert to lowercase for consistency
    
    # Get the actual model from the algorithm
    model = algorithm
    if hasattr(algorithm, 'featurizer'):
        model = algorithm.featurizer.network
    elif hasattr(algorithm, 'network'):
        model = algorithm.network
    
    # Handle ResNet models
    if isinstance(model, torchvision.models.ResNet):
        pre_box_list = []  # Layers before main blocks
        post_box_list = []  # Layers after main blocks
        layer_name = 'layer'  # Base name for layers
        
        # Set granularity level based on input
        if granularity == 'block':
            granularity_level = 1
        elif granularity == 'subblock':
            granularity_level = 2
        elif granularity == 'layer':
            granularity_level = -1
        elif granularity == 'model':
            return [layer_name]  # Return single name for whole model
        else:
            error_msg = f'Not implemented for granularity: {granularity}'
            raise NotImplementedError(error_msg)

    # Handle standard or WRN models (based on parameter count)
    elif len(list(model.named_parameters())) == 80:  # Standard.pt or WRN model
        pre_box_list = []
        post_box_list = []
        layer_name = 'layer'
        
        if granularity == 'block':
            granularity_level = 1
        elif granularity == 'subblock':
            granularity_level = 3
        elif granularity == 'layer':
            granularity_level = 4
        elif granularity == 'model':
            return [layer_name]
        else:
            error_msg = f'Not implemented for granularity: {granularity}'
            raise NotImplementedError(error_msg)
    
    # Handle ResNeXt models
    elif len(list(model.named_parameters())) == 95:  # Hendrycks2020AugMix_ResNeXt.pt
        pre_box_list = []
        post_box_list = []
        layer_name = 'stage'
        
        if granularity == 'block':
            granularity_level = 1
        elif granularity == 'subblock':
            granularity_level = 2
        elif granularity == 'layer':
            granularity_level = 3
        elif granularity == 'model':
            return [layer_name]
        else:
            error_msg = f'Not implemented for granularity: {granularity}'
            raise NotImplementedError(error_msg)
        
    else:
        error_msg = f'Unrecognized model type with {len(list(model.named_parameters()))} parameters'
        raise NotImplementedError(error_msg)
    
    # Create list of box names based on parameter names
    all_params = list(model.named_parameters())
    
    all_box_list = []
    for name, _ in all_params:
        if layer_name in name:  # Only parameters containing layer_name
            if granularity_level > 0:
                parts = name.split(".")
                if len(parts) >= granularity_level:
                    all_box_list.append(".".join(parts[:granularity_level]))
            else:
                all_box_list.append(name)
    
    all_box_list = remove_duplicates(all_box_list)
    
    # Return combined list
    result = pre_box_list + all_box_list + post_box_list
    return result


def get_clean_layer_name(layer_name):
    """Remove common prefixes from layer names for consistent matching.
    
    Args:
        layer_name: Original layer name
        
    Returns:
        str: Cleaned layer name
    """
    remove_words = ['featurizer', 'network']  # Words to remove
    # Join parts of the name that aren't in remove_words
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
    if trainable_layer_names is None:  # If no restrictions
        return True
    
    clean_name = get_clean_layer_name(name)
    is_trainable = clean_name in trainable_layer_names
    return is_trainable


# Global variables for LEAST
source_params_dict = {}  # Store source parameters
window_start_params_dict = {}  # Store window start parameters
window_index = None  # Current window index

# Global storage for LLS
_lls_storage = {}  # Store model-specific state for LLS

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
    
    # Create td hparams dictionary with default values
    td_variant = {
        # Metric design choices
        'nbox'   : '1',         # number of boxes: 1, n
        'curr'   : 'next',      # td = <> - anchor: next, prev
        'bin'    : '1',         # binarize the mask: 0, 1
        'trust'  : 'low',       # trust on init samples: high, low
        'lvl'    : 'Layer',     # granularity of the model: Layer, Block, Subblock
        'type'   : 'best',      # type of mask: best, notbest, worst, random
        'order'  : '0',         # seed for deciding sample order: 0, 1, 2

        # Metric's hyperparameters
        'sz'     : '20',        # window size: 5, 20, n
        'mthr'   : '0.75',      # threshold based on metric range: 0.5, 0.75, 0.99
        'sthr'   : '0.75',      # if trust == low, no. of sample / window sz threshold: 0.5, 0.75, 0.9
    }

    # Update the td hparams dictionary with the given hyperparameters
    for hparam in mask_algo.split('_'):  # Split by underscore
        if ':' in hparam:  # Check if it's a key-value pair
            key, value = hparam.split(':')  # Split into key and value
            if key not in td_variant:  # Check if key is valid
                error_msg = f'Invalid mask_hparam: {key}'
                raise ValueError(error_msg)
            td_variant[key] = value.lower()  # Update configuration value
    
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

    iter_idx = td_variant['iter_idx']  # Get current iteration
    
    anchor_params_dict = {}  # Dictionary to store anchor parameters
    
    if iter_idx == 0:  # First iteration
        # Start of TTA for any target domain
        # Store the current parameters as source/window start/anchor parameters
        source_params_dict = copy.deepcopy(before_params_dict)
        anchor_params_dict = copy.deepcopy(before_params_dict)
        window_start_params_dict = copy.deepcopy(before_params_dict)
        window_index = 0  # Reset window index

    else:  # Not first iteration
        if td_variant['sz'] == 'n':  # Use whole trajectory
            # Clone the source parameters to anchor parameters
            if not source_params_dict:  # Check source params exist
                error_msg = 'TD Error: Source parameters are empty'
                raise ValueError(error_msg)
            anchor_params_dict = copy.deepcopy(source_params_dict)
            window_index += 1  # Increment window index

        elif td_variant['sz'].isdigit() and int(td_variant['sz']) > 0:  # Use fixed window size
            sz = int(td_variant['sz'])
            # Reset the window start parameters after every sz iterations
            if iter_idx % sz == 0:  # Start of new window
                window_start_params_dict = copy.deepcopy(before_params_dict)
                window_index = 0  # Reset window index
            else:
                window_index += 1  # Increment window index
            
            # Clone the window start parameters to anchor parameters
            if not window_start_params_dict:  # Check window start params exist
                error_msg = 'TD Error: Window start parameters are empty'
                raise ValueError(error_msg)
            anchor_params_dict = copy.deepcopy(window_start_params_dict)

        else:  # Invalid window size
            error_msg = f'TD Error: Invalid window size: {td_variant["sz"]}'
            raise ValueError(error_msg)

    return anchor_params_dict  # Return anchor parameters


def get_curr_params_dict(before_params_dict, after_params_dict, td_variant):
    """Get current parameters for computing deltas based on strategy.
    
    Args:
        before_params_dict: Parameters before update
        after_params_dict: Parameters after update
        td_variant: Configuration dictionary
        
    Returns:
        dict: Current parameters for delta computation
    """
    if td_variant['curr'] == 'next':  # Use parameters after update
        return copy.deepcopy(after_params_dict)
    elif td_variant['curr'] == 'prev':  # Use parameters before update
        return copy.deepcopy(before_params_dict)        
    else:  # Invalid strategy
        error_msg = f'TD Error: Get current parameters not implemented for {td_variant["curr"]}'
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
    # Check shape compatibility
    if curr_params.shape != anchor_params.shape:
        error_msg = f'TD Error: Shape mismatch for curr ({curr_params.shape}) and anchor ({anchor_params.shape}) parameters'
        raise ValueError(error_msg)
    
    # Compute parameter delta
    total_delta = curr_params - anchor_params
    
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
    total_delta_tensor_dict = {}  # Dictionary to store delta tensors
    
    # Process each box
    for box_name in anchor_params_dict.keys():
        # Get the flattened current and anchor parameters
        curr_params = torch.cat([params.flatten() for params in curr_params_dict[box_name]])
        anchor_params = torch.cat([params.flatten() for params in anchor_params_dict[box_name]])

        # Compute the total delta for the current box
        total_delta_tensor_dict[box_name] = compute_total_delta_tensor_of_box(
            curr_params, anchor_params, td_variant)

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
    # Get the anchor and current parameters
    anchor_params_dict = get_anchor_params_dict(before_params_dict, td_variant)
    curr_params_dict = get_curr_params_dict(before_params_dict, after_params_dict, td_variant)

    # Compute the total delta
    total_delta_tensor_dict = get_total_delta_tensor(
        anchor_params_dict, curr_params_dict, td_variant)
    
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
    # Get the anchor and current parameters
    anchor_params_dict = copy.deepcopy(before_params_dict)
    curr_params_dict = copy.deepcopy(after_params_dict)

    # Compute the total delta
    total_delta_tensor_dict = get_total_delta_tensor(
        anchor_params_dict, curr_params_dict, td_variant)
    
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
    metric_values = {}  # Dictionary to store metrics
    
    # Process each box
    for box_name in td_traj_tensor_dict.keys():
        td_traj_tensor = td_traj_tensor_dict[box_name]  # Trajectory delta
        td_samp_tensor = td_samp_tensor_dict[box_name]  # Sample delta

        # Compute cosine similarity (alignment) between trajectory and sample deltas
        metric_value = torch.cosine_similarity(td_traj_tensor, td_samp_tensor, dim=0)
        metric_values[box_name] = metric_value  # Store metric

    return metric_values


def get_mistrust_window_sz(td_variant):
    """Get window size for mistrust calculation.
    
    Args:
        td_variant: Configuration dictionary
        
    Returns:
        int: Window size for mistrust calculation
    """
    if td_variant['sz'] == 'n':  # If window size is 'n' (all samples)
        # Hardcoding the window size for trust window if not provided
        # Cannot set extremely large as it will considerably slow down adaptation
        sz = 50  # Default size
    else:  # If explicit window size provided
        sz = int(td_variant['sz'])  # Convert to integer
    
    sthr = float(td_variant['sthr'])  # Get threshold
    mistrust_window_sz = int(sz * sthr)  # Calculate result
    return mistrust_window_sz


def compute_mask(metric_values_dict, td_variant):
    """Compute masks from metric values based on configuration.
    
    Args:
        metric_values_dict: Dictionary of metrics for each box
        td_variant: Configuration dictionary
        
    Returns:
        dict: Dictionary of mask values for each box
    """
    global window_index  # Use global window index
    
    # Special mask types
    type = td_variant['type']  # Get mask type
    if type == 'worst':  # Use worst boxes
        return compute_worst_mask(metric_values_dict, td_variant)
    if type == 'notbest':  # Use all but best box
        return compute_not_best_mask(metric_values_dict, td_variant)
    elif type == 'random':  # Use random box
        return compute_random_mask(metric_values_dict, td_variant)

    # Standard mask computation
    # Clone the metric_values_dict to mask_dict
    mask_dict = copy.deepcopy(metric_values_dict)

    # Mask application: single or multiple boxes
    nbox = td_variant['nbox']  # Get number of boxes to use
    if nbox == '1':  # Single box mode
        # Single box: Store the mask for box with max metric value
        max_mask_value = max(mask_dict.values())  # Get maximum metric
        for box_name, mask_value in mask_dict.items():
            if mask_value == max_mask_value:  # If this is the best box
                mask_dict[box_name] = mask_value  # Keep value
            else:  # Not the best box
                mask_dict[box_name] = torch.tensor(0.0, device=mask_value.device)  # Zero out mask
    elif nbox == 'n':  # Multiple box mode
        # Multiple boxes: Do nothing (keep all metrics)
        pass
    else:  # Invalid mode
        error_msg = f'TD Error: Invalid number of boxes: {nbox}'
        raise ValueError(error_msg)

    # Mask application per-box: binary (based on mthr), weighted (based on metric values)
    bin_mask = td_variant['bin']  # Get binarization flag
    mthr = float(td_variant['mthr'])  # Get threshold
    if bin_mask == '1':  # Binarize masks
        # Binarize the mask based on mthr
        for box_name, mask_value in mask_dict.items():
            if mask_value > mthr:  # If above threshold
                mask_dict[box_name] = torch.tensor(1.0, device=mask_value.device)  # Set to 1
            else:  # Below threshold
                mask_dict[box_name] = torch.tensor(0.0, device=mask_value.device)  # Set to 0
    elif bin_mask == '0':  # Keep continuous values
        # Layer-wise mask based on metric values: do nothing
        pass
    else:  # Invalid option
        error_msg = f'TD Error: Invalid binarize mask: {bin_mask}'
        raise ValueError(error_msg)

    # Mask application based on trust on init samples: high, low, no (based on sthr)
    trust = td_variant['trust']  # Get trust level
    if trust == 'low':  # Low trust mode
        # Low trust: Scale the mask based on the number of samples / mistrust window size
        mistrust_window_sz = get_mistrust_window_sz(td_variant)
        if window_index+1 < mistrust_window_sz:  # If still in mistrust window
            # Calculate trust factor based on window position
            mistrust_factor = (window_index+1) / mistrust_window_sz
            for box_name, mask_value in mask_dict.items():
                # Scale mask by mistrust factor
                mask_dict[box_name] *= mistrust_factor
    elif trust == 'high':  # High trust mode
        # High trust: Trust the metric/mask on init samples (do nothing)
        pass
    else:  # Invalid trust mode
        error_msg = f'TD Error: Invalid trust on init samples: {trust}'
        raise ValueError(error_msg)

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
    lr_values = []  # List to store mask values
    
    # Process each parameter
    for idx, (name, param) in enumerate(model.named_parameters()):
        # Check if parameter is trainable
        trainable_layer = is_layer_trainable(name, trainable_layer_names)
        if param.requires_grad and trainable_layer:
            lr_value = 0  # Default value (no mask)
            # Find which box this parameter belongs to
            for dname in trainable_lr_dict.keys():
                if dname in name:  # If parameter belongs to this box
                    lr_value = trainable_lr_dict[dname]  # Use box's mask value
                    break
            lr_values.append(float(lr_value))  # Add to list

    # Check if any parameter is masked
    if sum(lr_values) <= 0:
        debug_print("WARNING: No trainable parameters found (all masks are zero)", force=True)
    
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
    
    try:
        ### Step 1: Prepare everything
        # Get the td variant
        td_variant = get_td_variant(mask_algo)  # Parse configuration
        td_variant['iter_idx'] = iter_idx  # Set iteration index
        
        # Get all the block names
        granularity = td_variant['lvl']  # Get granularity
        box_names = get_all_param_name_list(model, granularity)  # Get box names

        # Copy all the current model parameters and gradients
        all_curr_params_dict, all_curr_grads_dict = {}, {}  # Store current state
        for name, params in model.named_parameters():
            all_curr_params_dict[name] = params.data.clone()  # Copy parameter
            if params.grad is None:  # If no gradient
                all_curr_grads_dict[name] = None  # Store None
            else:  # If gradient exists
                all_curr_grads_dict[name] = params.grad.clone()  # Copy gradient
        
        ### Step 2: Simulate the optimizer step for cloned model and optimizer
        # Get the base optimizer's parameters
        base_opt_params = copy.deepcopy(base_optimizer.defaults)  # Copy optimizer settings
        
        # Get the parameter list for optimizer step (without cloning)
        # Also, get the before-TTA parameter dict (by cloning)
        temp_params_list = []  # List to store parameters for temp optimizer
        before_params_dict = {}  # Dictionary to store parameters before update
        
        # Initialize parameter dictionary for each box
        for box_name in box_names:
            before_params_dict[box_name] = []  # Empty list for each box
            
            # Loop through model parameters
            for idx, (name, params) in enumerate(model.named_parameters()):
                # Check if parameter is trainable
                trainable_layer = is_layer_trainable(name, trainable_layer_names)
                if params.requires_grad and trainable_layer:
                    if box_name in name:  # If parameter belongs to this box
                        temp_params_list.append(params)  # Add to temp parameters
                        before_params_dict[box_name].append(params.data.clone())  # Store copy

        # Temporary optimizer step for the model
        temp_optimizer = type(base_optimizer)(temp_params_list, **base_opt_params)
        temp_optimizer.step()  # Perform update step

        # Get the after-TTA parameter dict (by cloning)
        after_params_dict = {}  # Dictionary to store parameters after update
        
        # Initialize parameter dictionary for each box
        for box_name in box_names:
            after_params_dict[box_name] = []  # Empty list for each box
            
            # Loop through model parameters
            for idx, (name, params) in enumerate(model.named_parameters()):
                # Check if parameter is trainable
                trainable_layer = is_layer_trainable(name, trainable_layer_names)
                if params.requires_grad and trainable_layer:
                    if box_name in name:  # If parameter belongs to this box
                        after_params_dict[box_name].append(params.data.clone())  # Store copy

        ### Step 3: Mask computation for each box
        # Get trajectory TD
        td_traj_tensor_dict = get_traj_td_tensor_dict(before_params_dict, after_params_dict, td_variant)

        # Get sample/batch TD
        td_samp_tensor_dict = get_samp_td_tensor_dict(before_params_dict, after_params_dict, td_variant)
        
        # Compute mask for each box
        metric_values_dict = compute_metric_values(td_traj_tensor_dict, td_samp_tensor_dict, td_variant)

        ### Step 4: Mask application for each box
        mask_dict = compute_mask(metric_values_dict, td_variant)

        ### Step 5: Pack up everything    
        # Restore all the current model's parameters and grads
        for name, params in model.named_parameters():
            params.data = all_curr_params_dict[name]  # Restore parameter
            params.grad = all_curr_grads_dict[name]  # Restore gradient

        # Get mask values for each layer
        mask_values = get_mask_values(mask_dict, model, trainable_layer_names)

        # Log intermediate results
        info = {
            'metric_values_dict': metric_values_dict,
            'mask_dict': mask_dict
        }

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
    print(f"[LLS_FLOW_1] get_lls_variant called with mask_algo: {mask_algo}")
    
    # Default to gradient masking if not specified
    mtype = False  # Default update method
    if 'update' in mask_algo.lower():  # Check if update method is specified
        mtype = True  # Use model interpolation instead of gradient masking
    
    # Create LLS hyperparameters dictionary with default values
    lls_variant = {
        'init'    : 'rgn',    # mask initialization: curr, prev, rgn, rgnbin, least
        'finit'   : '0.5',    # first mask init value: 0, 0.5, 1, rand
        'emalr'   : '0.5',    # ema learning rate: 0.3, 0.5, 0.7
        'lr'      : '0.01',   # mask learning rate: 0.1, 0.01, 0.001
        'steps'   : '1',     # mask training steps: 10, 50, 100, 200
        'patience': '3',      # early stopping patience: 1, 3, 5, 10
        'loss'    : 'curr',   # mask loss: curr, prev
        'reg'     : 'l1',     # regularization: none, l1, l2
        'regstr'  : '0.01',   # regularization strength: 1, 0.1, 0.01
        'bnd'     : 'clip',   # mask bound: none, clip, scale
        'bin'     : '0',      # binarize masks: 0, 1
        'appl'    : 'avg',    # mask application: curr, prod, avg, ema
        'update'  : mtype,    # mask type: update or grad
        'lvl'     : 'Layer',  # granularity: Layer, Block, Subblock
        'nbox'    : '1',      # number of boxes: 1, n (from LEAST)
        'mthr'    : '0.75',   # threshold based on metric range: 0.5, 0.75, 0.99 (from LEAST)
    }
    
    # Parse configuration string
    for hparam in mask_algo.split('_'):  # Split by underscore
        if ':' in hparam:  # Check if it's a key-value pair
            key, value = hparam.split(':')  # Split into key and value
            if key not in lls_variant:  # Check if key is valid
                debug_print(f"WARNING: Invalid LLS hyperparameter: {key}, ignoring")
            else:
                lls_variant[key] = value  # Update configuration value
    
    print(f"[LLS_FLOW_2] LLS variant parsed - init: {lls_variant['init']}, steps: {lls_variant['steps']}")
    return lls_variant


def tta_loss(outputs):
    """Compute test-time adaptation loss (entropy minimization).
    
    Args:
        outputs: Model output logits
        
    Returns:
        torch.Tensor: Mean entropy loss
    """
    # Compute entropy of softmax distribution
    loss = softmax_entropy(outputs).mean(0)
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
    if string is None:  # Handle None case
        return False
    try:
        # Try to iterate through characters
        for char in string:
            # Check if each character is a digit or decimal point
            if not (char.isdigit() or char == '.'):
                return False
        return True  # All characters were valid
    except:  # Any exception means it's not numeric
        return False


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
    global _lls_storage  # Use global storage
    
    print(f"\n[LLS_FLOW_3] ========== LLS ENTRY POINT ==========")
    print(f"[LLS_FLOW_4] get_lls_masks called - iter_idx: {iter_idx}, mask_algo: {mask_algo}")
    print(f"[LLS_FLOW_5] Input tensor provided: {x is not None}, kwargs keys: {list(kwargs.keys())}")
    
    try:
        # ====================== INITIALIZATION AND VALIDATION ======================
        
        # Check for required input tensor
        if x is None:  # Input tensor is required for LLS
            error_msg = "[LLS] ERROR: Input tensor required for LLS but not provided"
            print(error_msg)
            raise ValueError(error_msg)
        
        # Clean trainable layer names if provided
        if trainable_layer_names is not None:  # If trainable layers are specified
            # Clean layer names by removing common prefixes
            trainable_layer_names = [get_clean_layer_name(name) for name in trainable_layer_names]
        
        # Get device for tensor operations
        device = next(model.parameters()).device  # Get device from model parameters
        
        # ====================== MODEL STORAGE MANAGEMENT ======================
        
        # Get model-specific storage
        model_id = id(model)  # Get unique identifier
        if model_id not in _lls_storage:  # If new model
            _lls_storage[model_id] = {}  # Create storage
        storage = _lls_storage[model_id]  # Get storage
        
        # Reset storage if new adaptation round
        if iter_idx == 0 and 'prev_mask' in storage:
            storage.clear()  # Clear storage
        
        # ====================== CONFIGURATION PARSING ======================
        
        # Parse LLS hyperparameters
        lls_variant = get_lls_variant(mask_algo)  # Get configuration
        
        # Debug information for initialization parameter
        init_value = str(lls_variant['init']).strip().lower()
        print(f"[LLS_FLOW_6] CRITICAL: Initialization method detected = '{init_value}'")
        print(f"[LLS_FLOW_7] Check if init == 'least': {init_value == 'least'}")
        print(f"[LLS_FLOW_7b] Check if init in ['rgn', 'rgnbin']: {init_value in ['rgn', 'rgnbin']}")
        print(f"[LLS_FLOW_7c] Check if init == 'prev': {init_value == 'prev'}")

        # Get granularity and box names
        granularity = lls_variant['lvl']  # Get granularity level
        box_names = get_all_param_name_list(model, granularity)
        print(f"[LLS_FLOW_8] Using {len(box_names)} boxes at {granularity} granularity")
        
        # ====================== PARAMETER MAPPING ======================
        
        # Map each parameter to its corresponding box index
        param_to_box_idx = {}  # Dictionary: parameter name -> box index
        # Loop through all parameters in the model
        for idx, (name, param) in enumerate(model.named_parameters()):
            # Check if parameter is trainable and in allowed layers
            if param.requires_grad and (trainable_layer_names is None or 
                                       get_clean_layer_name(name) in trainable_layer_names):
                # Find which box this parameter belongs to
                for box_idx, box_name in enumerate(box_names):
                    if box_name in name:  # If parameter belongs to this box
                        param_to_box_idx[name] = box_idx  # Store box index
                        break
                else:  # Parameter doesn't belong to any box
                    param_to_box_idx[name] = -1  # Use -1 to indicate not in any box
        
        # ====================== MASK INITIALIZATION ======================
        
        # Create initial masks based on specified initialization method
        # Normalize initialization value for consistent comparison
        init_value = str(lls_variant['init']).strip().lower()
        
        print(f"[LLS_FLOW_9] EXECUTING INITIALIZATION BRANCH: '{init_value}'")
        
        if init_value == 'least':  # LEAST initialization
            print(f"[LLS_FLOW_10] *** LEAST INITIALIZATION BRANCH ENTERED ***")
            
            # Call LEAST directly to get its metrics
            print(f"[LLS_FLOW_11] About to call get_LEAST_masks with mask_algo='LEAST'")
            try:
                _, least_info = get_LEAST_masks("LEAST", model, base_optimizer, iter_idx, 
                                              trainable_layer_names, track_flag=True, x=x, **kwargs)
                print(f"[LLS_FLOW_12] *** LEAST CALL SUCCESSFUL *** - returned keys: {list(least_info.keys())}")
                
                # Validate LEAST return values
                if 'metric_values_dict' not in least_info:
                    error_msg = "LEAST didn't return 'metric_values_dict'. Keys: " + str(least_info.keys())
                    print(error_msg)
                    raise KeyError(error_msg)
                
                # Extract the raw metrics (not the processed masks)
                box_metrics = least_info['metric_values_dict']
                print(f"[LLS_FLOW_13] *** LEAST METRICS EXTRACTED ***")
                print(f"[LLS_FLOW_14] LEAST metrics summary: {[(k, f'{v.item():.4f}') for k, v in box_metrics.items()]}")
                
                # Create mask parameters from LEAST's metrics
                init_mask = nn.ParameterList()
                for box_name in box_names:
                    if box_name in box_metrics:
                        value = box_metrics[box_name].item()  # Get metric value
                        # Clip to valid range [0, 1]
                        value = max(0.0, min(1.0, value))
                        # Create parameter on correct device
                        init_mask.append(nn.Parameter(torch.tensor([value], device=device), requires_grad=True))
                        print(f"[LLS_FLOW_15] Initialized mask for {box_name}: {value:.4f} (from LEAST)")
                    else:
                        print(f"[LLS_FLOW_16] WARNING: No LEAST metric for box {box_name}, using default")
                        value = 0.5  # Default value
                        box_metrics[box_name] = torch.tensor(value)  # Store default value
                        # Create parameter with default value
                        init_mask.append(nn.Parameter(torch.tensor([value], device=device), requires_grad=True))
                
                print(f"[LLS_FLOW_17] *** MASK INITIALIZATION FROM LEAST COMPLETE ***")
                print(f"[LLS_FLOW_18] Created {len(init_mask)} mask parameters from LEAST metrics")
                
            except Exception as e:
                error_msg = f"ERROR in LEAST initialization: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                raise ValueError(error_msg)
        
        elif init_value in ['rgn', 'rgnbin']:  # RGN initialization
            print(f"[LLS_FLOW_19] *** RGN INITIALIZATION BRANCH ENTERED ***")
            print(f"[LLS_FLOW_19a] RGN type: {'binary' if init_value == 'rgnbin' else 'continuous'}")
            
            # Ensure we have gradients
            print(f"[LLS_FLOW_19b] Computing gradients for RGN calculation...")
            outputs = model(x)  # Forward pass
            loss = tta_loss(outputs)  # Compute adaptation loss
            print(f"[LLS_FLOW_19c] Initial loss for RGN: {loss.item():.6f}")
            base_optimizer.zero_grad()  # Clear any existing gradients
            loss.backward()  # Backward pass to get gradients
            
            # Get RGN values for each box
            box_rgn = {}  # Dictionary: box index -> list of RGN values
            # Initialize empty lists for all boxes
            for box_idx in range(len(box_names)):
                box_rgn[box_idx] = []
            
            print(f"[LLS_FLOW_19d] Calculating RGN for each parameter...")
            # Calculate RGN for each parameter
            param_rgn_details = []  # For detailed logging
            for name, param in model.named_parameters():
                # Check if parameter belongs to a box
                if name in param_to_box_idx and param_to_box_idx[name] >= 0:
                    box_idx = param_to_box_idx[name]  # Get box index
                    # Calculate RGN if gradient and parameter exist and are non-zero
                    if param.grad is not None and param.data.norm() > 0:
                        # RGN = gradient norm / parameter norm
                        grad_norm = param.grad.norm().item()
                        param_norm = param.data.norm().item()
                        rgn = param.grad.norm() / param.data.norm()
                        box_rgn[box_idx].append(rgn)  # Add to box's RGN list
                        param_rgn_details.append((name, grad_norm, param_norm, rgn.item()))
            
            # Log sample of RGN calculations
            print(f"[LLS_FLOW_19e] Sample RGN calculations (first 5):")
            for i, (name, gn, pn, rgn) in enumerate(param_rgn_details[:5]):
                print(f"  {name}: grad_norm={gn:.6f}, param_norm={pn:.6f}, rgn={rgn:.6f}")
            if len(param_rgn_details) > 5:
                print(f"  ... and {len(param_rgn_details)-5} more parameters")
            
            # Compute average RGN for each box
            init_mask = nn.ParameterList()  # List to store mask parameters
            box_metrics = {}  # Dictionary to store metric values
            
            print(f"[LLS_FLOW_19f] Computing average RGN per box...")
            # Process each box
            for box_idx in range(len(box_names)):
                if box_idx in box_rgn and len(box_rgn[box_idx]) > 0:  # If box has RGN values
                    # Calculate average RGN
                    rgn_list = [r.item() for r in box_rgn[box_idx]]
                    avg_rgn = sum(rgn_list) / len(rgn_list)
                    value = torch.tensor(avg_rgn, device=device)
                    box_metrics[box_names[box_idx]] = avg_rgn  # Store metric
                    print(f"[LLS_FLOW_19g] Box {box_idx} ({box_names[box_idx]}): avg_rgn={avg_rgn:.6f} from {len(rgn_list)} parameters")
                else:  # No RGN values for this box
                    # Use default value
                    value = torch.tensor(0.5, device=device)
                    box_metrics[box_names[box_idx]] = 0.5  # Store default metric
                    print(f"[LLS_FLOW_19h] Box {box_idx} ({box_names[box_idx]}): default=0.5 (no valid parameters)")
                
                # Create parameter on correct device
                init_mask.append(nn.Parameter(value.view(1), requires_grad=True))
            
            # Normalize the masks
            print(f"[LLS_FLOW_19i] Normalizing RGN masks...")
            mask_values = [m.item() for m in init_mask]  # Get all mask values
            print(f"[LLS_FLOW_19j] Pre-normalization mask values: {[f'{v:.6f}' for v in mask_values[:5]]}")
            if any(v > 0 for v in mask_values):  # If any mask is positive
                max_value = max(mask_values)  # Get maximum value
                if max_value > 0:  # If maximum is positive
                    print(f"[LLS_FLOW_19k] Max RGN value: {max_value:.6f}, normalizing...")
                    # Normalize all masks by maximum value
                    for i, m in enumerate(init_mask):
                        old_val = m.item()
                        init_mask[i].data = m.data / max_value
                        print(f"[LLS_FLOW_19l] Normalized mask for {box_names[i]}: {old_val:.6f} -> {init_mask[i].item():.6f}")
                        
            # Binarize if requested
            if init_value == 'rgnbin':  # If binarization requested
                print(f"[LLS_FLOW_19m] Binarizing RGN masks with threshold 0.5...")
                for i, m in enumerate(init_mask):
                    old_val = m.item()
                    # Convert to binary (0 or 1) based on threshold 0.5
                    init_mask[i].data = (m.data > 0.5).float()
                    print(f"[LLS_FLOW_19n] Binarized mask for {box_names[i]}: {old_val:.6f} -> {init_mask[i].item():.1f}")
            
            print(f"[LLS_FLOW_19o] *** RGN INITIALIZATION COMPLETE ***")
            print(f"[LLS_FLOW_19p] RGN metrics summary: {[(k, f'{v:.4f}') for k, v in box_metrics.items()]}")
        
        elif init_value == 'prev' and 'prev_mask' in storage:  # Previous mask initialization
            print(f"[LLS_FLOW_40] *** PREVIOUS MASK INITIALIZATION BRANCH ENTERED ***")
            # Use previous iteration's masks
            init_mask = copy.deepcopy(storage['prev_mask'])  # Copy previous masks
            # Create metrics dictionary
            box_metrics = {box_names[i]: m.item() for i, m in enumerate(init_mask)}
            print(f"[LLS_FLOW_41] Loaded {len(init_mask)} masks from previous iteration")
            print(f"[LLS_FLOW_42] Previous mask values: {[(box_names[i], f'{m.item():.4f}') for i, m in enumerate(init_mask)]}")
            
            # Add new masks for any new boxes
            if len(init_mask) < len(box_names):  # If we have more boxes now
                print(f"[LLS_FLOW_43] Adding masks for {len(box_names) - len(init_mask)} new boxes")
                # Add default masks for new boxes
                for i in range(len(init_mask), len(box_names)):
                    # Create parameter with default value 0.5
                    init_mask.append(nn.Parameter(torch.tensor([0.5], device=device), requires_grad=True))
                    box_metrics[box_names[i]] = 0.5  # Store default metric
                    print(f"[LLS_FLOW_44] Added new mask for box {i} ({box_names[i]}): 0.5 (default)")
            print(f"[LLS_FLOW_45] *** PREVIOUS MASK INITIALIZATION COMPLETE ***")
        
        else:  # Default initialization
            print(f"[LLS_FLOW_50] *** DEFAULT INITIALIZATION BRANCH ENTERED ***")
            print(f"[LLS_FLOW_51] finit value: '{lls_variant['finit']}'")
            # Use default or specified initialization value
            init_mask = nn.ParameterList()  # List to store mask parameters
            box_metrics = {}  # Dictionary to store metric values
            
            # Get initialization value
            if is_numeric(lls_variant['finit']):  # If numeric value specified
                init_value_num = float(lls_variant['finit'])  # Convert to float
                print(f"[LLS_FLOW_52] Using constant initialization value: {init_value_num}")
            elif lls_variant['finit'] == 'rand':  # If random initialization requested
                init_value_num = None  # Flag for random values
                print(f"[LLS_FLOW_53] Using random initialization")
            else:  # Invalid value
                init_value_num = 0.5  # Use default value
                print(f"[LLS_FLOW_54] Invalid finit value, using default: 0.5")
            
            # Create masks for each box
            print(f"[LLS_FLOW_55] Creating initial masks for {len(box_names)} boxes...")
            for i, box_name in enumerate(box_names):
                if init_value_num is None:  # Random initialization
                    # Generate random value between 0 and 1
                    value = torch.rand(1, device=device).item()
                    # Create parameter with random value
                    init_mask.append(nn.Parameter(torch.tensor([value], device=device), requires_grad=True))
                    box_metrics[box_name] = value  # Store metric
                    print(f"[LLS_FLOW_56] Initial mask for {box_name}: {value:.6f} (random)")
                else:  # Constant initialization
                    # Create parameter with constant value
                    init_mask.append(nn.Parameter(torch.tensor([init_value_num], device=device), requires_grad=True))
                    box_metrics[box_name] = init_value_num  # Store metric
                    print(f"[LLS_FLOW_57] Initial mask for {box_name}: {init_value_num:.6f} (constant)")
            
            print(f"[LLS_FLOW_58] *** DEFAULT INITIALIZATION COMPLETE ***")
            print(f"[LLS_FLOW_59] Default init metrics summary: {[(k, f'{v:.4f}') for k, v in box_metrics.items()]}")
        
        # Print initial mask values for verification
        print(f"[LLS_FLOW_20] *** INITIAL MASK VALUES SUMMARY ***")
        for i, mask in enumerate(init_mask):
            if i < len(box_names):
                print(f"[LLS_FLOW_21] {box_names[i]}: {mask.item():.6f}")
        
        # ====================== MODEL PREPARATION ======================
        
        # Create a copy of the current model as the reference "pre-adaptation" model
        pre_model = copy.deepcopy(model)  # Deep copy to preserve all parameters
        
        # For model interpolation strategy - create a finetuned model
        if lls_variant['update'] == True:  # If using model interpolation
            ft_model = copy.deepcopy(model)  # Deep copy for finetuned model
            # Copy optimizer parameters
            base_opt_params = copy.deepcopy(base_optimizer.defaults)
            # Create optimizer for finetuned model
            ft_opt = type(base_optimizer)(ft_model.parameters(), **base_opt_params)
            
            # Forward pass to compute loss
            ft_outputs = ft_model(x)  # Forward pass
            ft_loss = tta_loss(ft_outputs)  # Compute adaptation loss
            
            # Backward pass and update
            ft_opt.zero_grad()  # Clear gradients
            ft_loss.backward()  # Backward pass
            ft_opt.step()  # Update parameters
            
            # Compute new loss after update
            with torch.no_grad():  # No gradients needed
                # Forward pass with updated model
                ft_outputs_after = ft_model(x)
                ft_loss_after = tta_loss(ft_outputs_after).item()
        
        # For gradient masking strategy - collect the gradients
        else:  # If using gradient masking
            # Ensure gradients exist
            outputs = model(x)  # Forward pass
            loss = tta_loss(outputs)  # Compute adaptation loss
            base_optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backward pass
            
            # Collect all gradients
            grads = {}  # Dictionary: parameter name -> gradient
            # Loop through all parameters
            for name, param in model.named_parameters():
                # Store gradient (or zero tensor if no gradient)
                if param.grad is None:
                    grads[name] = torch.zeros_like(param)
                else:
                    grads[name] = param.grad.clone()
        
        # ====================== MASK OPTIMIZATION ======================
        
        # Make a copy for tracking best masks
        print(f"[LLS_FLOW_22] *** STARTING MASK OPTIMIZATION ***")
        best_mask = copy.deepcopy(init_mask)  # Copy initial masks
        best_loss = float('inf')  # Initialize with infinity
        
        # Setup for early stopping
        patience = int(lls_variant['patience'])  # Convert patience to integer
        patience_counter = patience  # Initialize patience counter
        
        # Create optimizer for mask parameters
        mask_lr = float(lls_variant['lr'])  # Convert learning rate to float
        # Create Adam optimizer for mask parameters
        mask_opt = torch.optim.Adam(init_mask, lr=mask_lr)
        
        # Optimization loop
        max_steps = int(lls_variant['steps'])  # Convert to integer
        print(f"[LLS_FLOW_23] Starting optimization with {max_steps} steps (patience={patience})")
        
        for step in range(max_steps):  # Loop through optimization steps
            # Current mask being optimized
            curr_mask = copy.deepcopy(init_mask)  # Copy current masks
            
            # Retain gradients for inspection
            for param in curr_mask:  # Loop through all mask parameters
                param.retain_grad()  # Keep gradients after backward pass
            
            # *** THE CRITICAL FIX: Recreate optimizer for current mask tensor objects ***
            mask_opt = torch.optim.Adam(curr_mask, lr=mask_lr)
            
            # Apply mask to create modified model
            if lls_variant['update'] == True:  # Model interpolation strategy
                # Create a new model by interpolating between pre_model and ft_model
                new_model = copy.deepcopy(pre_model)  # Start with pre-adaptation model
                
                # Loop through all parameters
                for idx, (name, param) in enumerate(new_model.named_parameters()):
                    # Check if parameter belongs to a box
                    if name in param_to_box_idx and param_to_box_idx[name] >= 0:
                        # Get corresponding parameter from finetuned model
                        ft_param = None
                        # Find matching parameter in finetuned model
                        for ft_name, ft_p in ft_model.named_parameters():
                            if ft_name == name:  # If names match
                                ft_param = ft_p  # Store parameter
                                break
                        
                        if ft_param is not None:  # If found in finetuned model
                            # Apply mask as interpolation weight
                            box_idx = param_to_box_idx[name]  # Get box index
                            mask_value = curr_mask[box_idx].item()  # Get mask value
                            # Interpolate: (1-mask)*pre + mask*finetuned
                            param.data = (1 - mask_value) * param.data + mask_value * ft_param.data
            else:  # Gradient masking strategy
                # Create a new model by applying masked gradients
                new_model = copy.deepcopy(pre_model)  # Start with pre-adaptation model
                
                # Loop through all parameters
                for name, param in new_model.named_parameters():
                    # Check if parameter belongs to a box and has a gradient
                    if name in param_to_box_idx and param_to_box_idx[name] >= 0 and name in grads:
                        box_idx = param_to_box_idx[name]  # Get box index
                        mask_value = curr_mask[box_idx].item()  # Get mask value
                        
                        # Apply masked gradient update
                        if mask_value > 0:  # If mask is non-zero
                            # Simplified Adam-like update (just for simulation)
                            # param = param - mask * lr * gradient
                            param.data = param.data - mask_value * base_optimizer.defaults['lr'] * grads[name]
            
            # Calculate loss for the current mask
            if lls_variant['loss'] == 'prev' and 'prev_x' in storage:  # Use previous input
                outputs = new_model(storage['prev_x'])  # Forward pass with previous input
            else:  # Default to current input
                outputs = new_model(x)  # Forward pass with current input
                
            # Calculate loss
            mask_loss = tta_loss(outputs)  # Compute adaptation loss
            
            # Add regularization
            reg_loss = 0  # Initialize regularization loss
            if lls_variant['reg'] == 'l1':  # L1 regularization
                reg_strength = float(lls_variant['regstr'])  # Get regularization strength
                # Sum of absolute values of all mask parameters
                reg_loss = reg_strength * sum(param.abs().sum() for param in curr_mask)
            elif lls_variant['reg'] == 'l2':  # L2 regularization
                reg_strength = float(lls_variant['regstr'])  # Get regularization strength
                # Sum of squares of all mask parameters
                reg_loss = reg_strength * sum((param ** 2).sum() for param in curr_mask)
            
            # Total loss
            total_loss = mask_loss + reg_loss  # Add main loss and regularization
            
            # Skip if loss is NaN or Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):  # Check for invalid loss
                continue
            
            # Update mask
            mask_opt.zero_grad()  # Clear gradients
            total_loss.backward()  # Backward pass
            
            # Check for NaN or Inf gradients
            has_bad_grad = False  # Flag for bad gradients
            for idx, param in enumerate(curr_mask):
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        param.grad.zero_()  # Zero out bad gradients
                        has_bad_grad = True  # Set flag
            
            if not has_bad_grad:  # If no bad gradients
                mask_opt.step()  # Update parameters
            
            # Apply bounds to mask values
            if lls_variant['bnd'] == 'clip':  # Clipping
                for idx, param in enumerate(curr_mask):
                    # Clip values to [0, 1] range
                    curr_mask[idx].data.clamp_(0, 1)
            elif lls_variant['bnd'] == 'scale':  # Min-max scaling
                all_values = torch.cat([param.view(-1) for param in curr_mask])
                min_val = all_values.min()  # Get minimum value
                max_val = all_values.max()  # Get maximum value
                if max_val > min_val:  # If range is non-zero
                    for idx, param in enumerate(curr_mask):
                        # Scale to [0, 1] range
                        scaled_value = (param.data - min_val) / (max_val - min_val)
                        curr_mask[idx].data = scaled_value
            
            # Update best mask if loss improved
            if mask_loss.item() < best_loss:  # If loss improved
                best_mask = copy.deepcopy(curr_mask)  # Update best mask
                best_loss = mask_loss.item()  # Update best loss
                patience_counter = patience  # Reset patience counter
                
                # Only print for key steps to avoid flood
                if step == 0 or step % 10 == 0 or step == max_steps - 1:
                    print(f"[LLS_FLOW_24] Step {step}: New best loss {best_loss:.6f}")
            else:  # If loss didn't improve
                patience_counter -= 1  # Decrement patience counter
                
                if patience_counter <= 0:  # If patience exhausted
                    print(f"[LLS_FLOW_25] Early stopping at step {step} with best loss {best_loss:.6f}")
                    break  # Stop optimization
            
            # Update masks for next iteration
            init_mask = curr_mask  # Update masks
        
        # ====================== POST-PROCESSING ======================
        
        # Box selection (single box) if specified
        if lls_variant['nbox'] == '1':  # Single box mode
            # Find max mask value
            max_idx = -1
            max_value = -float('inf')
            for i, param in enumerate(best_mask):
                if param.item() > max_value:  # If larger than current max
                    max_value = param.item()  # Update max value
                    max_idx = i  # Update max index
            
            # Zero out all except maximum
            if max_idx >= 0:  # If valid max found
                print(f"[LLS_FLOW_26] Single box mode: keeping only box {max_idx} ({box_names[max_idx]}) = {max_value:.6f}")
                for i, param in enumerate(best_mask):
                    if i != max_idx:  # If not the max box
                        best_mask[i].data = torch.tensor([0.0], device=device)  # Zero out
        
        # Binarize masks if specified
        if lls_variant['bin'] == '1':  # Binarization mode
            threshold = float(lls_variant['mthr']) if 'mthr' in lls_variant else 0.5  # Get threshold
            print(f"[LLS_FLOW_27] Binarizing masks with threshold {threshold}")
            for i, param in enumerate(best_mask):
                # Convert to binary (0 or 1) based on threshold
                binary_value = 1.0 if param.item() >= threshold else 0.0
                best_mask[i].data = torch.tensor([binary_value], device=device)
        
        # Apply mask combination strategy
        if lls_variant['appl'] == 'curr' or 'prev_mask' not in storage:  # Current masks
            final_mask = best_mask  # Use best masks from this iteration
        elif lls_variant['appl'] == 'avg':  # Average with previous
            # Average with previous mask
            count = storage.get('count', 0)  # Get previous count
            prev_mask = storage['prev_mask']  # Get previous masks
            final_mask = copy.deepcopy(best_mask)  # Create copy for final masks
            
            for i, (curr, prev) in enumerate(zip(best_mask, prev_mask)):
                if i < len(box_names):  # Check index is valid
                    # Compute weighted average based on count
                    avg_value = (count * prev.data + curr.data) / (count + 1)
                    final_mask[i].data = avg_value  # Store average
        elif lls_variant['appl'] == 'prod':  # Product with previous
            # Multiply with previous mask
            prev_mask = storage['prev_mask']  # Get previous masks
            final_mask = copy.deepcopy(best_mask)  # Create copy for final masks
            
            for i, (curr, prev) in enumerate(zip(best_mask, prev_mask)):
                if i < len(box_names):  # Check index is valid
                    # Compute product
                    prod_value = curr.data * prev.data
                    final_mask[i].data = prod_value  # Store product
        elif lls_variant['appl'] == 'ema':  # Exponential moving average
            # Exponential moving average
            prev_mask = storage['prev_mask']  # Get previous masks
            alpha = float(lls_variant['emalr'])  # Get EMA alpha
            final_mask = copy.deepcopy(best_mask)  # Create copy for final masks
            
            for i, (curr, prev) in enumerate(zip(best_mask, prev_mask)):
                if i < len(box_names):  # Check index is valid
                    # Compute EMA: (1-alpha)*prev + alpha*curr
                    ema_value = (1 - alpha) * prev.data + alpha * curr.data
                    final_mask[i].data = ema_value  # Store EMA
        else:  # Unknown strategy
            final_mask = best_mask  # Use best masks from this iteration
        
        # Store for next iteration
        storage['prev_mask'] = final_mask  # Store masks
        storage['count'] = storage.get('count', 0) + 1  # Increment count
        
        # Store input for 'prev' loss mode
        storage['prev_x'] = x.detach().clone()  # Store input tensor
        
        # ====================== RETURN VALUES ======================
        
        # Convert to learning rate values for each parameter
        lr_values = []  # List to store mask values
        for name, param in model.named_parameters():
            # Check if parameter is trainable and in allowed layers
            if param.requires_grad and (trainable_layer_names is None or 
                                       get_clean_layer_name(name) in trainable_layer_names):
                box_idx = param_to_box_idx.get(name, -1)  # Get box index
                if box_idx >= 0 and box_idx < len(final_mask):  # If valid box
                    lr_value = final_mask[box_idx].item()  # Get mask value
                    lr_values.append(lr_value)  # Add to list
                else:  # Not in any box
                    lr_values.append(0.0)  # No update
        
        # Create info dict for tracking and debugging
        info = {
            'best_loss': best_loss,
            'mask_dict': {box_names[i]: final_mask[i].item() 
                         for i in range(min(len(box_names), len(final_mask)))},
            'init_metrics': box_metrics,
        }
        
        print(f"[LLS_FLOW_28] *** LLS OPTIMIZATION COMPLETE ***")
        print(f"[LLS_FLOW_29] Final best loss: {best_loss:.6f}")
        print(f"[LLS_FLOW_30] Final mask summary:")
        for i, box_name in enumerate(box_names):
            if i < len(final_mask):
                print(f"[LLS_FLOW_31] {box_name}: {final_mask[i].item():.6f}")
        print(f"[LLS_FLOW_32] ========== LLS EXIT POINT ==========\n")
        
        return lr_values, info
        
    except Exception as e:
        print(f"[LLS_ERROR] CRITICAL ERROR in get_lls_masks: {str(e)}")
        print(traceback.format_exc())
        raise

def get_auto_rgn_values(param_groups):
    """Compute automatic Relative Gradient Norm (RGN) values.
    
    Args:
        param_groups: Optimizer parameter groups
        
    Returns:
        torch.Tensor: RGN values for each parameter
    """
    # Compute the RGNs for each layer
    rgn_values = []  # List to store RGN values
    
    # Loop through parameter groups
    for group in param_groups:
        for p in group['params']:  # Loop through parameters
            if p.grad is None:  # Skip parameters without gradients
                continue
            grad_norm = p.grad.norm()  # Compute gradient norm
            param_norm = p.norm()  # Compute parameter norm
            rgn = grad_norm / (param_norm + 1e-8)  # Compute RGN with epsilon for stability
            rgn_values.append(rgn)  # Add to list

    # Normalize RGNs between 0 and 1
    rgn_values = torch.tensor(rgn_values)  # Convert to tensor
    # Normalize by maximum value
    if rgn_values.max() > 0:  # Avoid division by zero
        rgn_values = (rgn_values) / (rgn_values.max() + 1e-8)
    
    return rgn_values


def get_auto_snr_values(param_groups):
    """Compute automatic Signal-to-Noise Ratio (SNR) values.
    
    Args:
        param_groups: Optimizer parameter groups
        
    Returns:
        torch.Tensor: SNR values for each parameter
    """
    # Compute the SNRs for each layer
    snr_values = []  # List to store SNR values
    
    # Loop through parameter groups
    for group in param_groups:
        for p in group['params']:  # Loop through parameters
            if p.grad is None:  # Skip parameters without gradients
                continue
            grad = p.grad  # Get gradient
            # Compute SNR as squared gradient divided by variance
            snr = (grad*grad) / (torch.var(grad, dim=0, keepdim=True)+1e-8)
            snr = snr.mean()  # Take mean
            snr_values.append(snr)  # Add to list

    # Normalize SNRs between 0 and 1
    snr_values = torch.tensor(snr_values)  # Convert to tensor
    # Binarize based on threshold 0.95
    snr_values = torch.where(snr_values < 0.95,
                             torch.zeros_like(snr_values),
                             torch.ones_like(snr_values))
    
    return snr_values


def diff_lr_optimizer(base_optimizer):
    """Create a new optimizer with separate learning rate for each parameter.
    
    Args:
        base_optimizer: Original optimizer
        
    Returns:
        Optimizer: New optimizer with per-parameter learning rates
    """
    # Get base optimizer parameters
    base_opt_params = base_optimizer.defaults  # Get default parameters
    base_lr = base_opt_params['lr']  # Get base learning rate

    # Create parameter list with individual learning rates
    params = []  # List to store parameter dictionaries
    for group in base_optimizer.param_groups:  # Loop through parameter groups
        for p in group['params']:  # Loop through parameters
            if p.grad is None:  # Skip parameters without gradients
                continue
            params.append({'params': p, 'lr': base_lr})  # Add parameter with base learning rate

    # Create new optimizer with same type and parameters
    new_optimizer = type(base_optimizer)(params, **base_opt_params)
    return new_optimizer


def diff_lr(base_optimizer, lr_factor_values):
    """Apply layer-specific learning rates to optimizer.
    
    Args:
        base_optimizer: Optimizer to modify
        lr_factor_values: List of learning rate scaling factors
    """
    # Check if lists have same length
    if len(base_optimizer.param_groups) != len(lr_factor_values):
        error_msg = f'Length mismatch: {len(lr_factor_values)} lr factors, {len(base_optimizer.param_groups)} param groups'
        raise ValueError(error_msg)
    
    # Get base learning rate
    base_lr = base_optimizer.defaults['lr']

    # Apply scaling factors to each parameter group
    for idx, group in enumerate(base_optimizer.param_groups):
        # Scale learning rate by factor
        lr_layer = base_lr * lr_factor_values[idx]
        group['lr'] = lr_layer  # Set new learning rate


def compute_worst_mask(metric_values_dict, td_variant):
    """Compute masks that select the worst box.
    
    Args:
        metric_values_dict: Dictionary of metrics for each box
        td_variant: Configuration dictionary
        
    Returns:
        dict: Dictionary of mask values for each box
    """
    # Clone the metric_values_dict to mask_dict
    mask_dict = copy.deepcopy(metric_values_dict)

    # Worst case scenario: train the worst box
    min_mask_value = min(mask_dict.values())  # Get minimum metric
    
    for box_name, mask_value in mask_dict.items():
        if mask_value == min_mask_value:  # If this is the worst box
            mask_dict[box_name] = torch.tensor(1.0, device=mask_value.device)  # Set to 1
        else:  # Not the worst box
            mask_dict[box_name] = torch.tensor(0.0, device=mask_value.device)  # Set to 0

    return mask_dict


def compute_not_best_mask(metric_values_dict, td_variant):
    """Compute masks that select all boxes except the best one.
    
    Args:
        metric_values_dict: Dictionary of metrics for each box
        td_variant: Configuration dictionary
        
    Returns:
        dict: Dictionary of mask values for each box
    """
    # Clone the metric_values_dict to mask_dict
    mask_dict = copy.deepcopy(metric_values_dict)

    # Train all boxes except the best one
    max_mask_value = max(mask_dict.values())  # Get maximum metric
    
    # Check if this is the first sample (all metrics equal)
    metrics_sum = sum(v.item() for v in mask_dict.values())
    first_sample_in_window = abs(metrics_sum - len(mask_dict)) < 1e-6
    
    for box_name, mask_value in mask_dict.items():
        # Train box if it's not the best or this is the first sample
        if mask_value != max_mask_value or first_sample_in_window:
            mask_dict[box_name] = torch.tensor(1.0, device=mask_value.device)  # Set to 1
        else:  # Best box (unless first sample)
            mask_dict[box_name] = torch.tensor(0.0, device=mask_value.device)  # Set to 0

    return mask_dict


def compute_random_mask(metric_values_dict, td_variant):
    """Compute masks that select a random box.
    
    Args:
        metric_values_dict: Dictionary of metrics for each box
        td_variant: Configuration dictionary
        
    Returns:
        dict: Dictionary of mask values for each box
    """
    # Clone the metric_values_dict to mask_dict
    mask_dict = copy.deepcopy(metric_values_dict)

    # Choose a random box to train
    box_names = list(mask_dict.keys())  # Get box names
    random_box = random.choice(box_names)  # Random box name

    # Train only the random box
    for box_name, mask_value in mask_dict.items():
        if box_name == random_box:  # If this is the chosen box
            mask_dict[box_name] = torch.tensor(1.0, device=mask_value.device)  # Set to 1
        else:  # Not the chosen box
            mask_dict[box_name] = torch.tensor(0.0, device=mask_value.device)  # Set to 0

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
    # Compute mask for each specified box
    # Extract box names from mask_algo string
    parts = mask_algo.lower().split('_')
    box_parts = [part for part in parts if 'layer' in part or 'stage' in part]
    
    mask_dict = {box_name: 1 for box_name in box_parts}

    # Get mask values for each layer
    mask_values = get_mask_values(mask_dict, model, trainable_layer_names)
    return mask_values


def get_es_variant(mask_algo):
    """Parse early stopping configuration from mask_algo string.
    
    Args:
        mask_algo: Configuration string
        
    Returns:
        dict: Dictionary of early stopping configuration options
    """
    # Create td hparams dictionary with default values
    td_variant = {
        'pat'       : '5',      # Early stopping patience: 0, 3, 5, 10
        'type'      : 'hard',   # Early stopping type: hard, soft
        'dataset'   : 'ood',    # Early stopping set: iid, ood
        'lvl'       : 'Block',  # Early stopping granularity: Layer, Block, Subblock
    }

    # Update the td hparams dictionary with the given hyperparameters
    for hparam in mask_algo.split('_'):  # Split by underscore
        if ':' in hparam:  # Check if it's a key-value pair
            key, value = hparam.split(':')  # Split into key and value
            if key not in td_variant:  # Check if key is valid
                error_msg = f'Invalid mask_hparam: {key}'
                raise ValueError(error_msg)
            td_variant[key] = value.lower()  # Update configuration value
    
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
    global stop_flag_dict, min_loss_dict, steps_no_improve_dict  # Use global variables

    ### Prepare everything
    # Get the es variant
    es_variant = get_es_variant(mask_algo)  # Parse configuration
    es_variant['iter_idx'] = iter_idx  # Set iteration index
    
    # Get all the block names
    granularity = es_variant['lvl']  # Get granularity
    box_names = get_all_param_name_list(model, granularity)  # Get box names

    # Initialize stop flags for first iteration
    if iter_idx == 0:
        stop_flag_dict = {box_name: False for box_name in box_names}

    # Get the data loader for the validation set
    if es_variant['dataset'] == 'iid':  # IID validation set
        if 'iidte_loader' not in kwargs:
            error_msg = "iidte_loader not found in kwargs"
            raise ValueError(error_msg)
        loader = kwargs['iidte_loader']  # Get loader
    elif es_variant['dataset'] == 'ood':  # OOD validation set
        if 'oodte_loader' not in kwargs:
            error_msg = "oodte_loader not found in kwargs"
            raise ValueError(error_msg)
        loader = kwargs['oodte_loader']  # Get loader

    # Log intermediate results
    info = {
        'valid_loss_dict': {}  # Dictionary to store validation losses
    }

    # Copy all the current model parameters and gradients
    all_curr_params_dict, all_curr_grads_dict = {}, {}  # Store current state
    for name, params in model.named_parameters():
        all_curr_params_dict[name] = params.data.clone()  # Copy parameter
        if params.grad is None:  # If no gradient
            all_curr_grads_dict[name] = None  # Store None
        else:  # If gradient exists
            all_curr_grads_dict[name] = params.grad.clone()  # Copy gradient
    
    ### Simulate the optimizer step and get mask dict
    mask_dict = {}  # Dictionary to store masks
    
    # Process each box
    for box_name in box_names:
        # No need to check for early stopping if already stopped
        if es_variant['type'] == 'hard' and stop_flag_dict[box_name] == True:
            mask_dict[box_name] = 0  # Don't update this box
            continue

        # Get the parameter list for optimizer step (without cloning)
        temp_params_list = []  # List to store parameters for temp optimizer
        
        # Loop through model parameters
        for idx, (name, params) in enumerate(model.named_parameters()):
            # Check if parameter is trainable
            trainable_layer = is_layer_trainable(name, trainable_layer_names)
            if params.requires_grad == True and trainable_layer:
                if box_name in name:  # If parameter belongs to this box
                    temp_params_list.append(params)  # Add to temp parameters

        # Get the base optimizer's parameters
        base_opt_params = copy.deepcopy(base_optimizer.defaults)  # Copy optimizer settings

        # Temporary optimizer step for the model
        temp_optimizer = type(base_optimizer)(temp_params_list, **base_opt_params)
        temp_optimizer.step()  # Perform update step

        # Get the loss for the validation set
        sup_loss = 0  # Initialize loss
        device = next(model.parameters()).device  # Get device
        
        # Compute validation loss
        count = 0  # Counter for number of batches
        with torch.no_grad():  # No gradients needed
            for x, y in loader:  # Loop through validation batches
                x = x.to(device)  # Move input to device
                y = y.to(device)  # Move target to device
                p = model(x)  # Forward pass
                batch_loss = F.cross_entropy(p, y).item()  # Compute batch loss
                sup_loss += batch_loss  # Add batch loss
                count += 1  # Increment batch counter
        
        if count > 0:  # If we have batches
            sup_loss /= count  # Average loss

        # Check for early stopping
        stop_flag = early_stopping_with_loss(
            sup_loss, int(es_variant['pat']), iter_idx, box_name)
        
        # Store flags and losses
        stop_flag_dict[box_name] = stop_flag  # Update stop flag
        info['valid_loss_dict'][box_name] = sup_loss  # Store loss

        # Compute mask for box
        mask_dict[box_name] = 1 if stop_flag == False else 0  # 1 if not stopped, 0 if stopped

        # Restore all the current model's parameters and grads
        for name, params in model.named_parameters():
            params.data = all_curr_params_dict[name]  # Restore parameter
            params.grad = all_curr_grads_dict[name]  # Restore gradient

    # Get mask values for each layer
    mask_values = get_mask_values(mask_dict, model, trainable_layer_names)
    info['mask_dict'] = mask_dict  # Store masks in info

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
    global min_loss_dict, steps_no_improve_dict  # Use global dictionaries

    # If the first step
    if iter_idx == 0:  # First iteration
        min_loss_dict[box_name] = float('inf')  # Initialize minimum loss
        steps_no_improve_dict[box_name] = 0  # Initialize steps counter

    # If the validation loss is at a minimum
    if loss <= min_loss_dict[box_name]:  # If loss improved
        min_loss_dict[box_name] = loss  # Update minimum loss
        steps_no_improve_dict[box_name] = 0  # Reset counter
    else:  # Loss didn't improve
        steps_no_improve_dict[box_name] += 1  # Increment counter
        # Check early stopping condition
        if steps_no_improve_dict[box_name] > patience:  # If patience exceeded
            return True  # Should stop
        
    return False  # Should continue


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
    
    print(f"\n[MAIN_FLOW_1] ========== GET_NEW_OPTIMIZER ENTRY ==========")
    print(f"[MAIN_FLOW_2] mask_algo: {mask_algo}, iter_idx: {iter_idx}")
    print(f"[MAIN_FLOW_3] track_flag: {track_flag}, kwargs keys: {list(kwargs.keys())}")
    
    try:
        info = {}  # Dictionary for tracking info
        new_opt_flag = False  # Flag for optimizer modification

        # Clean trainable layer names if provided
        if trainable_layer_names is not None:
            trainable_layer_names = [get_clean_layer_name(name) for name in trainable_layer_names]
        
        # Choose layer selection method based on mask_algo
        if 'least' in mask_algo.lower():  # LEAST method
            print(f"[MAIN_FLOW_4] Using LEAST layer selection")
            mask_values, info = get_LEAST_masks(mask_algo, model, optimizer, iter_idx, trainable_layer_names, track_flag=True, **kwargs)
            new_opt_flag = True
        elif 'lls' in mask_algo.lower():  # LLS method
            print(f"[MAIN_FLOW_5] *** LLS METHOD DETECTED ***")
            # Check for required input tensor
            if 'x' not in kwargs:
                error_msg = "Input tensor 'x' is required for LLS but not provided"
                print(error_msg)
                raise ValueError(error_msg)
            print(f"[MAIN_FLOW_6] Input tensor confirmed, calling get_lls_masks...")
            # Get LLS masks
            mask_values, info = get_lls_masks(mask_algo, model, optimizer, iter_idx, trainable_layer_names, **kwargs)
            new_opt_flag = True
            print(f"[MAIN_FLOW_7] LLS call returned, new_opt_flag: {new_opt_flag}")
        elif 'layer' in mask_algo.lower() or 'stage' in mask_algo.lower():  # Explicit layer selection
            print(f"[MAIN_FLOW_8] Using explicit layer selection")
            mask_values = get_box_masks(mask_algo, model, trainable_layer_names)
            new_opt_flag = True
        elif 'valid' in mask_algo.lower():  # Validation-based selection
            print(f"[MAIN_FLOW_9] Using validation-based layer selection")
            mask_values, info = get_valid_mask(mask_algo, model, optimizer, iter_idx, trainable_layer_names, **kwargs)
            new_opt_flag = True
        elif 'autorgn' in mask_algo.lower():  # Automatic RGN-based selection
            print(f"[MAIN_FLOW_10] Using automatic RGN-based layer selection")
            mask_values = get_auto_rgn_values(optimizer.param_groups)
            new_opt_flag = True
        elif 'autosnr' in mask_algo.lower():  # Automatic SNR-based selection
            print(f"[MAIN_FLOW_11] Using automatic SNR-based layer selection")
            mask_values = get_auto_snr_values(optimizer.param_groups)
            new_opt_flag = True
        else:
            print(f"[MAIN_FLOW_12] No layer selection method identified in: {mask_algo}")
            if track_flag:
                return optimizer, info
            else:
                return optimizer

        # Apply layer-specific learning rates if needed
        if new_opt_flag:
            print(f"[MAIN_FLOW_13] Applying layer-specific learning rates...")
            # Create per-parameter optimizer if needed
            if len(optimizer.param_groups) == 1:
                optimizer = diff_lr_optimizer(optimizer)
            
            # Apply layer-specific learning rates
            diff_lr(optimizer, mask_values)
            
            # Print final learning rates summary (first few only)
            if len(optimizer.param_groups) > 0:
                sample_size = min(3, len(optimizer.param_groups))
                lr_sample = [f"{g['lr']:.6f}" for g in optimizer.param_groups[:sample_size]]
                print(f"[MAIN_FLOW_14] Sample learning rates: {lr_sample} (total: {len(optimizer.param_groups)} groups)")

        # Return optimizer and info based on track_flag
        print(f"[MAIN_FLOW_15] ========== GET_NEW_OPTIMIZER EXIT ==========\n")
        if track_flag:
            return optimizer, info
        else:
            return optimizer
            
    except Exception as e:
        print(f"[MAIN_ERROR] CRITICAL ERROR in get_new_optimizer: {str(e)}")
        print(traceback.format_exc())
        raise
