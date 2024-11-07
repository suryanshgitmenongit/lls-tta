import torch
import torchvision
import torch.nn.functional as F

import copy
import random
from least import *


def get_LEAST_masks(mask_algo, model, base_optimizer, iter_idx, trainable_layer_names, **kwargs):

    ### Step 1: Prepare everything
    # Get the td variant
    td_variant = get_td_variant(mask_algo)
    td_variant['iter_idx'] = iter_idx
    
    # Get all the block names
    granularity = td_variant['lvl']
    box_names = get_all_param_name_list(model, granularity)
    # to enable training on specific trainable layers
    # if trainable_layer_names is not None:
    #     box_names = remove_duplicates([box_name for box_name in box_names \
    #                                         for trainable_layer_name in trainable_layer_names \
    #                                             if box_name in trainable_layer_name])

    # Copy all the current model parameters and gradients
    all_curr_params_dict, all_curr_grads_dict = {}, {}
    for name, params in model.named_parameters():
        all_curr_params_dict[name] = params.data.clone()
        if params.grad is None:
            all_curr_grads_dict[name] = None
        else:
            all_curr_grads_dict[name] = params.grad.clone()
    
    ### Step 2: Simulate the optimizer step for cloned model and optimizer
    # Get the base optimizer's parameters
    base_opt_params = copy.deepcopy(base_optimizer.defaults)
    
    # Get the parameter list for optimizer step (without cloning)
    # Also, get the before-TTA parameter dict (by cloning)
    temp_params_list = []
    before_params_dict = {}
    for box_name in box_names:
        before_params_dict[box_name] = []
        for idx, (name, params) in enumerate(model.named_parameters()):
            trainable_layer = is_layer_trainable(name, trainable_layer_names)
            if params.requires_grad == True and trainable_layer:
                # if box_name == 'layer0' and idx < 3: # and \
                #     # isinstance(model, torchvision.models.ResNet):
                #     temp_params_list.append(params)
                #     before_params_dict[box_name].append(params.data.clone())
                if box_name in name:
                    temp_params_list.append(params)
                    before_params_dict[box_name].append(params.data.clone())

    # Temporary optimizer step for the model
    temp_optimizer_type = type(base_optimizer.base_optimizer)
    temp_optimizer = type(base_optimizer)(temp_params_list, temp_optimizer_type, **base_opt_params)

    x = kwargs['x']
    margin = kwargs['margin']
    ema = kwargs['ema']
    filter_ids_1 = kwargs['filter_ids_1']
    optimizer_step_fn = kwargs['optimizer_fn']
    optimizer_step_fn(x, model, temp_optimizer, margin, ema, filter_ids_1)
    # temp_optimizer.step()

    # Get the after-TTA parameter dict (by cloning): TODO: automate this
    after_params_dict = {}
    box_names = get_all_param_name_list(model, granularity)
    for box_name in box_names:
        after_params_dict[box_name] = []
        for idx, (name, params) in enumerate(model.named_parameters()):
            trainable_layer = is_layer_trainable(name, trainable_layer_names)
            if params.requires_grad == True and trainable_layer:
                # if box_name == 'layer0' and idx < 3: # and \
                #     # isinstance(model, torchvision.models.ResNet):
                #     temp_params_list.append(params)
                #     before_params_dict[box_name].append(params.data.clone())
                if box_name in name:
                    after_params_dict[box_name].append(params.data.clone())

    ### Step 3: Mask computation for each box
    # Get trajectory TD
    td_traj_tensor_dict = get_traj_td_tensor_dict(before_params_dict, after_params_dict, td_variant)

    # Get sample/batch TD
    td_samp_tensor_dict = get_samp_td_tensor_dict(before_params_dict, after_params_dict, td_variant)
    
    # Compute mask for each box
    metric_values_dict = compute_metric_values(td_traj_tensor_dict, td_samp_tensor_dict, td_variant)

    ### Step 4: Mask application for each box
    mask_dict = compute_mask(metric_values_dict, td_variant)
    # print('Mask values : ', mask_dict)

    ### Step 5: Pack up everything    
    # Restore all the current model's parameters and grads
    for name, params in model.named_parameters():
        params.data = all_curr_params_dict[name]
        params.grad = all_curr_grads_dict[name]

    # Get mask values for each layer
    mask_values = get_mask_values(mask_dict, model, trainable_layer_names)

    # Log intermediate results
    info = {}
    info['metric_values_dict'] = metric_values_dict
    info['mask_dict'] = mask_dict

    return mask_values, info


def diff_lr_optimizer(base_optimizer):
    base_opt_params = base_optimizer.defaults
    base_lr = base_opt_params['lr']

    params = []
    for group in base_optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            params.append({'params': p, 'lr': base_lr})

    optimizer_type = type(base_optimizer.base_optimizer)
    new_optimizer = type(base_optimizer)(params, optimizer_type, **base_opt_params)
    return new_optimizer


def get_new_optimizer(mask_algo, model, optimizer, iter_idx, trainable_layer_names=None, track_flag=False, **kwargs):
    info = {}
    new_opt_flag = False

    if trainable_layer_names is not None:
        trainable_layer_names = [get_clean_layer_name(name) for name in trainable_layer_names]
    
    if 'least' in mask_algo.lower():
        mask_values, info = get_LEAST_masks(mask_algo, model, optimizer, iter_idx, trainable_layer_names, **kwargs)
        new_opt_flag = True

    if new_opt_flag:
        if len(optimizer.param_groups) == 1:
            optimizer = diff_lr_optimizer(optimizer)
        diff_lr(optimizer, mask_values)

    if track_flag:
        return optimizer, info
    else:
        return optimizer