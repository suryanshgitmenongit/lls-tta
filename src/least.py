import torch
import torchvision

import copy


def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_all_param_name_list(algorithm, granularity):
    granularity = granularity.lower()
    model = algorithm
    if hasattr(algorithm, 'featurizer'):
        model = algorithm.featurizer.network
    elif hasattr(algorithm, 'network'):
        model = algorithm.network

    if isinstance(model, torchvision.models.ResNet):
        pre_box_list = []
        post_box_list = []
        layer_name = 'layer'
        if granularity == 'block':
            granularity_level = 1
            # pre_box_list = ['layer0']
            # post_box_list = ['classifier']
        elif granularity == 'subblock':
            granularity_level = 2
            # pre_box_list = ['layer0']
            # post_box_list = ['classifier']
        elif granularity == 'layer':
            granularity_level = -1
            # pre_box_list = ['layer0.conv1', 'layer0.bn1']
            # post_box_list = ['classifier']
        elif granularity == 'model':
            return [layer_name]
        else:
            raise NotImplementedError('Not implemented for granularity : {}'.format(granularity))

    elif len(list(model.named_parameters())) == 80: # Standard.pt or WRN model
        #TODO: Change this to check if model is an instance of torchvision.models.ResNet
        pre_box_list = []
        post_box_list = []
        layer_name = 'layer'
        if granularity == 'block':
            granularity_level = 1
            # pre_box_list = ['layer0']
            # post_box_list = ['classifier']
        elif granularity == 'subblock':
            granularity_level = 3
            # pre_box_list = ['layer0']
            # post_box_list = ['classifier']
        elif granularity == 'layer':
            granularity_level = 4
            # pre_box_list = ['layer0.conv1', 'layer0.bn1']
            # post_box_list = ['classifier']
        elif granularity == 'model':
            return [layer_name]
        else:
            raise NotImplementedError('Not implemented for granularity : {}'.format(granularity))
    
    elif len(list(model.named_parameters())) == 95: # Hendrycks2020AugMix_ResNeXt.pt
        pre_box_list = []
        post_box_list = []
        layer_name = 'stage'
        if granularity == 'block':
            granularity_level = 1
            # pre_box_list = ['layer0']
            # post_box_list = ['classifier']
        elif granularity == 'subblock':
            granularity_level = 2
            # pre_box_list = ['layer0']
            # post_box_list = ['classifier']
        elif granularity == 'layer':
            granularity_level = 3
            # pre_box_list = ['layer0.conv1', 'layer0.bn1']
            # post_box_list = ['classifier']
        elif granularity == 'model':
            return [layer_name]
        else:
            raise NotImplementedError('Not implemented for granularity : {}'.format(granularity))
        
    else:
        raise NotImplementedError('Unrecognized model type')
    
    all_box_list = remove_duplicates([".".join(name.split(".")[:granularity_level])
                                            for name, _ in model.named_parameters()
                                                if layer_name in name])
        
    # if granularity == 'Layer':
    #     all_box_list = sorted(set([box
    #                                for box in all_box_list
    #                                     if 'conv' in box]))
    return pre_box_list + all_box_list + post_box_list

def get_clean_layer_name(layer_name):
    remove_words = ['featurizer', 'network']
    clean_name = '.'.join([split_name for split_name in layer_name.split('.')
                            if split_name not in remove_words])
    return clean_name

def is_layer_trainable(name, trainable_layer_names):
    if trainable_layer_names is None:
        return True
    if get_clean_layer_name(name) in trainable_layer_names:
        return True
    else:
        return False


# Global variables
source_params_dict = {}
window_start_params_dict = {}
window_index = None
# box_names = ['layer1', 'layer2', 'layer3', 'layer4']    # For block granularity

def get_td_variant(mask_algo):
    # Create td hparams dictionary with default values
    td_variant = {
        # Metric design choices
        'nbox'   : 'n',         # number of boxes: 1, n
        'curr'   : 'next',      # td = <> - anchor: next, prev
        'bin'    : '1',         # binarize the mask: 0, 1
        'trust'  : 'low',       # trust on init samples: high, low
        'lvl'    : 'Layer',     # granularity of the model: Layer, Block, Subblock

        # Metric's hyperparameters
        'sz'     : '20',        # window size: 5, 20, n
        'mthr'   : '0.75',      # threshold based on metric range: 0.5, 0.75, 0.99
        'sthr'   : '0.75',      # if trust == low, no. of sample / window sz threshold: 0.5, 0.75, 0.9
    }

    # Update the td hparams dictionary with the given hyperparameters
    for hparam in mask_algo.split('_'):
        if ':' in hparam:
            key, value = hparam.split(':')
            if key not in td_variant:
                raise ValueError('Invalid mask_hparam : {}'.format(key))
            td_variant[key] = value.lower()
    
    return td_variant


def get_anchor_params_dict(before_params_dict, td_variant):
    global source_params_dict, window_start_params_dict, window_index

    iter_idx = td_variant['iter_idx']
    
    anchor_params_dict = {}
    
    if iter_idx == 0:
        # Start of TTA for any target domain
        # Store the current parameters as source/window start/anchor parameters
        source_params_dict = copy.deepcopy(before_params_dict)
        anchor_params_dict = copy.deepcopy(before_params_dict)
        window_start_params_dict = copy.deepcopy(before_params_dict)
        window_index = 0

    else:
        if td_variant['sz'] == 'n':
            # Clone the source parameters to anchor parameters
            if not source_params_dict:
                raise ValueError('TD Error: Source parameters are empty')
            anchor_params_dict = copy.deepcopy(source_params_dict)
            window_index += 1

        elif td_variant['sz'].isdigit() and int(td_variant['sz']) > 0:
            # Reset the window start parameters after every sz iterations
            if iter_idx % int(td_variant['sz']) == 0:
                window_start_params_dict = copy.deepcopy(before_params_dict)
                window_index = 0
            else:
                window_index += 1
            
            # Clone the window start parameters to anchor parameters
            if not window_start_params_dict:
                raise ValueError('TD Error: Window start parameters are empty')
            anchor_params_dict = copy.deepcopy(window_start_params_dict)

        else:
            raise ValueError('TD Error: Invalid window size : {}'.format(td_variant['sz']))

    return anchor_params_dict

def get_curr_params_dict(before_params_dict, after_params_dict, td_variant):
    if td_variant['curr'] == 'next':
        return copy.deepcopy(after_params_dict)
    elif td_variant['curr'] == 'prev':
        return copy.deepcopy(before_params_dict)        
    else:
        raise ValueError('TD Error: Get current parameters not implemented for {}'.format(td_variant['curr']))

def compute_total_delta_tensor_of_box(curr_params, anchor_params, td_variant):
    if curr_params.shape != anchor_params.shape:
        raise ValueError('TD Error: Shape mismatch for curr ({}) and anchor ({}) parameters'.format(curr_params.shape, anchor_params.shape))
    
    total_delta = curr_params - anchor_params
    
    return total_delta

def get_total_delta_tensor(anchor_params_dict, curr_params_dict, td_variant):
    total_delta_tensor_dict = {}
    for box_name in anchor_params_dict.keys():
        # Get the flattened current and anchor parameters
        curr_params = torch.cat([params.flatten() for params in curr_params_dict[box_name]])
        anchor_params = torch.cat([params.flatten() for params in anchor_params_dict[box_name]])

        # Compute the total delta for the current box
        total_delta_tensor_dict[box_name] = compute_total_delta_tensor_of_box(curr_params, anchor_params, td_variant)

    return total_delta_tensor_dict

def get_traj_td_tensor_dict(before_params_dict, after_params_dict, td_variant):
    # Get the anchor and current parameters
    anchor_params_dict = get_anchor_params_dict(before_params_dict, td_variant)
    curr_params_dict = get_curr_params_dict(before_params_dict, after_params_dict, td_variant)

    # Compute the total delta
    total_delta_tensor_dict = get_total_delta_tensor(anchor_params_dict, curr_params_dict, td_variant)

    return total_delta_tensor_dict

def get_samp_td_tensor_dict(before_params_dict, after_params_dict, td_variant):
    # Get the anchor and current parameters
    anchor_params_dict = copy.deepcopy(before_params_dict)
    curr_params_dict = copy.deepcopy(after_params_dict)

    # Compute the total delta
    total_delta_tensor_dict = get_total_delta_tensor(anchor_params_dict, curr_params_dict, td_variant)

    return total_delta_tensor_dict

def compute_metric_values(td_traj_tensor_dict, td_samp_tensor_dict, td_variant):
    metric_values = {}
    for box_name in td_traj_tensor_dict.keys():
        td_traj_tensor = td_traj_tensor_dict[box_name]
        td_samp_tensor = td_samp_tensor_dict[box_name]

        # Return cosine distance of the traj and samp total deltas
        metric_value = torch.cosine_similarity(td_traj_tensor, td_samp_tensor, dim=0)
        # if metric_value > 1-1e-4:
        #         metric_value = torch.ones_like(metric_value)
        metric_values[box_name] = metric_value

    return metric_values

def get_mistrust_window_sz(td_variant):
    if td_variant['sz'] == 'n':
        # Hardcoding the window size for trust window if not provided
        # Cannot set extremely large as it will considerably slow down adaptation
        sz = 50
    else:
        sz = int(td_variant['sz'])
    sthr = float(td_variant['sthr'])
    return int(sz * sthr)

def compute_mask(metric_values_dict, td_variant):
    # Clone the metric_values_dict to mask_dict
    mask_dict = copy.deepcopy(metric_values_dict)

    # Mask appln: single or multiple boxes
    nbox = td_variant['nbox']
    if nbox == '1':
        # Single box: Store the mask for box with max metric value
        for box_name, mask_value in mask_dict.items():
            if mask_value == max(mask_dict.values()):
                mask_dict[box_name] = mask_value
            else:
                mask_dict[box_name] = 0      
    elif nbox == 'n':
        # Multiple boxes: Do nothing
        pass
    else:
        raise ValueError('TD Error: Invalid number of boxes : {}'.format(nbox))

    # Mask appln per-box: binary (based on mthr), weighted (based on metric values)
    bin_mask = td_variant['bin']
    mthr = float(td_variant['mthr'])
    if bin_mask == '1':
        # Binarize the mask based on mthr
        for box_name, mask_value in mask_dict.items():
            if mask_value > mthr:
                mask_dict[box_name] = 1
            else:
                mask_dict[box_name] = 0
    elif bin_mask == '0':
        # Layer-wise mask based on metric values: do nothing
        # for box_name, mask_value in mask_dict.items():
        #     mask_dict[box_name] = mask_value
        pass
    else:
        raise ValueError('TD Error: Invalid binarize mask : {}'.format(bin_mask))

    # Mask appln based on trust on init samples: high, low, no (based on sthr)
    trust = td_variant['trust']
    if trust == 'low':
        # Low trust: Scale the mask based on the number of samples / mistrust window size
        mistrust_window_sz = get_mistrust_window_sz(td_variant)
        if window_index+1 < mistrust_window_sz:
            mistrust_factor = (window_index+1) / mistrust_window_sz
            for box_name, mask_value in mask_dict.items():
                mask_dict[box_name] *= mistrust_factor
    elif trust == 'high':
        # High trust: Trust the metric/mask on init samples
        pass
    else:
        raise ValueError('TD Error: Invalid trust on init samples : {}'.format(trust))

    return mask_dict

def get_mask_values(trainable_lr_dict, model, trainable_layer_names):
    lr_values = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        trainable_layer = is_layer_trainable(name, trainable_layer_names)
        if param.requires_grad and trainable_layer:
            lr_value = 0
            for dname in trainable_lr_dict.keys():
                # if 'layer0' in dname and idx < 3:# and \
                #     # isinstance(feat_model, torchvision.models.ResNet):
                #     # Handle for layer0
                #     if dname == 'layer0':
                #         lr_value = trainable_lr_dict[dname]
                #     elif dname == 'layer0.conv1' and idx == 0:
                #         lr_value = trainable_lr_dict[dname]
                #     elif dname == 'layer0.bn1' and idx >= 1 and idx <= 2:
                #         lr_value = trainable_lr_dict[dname]
                if dname in name:
                    lr_value = trainable_lr_dict[dname]
            lr_values.append(float(lr_value))

    if sum(lr_values) <= 0:
        # print('TD Warning: No trainable parameters found for final mask_dict : {}'.format(trainable_lr_dict))
        pass
    
    return lr_values

def get_LEAST_masks(mask_algo, model, base_optimizer, iter_idx, trainable_layer_names):

    ### Step 1: Prepare everything
    # Get the td variant
    td_variant = get_td_variant(mask_algo)
    td_variant['iter_idx'] = iter_idx
    
    # Get all the block names
    granularity = td_variant['lvl']
    box_names = get_all_param_name_list(model, granularity)

    # Copy all the current model parameters and gradients
    all_curr_params_dict = {name: params.data.clone() \
                            for name, params in model.named_parameters()}
    # Since total delta is only used in the adaptation phase,
    # we can assume that grad is not None
    # It is good that grad.clone breaks if grad is None
    # TODO: make it explicit
    all_curr_grads_dict = {name: params.grad.clone() \
                            for name, params in model.named_parameters()}
    
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
    temp_optimizer = type(base_optimizer)(temp_params_list, **base_opt_params)
    temp_optimizer.step()

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

    new_optimizer = type(base_optimizer)(params, **base_opt_params)
    return new_optimizer

def diff_lr(base_optimizer, lr_factor_values):
    if len(base_optimizer.param_groups) != len(lr_factor_values):
        raise ValueError('Length of lr_factor_values and param_groups do not match')
    
    base_lr = base_optimizer.defaults['lr']

    for idx, group in enumerate(base_optimizer.param_groups):
        lr_layer = base_lr * lr_factor_values[idx]
        group['lr'] = lr_layer

    return 

#TODO: A reset iter_idx function which can be called during optimizer reset

def get_new_optimizer(mask_algo, model, optimizer, iter_idx, trainable_layer_names=None, track_flag=False, **kwargs):
    info = {}
    new_opt_flag = False
    
    if 'least' in mask_algo.lower():
        mask_values, info = get_LEAST_masks(mask_algo, model, optimizer, iter_idx, trainable_layer_names)
        new_opt_flag = True

    if new_opt_flag:
        if len(optimizer.param_groups) == 1:
            optimizer = diff_lr_optimizer(optimizer)
        diff_lr(optimizer, mask_values)

    if track_flag:
        return optimizer, info
    else:
        return optimizer