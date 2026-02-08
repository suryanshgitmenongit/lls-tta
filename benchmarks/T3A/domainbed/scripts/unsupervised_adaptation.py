# The code is modified from domainbed.scripts.train
# ==============================================================================
# COMPLETE FIXED VERSION - All Oracle HP Selection Bugs Resolved
# ==============================================================================
# This is the COMPLETE script with ALL bug fixes applied:
# 
# BUG FIX #2: Cache validation in get_best_hp() (lines ~430-530)
#   - CRITICAL FIX: Extracts ALL configs from entire hparams_list
#   - Previous version only checked hparams_list[0], missing other configs
#   - Now correctly validates multi-config oracle runs
#   - Prevents contamination from old runs with different layer selection configs
#   - Deletes invalid cache files automatically
#
# BUG FIX #3: File mode logic (lines ~1280-1290)
#   - Always overwrites for first config (hp_idx == 0)
#   - Prevents contamination from old result files
#
# BUG FIX #4: Results file hp_mode suffix (lines ~1270-1280)
#   - Adds _orc or _iid suffix to separate oracle/iid results
#   - Prevents file contamination between HP selection modes  
#
# BUG FIX #7: Consistent loc_alg_name (lines ~37-57, 588-605)
#   - Always uses base method name (LEAST, LLS) regardless of mode
#   - Ensures consistent file naming between oracle and manual modes
#   - Works correctly for single config oracle submissions
#
# BUG FIX #9: Parallel execution support via oracle_exp_id (lines ~588, 625, 1282)
#   - Adds optional --oracle_exp_id parameter for parallel oracle runs
#   - Prevents file collisions when running multiple experiments simultaneously
#   - Includes exp_id in filenames and results JSON
# ==============================================================================

import argparse
from argparse import Namespace
import collections
import json
import os
import random
import sys
import time
import uuid
from itertools import chain
import itertools
import copy
import wandb
from tqdm import tqdm

import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn.functional as F
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
sys.path.append('../../src')
import least

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.misc import print_dbg
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, DataParallelPassthrough
from domainbed import model_selection
from domainbed.lib.query import Q
from domainbed import adapt_algorithms
import itertools


def get_base_method_name(find_algo_str):
    """Extract base method name from find_algo string for file naming.
    
    Handles comma-separated lists by taking the first config.
    Extracts base name before first underscore, with special handling for LLS_init:least.
    
    Examples:
        'LEAST_sz:5_mthr:0.5_lvl:Layer' -> 'LEAST'
        'LLS_init:least_steps:10_lr:0.01_...' -> 'LLS'
        'LEAST,LEAST_sz:5...' -> 'LEAST'
    """
    # If comma-separated, take first
    if ',' in find_algo_str:
        find_algo_str = find_algo_str.split(',')[0].strip()
    
    # Handle special case: LLS_init:least should return 'LLS'
    if find_algo_str.startswith('LLS'):
        return 'LLS'
    
    # Extract base name before first underscore
    if '_' in find_algo_str:
        return find_algo_str.split('_')[0]
    
    return find_algo_str

# Default is True, set to False during debugging
PRODN=True

# Default is True, set to False to use faster but non-PRODN sample order
CONST_ORDER=True

# Default is True, set to False if you want to recompute the best hp
RESUSE_BEST_HP=True


class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

def generate_featurelized_loader(loader, network, classifier, batch_size=32):
    """
    The classifier adaptation does not need to repeat the heavy forward path, 
    We speeded up the experiments by converting the observations into representations. 
    """
    z_list = []
    y_list = []
    p_list = []
    network.eval()
    classifier.eval()
    for x, y in loader:
        x = x.to(device)
        z = network(x)
        p = classifier(z)
        
        z_list.append(z.detach().cpu())
        y_list.append(y.detach().cpu())
        p_list.append(p.detach().cpu())
        # p_list.append(p.argmax(1).float().cpu().detach())
    network.train()
    classifier.train()
    z = torch.cat(z_list)
    y = torch.cat(y_list)
    p = torch.cat(p_list)
    ent = softmax_entropy(p)
    py = p.argmax(1).float().cpu().detach()
    dataset1, dataset2 = Dataset(z, y), Dataset(z, py)
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=False, drop_last=True)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False, drop_last=True)
    return loader1, loader2, ent


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def get_model_stats(model, layer_names=None):
    if layer_names is None:
        layer_names = least.get_all_param_name_list(algorithm, args.granularity)
        layer_names += [layer_names[0][:-1]] # Add the 'all' layer

    param_dict, grad_dict = {}, {}
    for layer_name in layer_names:
        layer_param_list, layer_grad_list = [], []
        for idx, (name, params) in enumerate(model.named_parameters()):
            # Skip if no grad
            if params.requires_grad is False:
                continue

            # Handle for layer0
            layer0_cond = False
            if 'layer0' in layer_name and idx < 3:# and \
                # isinstance(feat_model, torchvision.models.ResNet):
                cond1 = layer_name == 'layer0'
                cond2 = layer_name == 'layer0.conv1' and idx == 0
                cond3 = layer_name == 'layer0.bn1' and idx >= 1 and idx <= 2
                layer0_cond = cond1 or cond2 or cond3

            # Add to list if block_name is in name
            if layer0_cond or layer_name in name:
                layer_param_list.append(params.flatten())

                if params.grad is None:
                    layer_grad_list.append(torch.zeros_like(params).flatten())
                else:
                    layer_grad_list.append(params.grad.flatten())

        if len(layer_param_list) == 0:
            continue
        else:
            param_dict[layer_name] = torch.cat(layer_param_list).detach().clone()
            grad_dict[layer_name] = torch.cat(layer_grad_list).detach().clone()

    stats_dict = {}
    stats_dict['param_dict'] = param_dict
    stats_dict['grad_dict'] = grad_dict

    return stats_dict

reset_sz = 20       #TODO: make it a hyperparameter
all_u, all_td = {}, {}
sum_u = {}
first_dict = None
first_dict_with_reset = None

def visualize_layer_stats(next_dict, prev_dict, step):
    global first_dict, first_dict_with_reset, all_u, all_td, sum_u
    
    if step == 0:
        first_dict = copy.deepcopy(prev_dict)
        first_dict_with_reset = first_dict
    elif step % reset_sz == 0:
        first_dict_with_reset = copy.deepcopy(prev_dict)

    layer_names = least.get_all_param_name_list(algorithm, args.granularity)
    layer_names += [layer_names[0][:-1]] # Add the 'all' layer

    track_batch_results = {}
    for layer_name in layer_names:
        # Get all the layer params
        next_param_layer = next_dict['param_dict'][layer_name]
        prev_param_layer = prev_dict['param_dict'][layer_name]
        first_param_layer = first_dict['param_dict'][layer_name]
        first_param_with_reset_layer = first_dict_with_reset['param_dict'][layer_name]

        # Get all the layer vectors
        u = next_param_layer - prev_param_layer
        td = prev_param_layer - first_param_layer
        td_with_u = next_param_layer - first_param_layer
        td_with_reset = prev_param_layer - first_param_with_reset_layer
        td_with_u_with_reset = next_param_layer - first_param_with_reset_layer

        if step == 0:
            sum_u[layer_name] = torch.zeros_like(u)
            all_u[layer_name] = []
            all_td[layer_name] = []
        sum_u[layer_name] += torch.abs(u)
        all_u[layer_name].append(u.detach().cpu().clone())
        all_td[layer_name].append(td.detach().cpu().clone())

        # Log all the metrics
        track_batch_results[layer_name + '_u_magn'] = torch.norm(u).item()
        track_batch_results[layer_name + '_sum_u_magn'] = torch.norm(sum_u[layer_name]).item()

        track_batch_results[layer_name + '_td_magn'] = torch.norm(td).item()
        track_batch_results[layer_name + '_td_with_u_magn'] = torch.norm(td_with_u).item()
        track_batch_results[layer_name + '_td_with_reset_magn'] = torch.norm(td_with_reset).item()
        track_batch_results[layer_name + '_td_with_u_with_reset_magn'] = torch.norm(td_with_u_with_reset).item()
        track_batch_results[layer_name + '_td_by_d_magn'] = torch.norm(td).item() / td.numel()
        track_batch_results[layer_name + '_td_with_u_by_d_magn'] = torch.norm(td_with_u).item() / td_with_u.numel()
        track_batch_results[layer_name + '_td_with_reset_by_d_magn'] = torch.norm(td_with_reset).item() / td_with_reset.numel()
        track_batch_results[layer_name + '_td_with_u_with_reset_by_d_magn'] = torch.norm(td_with_u_with_reset).item() / td_with_u_with_reset.numel()

        if torch.norm(td) != 0:
            track_batch_results[layer_name + '_cos_td_and_u'] = torch.cosine_similarity(td, u, dim=0).item()
        if torch.norm(td_with_u) != 0:
            track_batch_results[layer_name + '_cos_td_with_u_and_u'] = torch.cosine_similarity(td_with_u, u, dim=0).item()
        if torch.norm(td_with_reset) != 0:
            track_batch_results[layer_name + '_cos_td_with_reset_and_u'] = torch.cosine_similarity(td_with_reset, u, dim=0).item()
        if torch.norm(td_with_u_with_reset) != 0:
            track_batch_results[layer_name + '_cos_td_with_u_with_reset_and_u'] = torch.cosine_similarity(td_with_u_with_reset, u, dim=0).item()

    return track_batch_results


def visualize_layer_movement():
    def pca(X, k=2):
        # Centering the data
        X_mean = torch.mean(X,0)
        X = X - X_mean.expand_as(X)

        # Computing the covariance matrix
        U,S,V = torch.svd(torch.t(X))
        return torch.mm(X,U[:,:k]), S/torch.sum(S)
    
    def get_effective_rank(normalized_eigenvalues):
        # Calculate the entropy
        entropy = -torch.sum(normalized_eigenvalues * torch.log(normalized_eigenvalues))
        
        # Calculate the effective rank
        effective_rank = torch.exp(entropy)
        
        return effective_rank.item()
    
    vis_results = {}
    td_explained_variance_dict = {}
    for layer_name, td_list in all_td.items():
        # Create a torch tensor of vectors
        X_td = torch.stack(td_list)

        # Perform PCA
        X_td_pca, td_explained_variance = pca(X_td, k=2)
        td_explained_variance_dict[layer_name] = td_explained_variance
        
        # Center the data at the first point
        X_td_pca = X_td_pca - X_td_pca[0]

        # Create a plot of X_td_pca arrows, connecting consecutive points
        cmap = mcolors.LinearSegmentedColormap.from_list("n", ["blue", "red"])
        fig = plt.figure(figsize=(10,10))
        
        colors = cmap(np.linspace(0, 1, len(X_td_pca)))
        for i in range(len(X_td_pca) - 1):
            plt.arrow(X_td_pca[i,0], X_td_pca[i,1], 
              X_td_pca[i+1,0] - X_td_pca[i,0], 
              X_td_pca[i+1,1] - X_td_pca[i,1], 
              width=0.0001,
              fc=colors[i],
              ec=colors[i])
        plt.scatter(0, 0, c='black', s=50)
        plt.title("Layer: {} - td".format(layer_name))
        plt.grid()
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.axis('equal')
        # plt.savefig(layer_name + '_td_movement.png')

        vis_results[layer_name + '_td_movement'] = wandb.Image(fig)
        vis_results[layer_name + '_effective_rank'] = get_effective_rank(td_explained_variance)
        vis_results[layer_name + '_variance_pc1'] = td_explained_variance[0].item()
        vis_results[layer_name + '_variance_pc2'] = td_explained_variance[1].item()
        vis_results[layer_name + '_variance_pc3'] = td_explained_variance[2].item()
        

    # Plot all the explained variance
    fig = plt.figure(figsize=(10,10))
    for layer_name, td_explained_variance in td_explained_variance_dict.items():
        plt.plot(td_explained_variance, 'o-', label=layer_name)
    plt.legend()
    plt.title("Explained Variance")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.grid()
    # plt.savefig('explained_variance.png')
    vis_results[layer_name + '_explained_variance'] = wandb.Image(fig)

    return vis_results


def accuracy_ent(network, loader, weights, device, adapt=False):
    correct = 0
    total = 0
    weights_offset = 0
    ent = 0

    network.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, total=len(loader), desc='accuracy_ent', disable=PRODN):
            x = x.to(device)
            y = y.to(device)
            if adapt is None:
                p = network(x)
            else:
                p = network(x, adapt)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            ent += softmax_entropy(p).sum().item()
    network.train()

    return correct / total, ent / total

def get_batch_noise(network, x, y):
    train_state = network.training
    network.eval()
    outputs = network(x, adapt=False)
    py, y_prime = F.softmax(outputs, dim=-1).max(1)

    prediction_acc = y_prime.eq(y).float().sum().item() / len(y)
    prediction_noise = 1 - prediction_acc

    flag = py > adapt_hparams['beta']
    filtered_prediction_acc = y_prime[flag].eq(y[flag]).float().sum().item() / len(y[flag])
    filtered_prediction_noise = 1 - filtered_prediction_acc

    if train_state:
        network.train()
    return prediction_noise, filtered_prediction_noise

def accuracy_ent_noise(network, loader, weights, device, adapt=False):
    correct = 0
    total = 0
    weights_offset = 0
    ent = 0
    p_noise_list, fp_noise_list = [], []

    network.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, total=len(loader), desc='accuracy_ent_noise', disable=PRODN):
            x = x.to(device)
            y = y.to(device)
            p_noise, fp_noise = get_batch_noise(network, x, y)
            p_noise_list.append(p_noise)
            fp_noise_list.append(fp_noise)
            if adapt is None:
                p = network(x)
            else:
                p = network(x, adapt)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            ent += softmax_entropy(p).sum().item()
    network.train()

    return correct / total, ent / total, p_noise_list, fp_noise_list


def accuracy_ent_loss(network, loader, weights, device, adapt=False):
    correct = 0
    total = 0
    weights_offset = 0
    ent = 0
    total_unsup_loss, total_sup_loss = 0, 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if adapt is None:
                p = network(x)
            else:
                p = network(x, adapt)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            ent += softmax_entropy(p).sum().item()

            ### Additional loss computation steps
            total_unsup_loss += network.loss(p)
            total_sup_loss += F.cross_entropy(p, y)
    network.train()

    return correct / total, ent / total, total_unsup_loss / total, total_sup_loss / total


def accuracy_ent_track(network, oodtr_evals, oodte_evals, iidte_evals, device, ckpt_freq, adapt=False):
    correct = 0
    total = 0
    weights_offset = 0
    ent = 0

    oodtr_name, loader, weights = oodtr_evals
    total_unsup_loss, total_sup_loss = 0, 0
    prev_dict = get_model_stats(network)

    network.eval()
    with torch.no_grad():
        for step, (x, y) in tqdm(enumerate(loader), total=len(loader), desc='accuracy_ent_track', disable=PRODN):
            x = x.to(device)
            y = y.to(device)
            if adapt is None:
                p = network(x)
            elif 'valid' in args.find_algo.lower() and adapt:
                p = network(x, adapt, oodte_loader=oodte_evals[0][1], iidte_loader=iidte_evals[0][1])
            else:
                p = network(x, adapt)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                per_batch_correct = (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                correct += per_batch_correct
            else:
                per_batch_correct = (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                correct += per_batch_correct
            total += batch_weights.sum().item()
            ent += softmax_entropy(p).sum().item()

            ### Additional tracking steps
            track_results = {}

            # Track accuracy and entropy on the ood training data
            track_results['batch_acc_'+oodtr_name] = per_batch_correct / batch_weights.sum().item()
            track_results['oodtr_acc_'+oodtr_name] = correct / total
            
            # Track losses on the ood training data
            batch_unsup_loss = network.loss(p)
            total_unsup_loss += batch_unsup_loss
            track_results['batch_unsup_loss_'+oodtr_name] = batch_unsup_loss.item()
            track_results['oodtr_unsup_loss_'+oodtr_name] = total_unsup_loss / total

            batch_sup_loss = F.cross_entropy(p, y)
            total_sup_loss += batch_sup_loss
            track_results['batch_sup_loss_'+oodtr_name] = batch_sup_loss.item()
            track_results['oodtr_sup_loss_'+oodtr_name] = total_sup_loss / total

            # Track mask based intermediate results
            if 'least' in args.find_algo.lower():
                for layer_name, metric_value in network.info['metric_values_dict'].items():
                    track_results['metric_value_' +layer_name] = metric_value.item()
                for layer_name, mask_value in network.info['mask_dict'].items():
                    track_results['mask_value_' +layer_name] = mask_value
                for layer_name, mask_value in network.info['mask_dict'].items():
                    track_results['selected_value_' +layer_name] = int(mask_value > 0)
            elif 'valid' in args.find_algo.lower():
                for layer_name, mask_value in network.info['mask_dict'].items():
                    track_results['mask_value_' +layer_name] = mask_value
                for layer_name, loss_value in network.info['valid_loss_dict'].items():
                    track_results['valid_loss_' +layer_name] = loss_value

            # Track the layer stats
            next_dict = get_model_stats(network)
            track_batch_results = visualize_layer_stats(next_dict, prev_dict, step)
            track_results.update(track_batch_results)
            prev_dict = next_dict

            if (step % ckpt_freq == 0) or (step == len(loader) - 1):
                # Track accuracy and losses on the ood test data
                for oodte_name, oodte_loader, oodte_weights in oodte_evals:
                    oodte_acc, oodte_ent, oodte_unsup_loss, oodte_sup_loss = accuracy_ent_loss(network, oodte_loader, oodte_weights, device, adapt=False)
                    track_results["oodte_acc_"+oodte_name] = oodte_acc
                    track_results["oodte_unsup_loss_"+oodte_name] = oodte_unsup_loss
                    track_results["oodte_sup_loss_"+oodte_name] = oodte_sup_loss

                # Track accuracy and losses on the iid test data
                for iidte_name, iidte_loader, iidte_weights in iidte_evals:
                    iidte_acc, iidte_ent, iidte_unsup_loss, iidte_sup_loss = accuracy_ent_loss(network, iidte_loader, iidte_weights, device, adapt=False)
                    track_results["iidte_acc_"+iidte_name] = iidte_acc
                    track_results["iidte_unsup_loss_"+iidte_name] = iidte_unsup_loss
                    track_results["iidte_sup_loss_"+iidte_name] = iidte_sup_loss
                    break  # Only track the first iidte_loader, TODO: remove this line

            # Log the tracked results
            wandb.log(track_results, step=step)
        # vis_results = visualize_layer_movement()
        # wandb.log(vis_results, step=step)

    network.train()

    return correct / total, ent / total


def get_best_hp(evals_list, hparams_list, hp_idx):
    """
    Select best hyperparameters by evaluating all configs on validation data.
    
    ===========================================================================
    CRITICAL BUG FIX #2: Cache validation prevents contamination
    ===========================================================================
    FINAL FIX: Extracts ALL unique find_algos from ENTIRE hparams_list.
    
    Previous broken version only checked hparams_list[0]['find_algo'], which:
    - Worked for single-config (all hparams have same find_algo)
    - FAILED for multi-config oracle (each hparams has different find_algo)
    
    Example of the bug:
      hparams_list = [
          {'alpha': 0.1, 'find_algo': "LEAST_sz:5"},   # hparams_list[0]
          {'alpha': 1.0, 'find_algo': "LEAST_sz:5"},
          {'alpha': 0.1, 'find_algo': "LEAST_sz:20"},  # Different!
          {'alpha': 1.0, 'find_algo': "LEAST_sz:20"},
      ]
      Old code: current_find_algos = {"LEAST_sz:5"}  # Only first!
      New code: current_find_algos = {"LEAST_sz:5", "LEAST_sz:20"}  # All!
    
    This now correctly validates multi-config oracle runs and prevents 
    contamination from old runs with different layer selection methods.
    ===========================================================================
    """
    best_acc = -1
    best_hparams = {}

    track_path = os.path.join(args.output_dir, 
                              'hpSearch_{}_{}_ord{}_{}.jsonl'.format(
                                  alg_name, loc_alg_name, CONST_ORDER, hp_idx))
    
    # ========================================================================
    # CRITICAL FIX: Validate cache before reuse
    # ========================================================================
    if os.path.exists(track_path) and RESUSE_BEST_HP:
        print(f"[CACHE] Found existing hpSearch file: {os.path.basename(track_path)}")
        
        # ====================================================================
        # CRITICAL BUG FIX: Extract ALL configs from ENTIRE hparams_list
        # ====================================================================
        # Iterate through ALL hparams to collect every unique find_algo value
        # This handles both single-config and multi-config oracle correctly
        # ====================================================================
        all_find_algos = set()
        for hp in hparams_list:
            fa = hp.get('find_algo')
            if isinstance(fa, list):
                # If it's a list, add all elements
                all_find_algos.update(fa)
            elif fa:
                # If it's a string, add it
                all_find_algos.add(fa)
        
        current_find_algos = all_find_algos
        print(f"[CACHE] Current run testing {len(current_find_algos)} unique configs:")
        for i, cfg in enumerate(sorted(current_find_algos), 1):
            print(f"  [{i}] {cfg}")
        
        adapt_hparams_keys = list(hparams_list[0].keys()) + ['hp_acc']
        cache_valid = True
        cache_line_count = 0
        cached_configs_found = set()
        
        with open(track_path, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                    
                cache_line_count += 1
                
                try:
                    # Read the hparams and acc into a dict
                    hparams_results = json.loads(line[:-1])
                except json.JSONDecodeError as e:
                    print(f"[CACHE INVALID] Line {line_num}: JSON decode error: {e}")
                    cache_valid = False
                    break
                
                # VALIDATION STEP 1: Check required keys exist
                missing_keys = [k for k in adapt_hparams_keys if k not in hparams_results.keys()]
                if missing_keys:
                    print(f"[CACHE INVALID] Line {line_num}: Missing keys: {missing_keys}")
                    cache_valid = False
                    break
                
                # VALIDATION STEP 2: Check find_algo is in current run's config set
                cached_find_algo = hparams_results.get('find_algo')
                
                # Normalize to string for comparison
                if isinstance(cached_find_algo, list):
                    if len(cached_find_algo) == 1:
                        cached_find_algo_str = cached_find_algo[0]
                    else:
                        print(f"[CACHE INVALID] Line {line_num}: Unexpected list length: {len(cached_find_algo)}")
                        cache_valid = False
                        break
                elif isinstance(cached_find_algo, str):
                    cached_find_algo_str = cached_find_algo
                else:
                    print(f"[CACHE INVALID] Line {line_num}: Unexpected find_algo type: {type(cached_find_algo)}")
                    cache_valid = False
                    break
                
                # Check if this cached config is in our current set
                if cached_find_algo_str not in current_find_algos:
                    print(f"[CACHE INVALID] Line {line_num}: Config not in current run!")
                    print(f"  Cached config: {cached_find_algo_str}")
                    print(f"  Current configs: {sorted(current_find_algos)}")
                    cache_valid = False
                    break
                
                # Track which configs we've seen in cache
                cached_configs_found.add(cached_find_algo_str)
                
                # If we reach here, this line is valid - extract results
                adapt_hparams = {k: v for k, v in hparams_results.items() 
                               if k in adapt_hparams_keys}
                mean_acc = hparams_results['hp_acc']

                # Save the best hparams and acc
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_hparams = copy.deepcopy(adapt_hparams)
        
        # VALIDATION STEP 3: Check cache completeness
        # Cache is only valid if it contains results for ALL current configs
        if cache_valid:
            missing_configs = current_find_algos - cached_configs_found
            if missing_configs:
                print(f"[CACHE INVALID] Missing results for {len(missing_configs)} configs:")
                for cfg in sorted(missing_configs):
                    print(f"  - {cfg}")
                cache_valid = False
        
        if cache_valid and best_hparams:
            print(f"[CACHE HIT] Successfully validated {cache_line_count} lines")
            print(f"[CACHE HIT] All {len(current_find_algos)} configs found in cache")
            print(f"[CACHE HIT] Best accuracy: {best_acc:.4f}")
            print(f"[CACHE HIT] Best config: {best_hparams.get('find_algo', 'N/A')}")
            return best_hparams, best_acc
        else:
            print(f"[CACHE INVALID] Discarding cache file and recomputing from scratch")
            # Delete invalid cache to prevent future confusion
            try:
                os.remove(track_path)
                print(f"[CACHE] Deleted invalid file: {os.path.basename(track_path)}")
            except Exception as e:
                print(f"[CACHE] Warning: Could not delete cache file: {e}")
            # Reset and fall through to recomputation
            best_acc = -1
            best_hparams = {}

    # ========================================================================
    # Compute HP selection from scratch
    # ========================================================================
    assert len(evals_list) > 0, "evals_list cannot be empty for HP selection"
    assert len(hparams_list) > 0, "hparams_list cannot be empty for HP selection"
    
    print(f"[HP SEARCH] Starting HP search with {len(hparams_list)} configurations")
    print(f"[HP SEARCH] Evaluating on {len(evals_list)} validation split(s):")
    for eval_name, _, _ in evals_list:
        print(f"  - {eval_name}")

    for config_idx, adapt_hparams in enumerate(tqdm(hparams_list, desc='get_best_hp', disable=PRODN)):
        # CRITICAL: Reset random seeds before each config evaluation
        # This ensures each config starts from the same random state,
        # preventing non-determinism from cascading across configs
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        
        adapt_hparams['cached_loader'] = use_featurer_cache
        adapted_algorithm = adapt_algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), adapt_hparams, algorithm
        )
        # adapted_algorithm = DataParallelPassthrough(adapted_algorithm)
        adapted_algorithm.to(device)

        hparams_results = {}
        hparams_results.update(adapt_hparams)

        score = []
        for name, loader, weights in evals_list:
            print_dbg('evaluating with adapt=True on {}...'.format(name), not PRODN)
            acc, ent = accuracy_ent(adapted_algorithm, loader, weights, device, adapt=True)
            score.append(acc)
            hparams_results[name+'_acc'] = acc
            adapted_algorithm.reset()
        
        mean_acc = np.mean(score)
        hparams_results['hp_acc'] = mean_acc
        adapt_hparams['hp_acc']  = mean_acc
        del adapt_hparams['cached_loader']
        if 'src_info_dict' in hparams_results.keys():
            del hparams_results['src_info_dict']
        
        # Track best configuration
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_hparams = copy.deepcopy(adapt_hparams)
            config_name = adapt_hparams.get('find_algo', 'N/A')
            if isinstance(config_name, list):
                config_name = config_name[0] if config_name else 'N/A'
            print(f"[HP SEARCH] New best: {mean_acc:.4f} with config: {config_name}")
        
        # Write to cache file (append mode)
        with open(track_path, 'a') as f:
            f.write(json.dumps(hparams_results, sort_keys=True) + "\n")
    
    print(f"[HP SEARCH] Completed HP search")
    print(f"[HP SEARCH] Final best accuracy: {best_acc:.4f}")
    print(f"[HP SEARCH] Final best config: {best_hparams.get('find_algo', 'N/A')}")
    
    return best_hparams, best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--adapt_algorithm', type=str, default="T3A")
    parser.add_argument('--find_algo', type=str, default="None")
    parser.add_argument('--fix_algo', type=str, default="None")
    parser.add_argument('--granularity', type=str, default="None")
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--best_hp', type=str, default='Man')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--track_tta', type=str, default="False")
    parser.add_argument('--ckpt_freq', type=int, default=10000)
    parser.add_argument('--adapt_hparams_dict', type=str, default="{}",
        help='JSON-serialized adaptation hparams dict')
    parser.add_argument('--oracle_exp_id', type=str, default=None,
        help='Experiment identifier for parallel oracle runs to prevent file collisions')
    args_in = parser.parse_args()

    epochs_path = os.path.join(args_in.input_dir, 'results.jsonl')
    records = []
    with open(epochs_path, 'r') as f:
        for line in f:
            records.append(json.loads(line[:-1]))
    records = Q(records)
    r = records[0]
    args = Namespace(**r['args'])
    args.input_dir = args_in.input_dir
    if not not args_in.data_dir:
        args.data_dir = args_in.data_dir

    if '-' in args_in.adapt_algorithm:
        args.adapt_algorithm, test_batch_size = args_in.adapt_algorithm.split('-')
        args.test_batch_size = int(test_batch_size)
    else:
        args.adapt_algorithm = args_in.adapt_algorithm
        args.test_batch_size = 32  # default

    args.find_algo = args_in.find_algo
    args.fix_algo = args_in.fix_algo
    args.granularity = args_in.granularity
    args.output_dir = args.input_dir
    args.best_hp = args_in.best_hp
    args.track_tta = args_in.track_tta
    args.ckpt_freq = args_in.ckpt_freq
    args.adapt_hparams_dict = args_in.adapt_hparams_dict

    # Set all Macros as args
    args.PRODN = PRODN
    args.CONST_ORDER = CONST_ORDER
    args.RESUSE_BEST_HP = RESUSE_BEST_HP
    
    alg_name = args_in.adapt_algorithm

    if args.adapt_algorithm in['T3A', 'TentPreBN', 'TentClf', 'PLClf']:
        use_featurer_cache = True
        loc_alg_name = "Nolocadapt"
        loc_dict = {}
    else:
        use_featurer_cache = False
        # ==================================================================
        # CRITICAL BUG FIX #7: Always use base name for consistency
        # ==================================================================
        # ORIGINAL: Single configs used full config string, oracle used base
        # FIXED: Always extract base name regardless of oracle/manual mode
        # This ensures consistent file naming and cache file reuse logic
        # Works correctly for both single-config and multi-config oracle
        # ==================================================================
        base_name = get_base_method_name(args.find_algo)
        
        # ==================================================================
        # BUG FIX #9: Add oracle_exp_id to loc_alg_name if provided
        # ==================================================================
        # Allows parallel execution of multiple oracle experiments by
        # differentiating their filenames through unique experiment IDs
        # ==================================================================
        if hasattr(args_in, 'oracle_exp_id') and args_in.oracle_exp_id:
            loc_alg_name = f"{base_name}_{args_in.oracle_exp_id}{args.fix_algo}{args.granularity}"
        else:
            loc_alg_name = base_name + args.fix_algo + args.granularity
        
        print(f"[CONFIG] find_algo input: {args.find_algo}")
        print(f"[CONFIG] Extracted base name: {base_name}")
        if hasattr(args_in, 'oracle_exp_id') and args_in.oracle_exp_id:
            print(f"[CONFIG] oracle_exp_id: {args_in.oracle_exp_id}")
        print(f"[CONFIG] Final loc_alg_name: {loc_alg_name}")
        
        # Parse comma-separated find_algos for oracle selection
        if ',' in args.find_algo:
            # Multiple configs: split and test all
            find_algo_list = [s.strip() for s in args.find_algo.split(',')]
            print(f"[CONFIG] Oracle mode detected: {len(find_algo_list)} configs to test")
            for i, cfg in enumerate(find_algo_list):
                print(f"  Config {i+1}: {cfg}")
        else:
            # Single config
            find_algo_list = [args.find_algo]
            print(f"[CONFIG] Single config mode: {args.find_algo}")

        loc_dict = {
            'find_algo': find_algo_list,  # Can be multiple values for oracle
            'granularity': [args.granularity],
            'track_tta': [args.track_tta],
        }
    
    # Get the td variant - FIXED VERSION to handle LLS
    mask_algo = args.find_algo + '_lvl:' + args.granularity
    
    # Check if this is an LLS configuration to avoid parsing error
    if mask_algo.lower().startswith('lls'):
        # For LLS, we only need the order parameter for data loader setup
        td_variant = {'order': '0'}  # Default order
    else:
        # For LEAST and other methods, parse as usual
        td_variant = least.get_td_variant(mask_algo)
    
    if 'least' in args.find_algo.lower():
        # Store td variant as args
        for k, v in td_variant.items():
            setattr(args, "LEAST_"+k, v)
        
    if os.path.exists(os.path.join(args.output_dir, 'done_{}_{}_{}'.format(alg_name, loc_alg_name, args.best_hp.lower()))):
        print("{} with {} with best_hp:{} has already executed".format(alg_name, loc_alg_name, args.best_hp.lower()))
        sys.exit(0)

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    algorithm_dict = None
    # os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out_{}_{}.txt'.format(alg_name, loc_alg_name)))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err_{}_{}.txt'.format(alg_name, loc_alg_name)))

    args.wandb_name = args_in.wandb_name    # Overwrites DG's wandb name
    args.task = 'test_time_adaptation'
    if args.wandb_name is None:
        args.wandb_mode = "disabled" # no name given -> do not init wandb
    else:
        if PRODN:
            args.wandb_mode = "offline"
        else:    
            args.wandb_mode = "online"

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    assert os.path.exists(os.path.join(args.output_dir, 'done'))
    assert os.path.exists(os.path.join(args.output_dir, 'IID_best.pkl'))  # IID_best is produced by train.py

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Additional determinism: force all PyTorch operations to be deterministic
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    
    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    # Use out splits as training data (to fair comparison with train.py)
    train_loaders = [FastDataLoader(
            dataset=env,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(out_splits)
            if i in args.test_envs]
    if CONST_ORDER:
        print("Get all train loaders")
        order_seed = int(td_variant['order'])   # default is 0
        train_loaders_new = []
        for loader in train_loaders:
            # Save the last copy of the loader based on the order seed
            for _ in range(order_seed+1):
                new_loader = list(loader)
            train_loaders_new.append(new_loader)
        train_loaders = train_loaders_new
    
    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=args.test_batch_size,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    if CONST_ORDER:
        print("Get all eval loaders")
        order_seed = int(td_variant['order'])   # default is 0
        eval_loaders_new = []
        for loader in tqdm(eval_loaders, desc="eval_loaders", disable=PRODN):
            # Save the last copy of the loader based on the order seed
            for _ in range(order_seed+1):
                new_loader = list(loader)
            eval_loaders_new.append(new_loader)
        eval_loaders = eval_loaders_new
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    if hasattr(algorithm, 'network'):
        algorithm.network = DataParallelPassthrough(algorithm.network)
    else:
        for m in algorithm.children():
            m = DataParallelPassthrough(m)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    # load trained model
    ckpt = torch.load(os.path.join(args.output_dir, 'IID_best.pkl'))
    algorithm_dict = ckpt['model_dict']
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    print("\nAfter {} with {}".format(alg_name, loc_alg_name))
    # Cache the inference results
    if use_featurer_cache:
        original_evals = zip(eval_loader_names, eval_loaders, eval_weights)
        loaders = []
        for name, loader, weights in original_evals:
            loader1, loader2, ent = generate_featurelized_loader(loader, network=algorithm.featurizer, classifier=algorithm.classifier, batch_size=32)
            loaders.append((name, loader1, weights))
    else:
        loaders = zip(eval_loader_names, eval_loaders, eval_weights)
    
    evals = []
    for name, loader, weights in loaders:
        if name in ['env{}_in'.format(i) for i in args.test_envs]:
            train_loader = (name, loader, weights)
        else:
            evals.append((name, loader, weights))

    last_results_keys = None
    adapt_algorithm_class = adapt_algorithms.get_algorithm_class(
        args.adapt_algorithm)

    if 'cmdline' in args.best_hp.lower():
        adapt_hparams_dict = json.loads(args.adapt_hparams_dict)
        if not bool(adapt_hparams_dict):
            raise Exception("best_hp is cmdline, but adapt_hparams_dict is empty")
        if args.adapt_algorithm in ['PseudoLabel', 'PLClf']:
            if 'beta' not in adapt_hparams_dict.keys():
                adapt_hparams_dict['beta'] = [0.9]
        elif args.adapt_algorithm in ['SHOT', 'SHOTIM']:
            if 'beta' not in adapt_hparams_dict.keys():
                adapt_hparams_dict['beta'] = [0.9]
            if 'theta' not in adapt_hparams_dict.keys():
                adapt_hparams_dict['theta'] = [0.1]
    elif args.adapt_algorithm in ['T3A']:
        adapt_hparams_dict = {
            'filter_K': [1, 5, 20, 50, 100, -1], 
        }
    elif args.adapt_algorithm in ['TentFull', 'TentPreBN', 'TentClf', 'TentNorm']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3]
        }
    elif args.adapt_algorithm in ['PseudoLabel', 'PLClf']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3], 
            'beta': [0.9]
        }
    elif args.adapt_algorithm in ['SHOT', 'SHOTIM']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3], 
            'beta': [0.9], 
            'theta': [0.1], 
        }
    elif args.adapt_algorithm in ['SAR']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3],
        }
    else:
        raise Exception("Not Implemented Error")
    if 'domaingen' in args.find_algo.lower():
        # TODO: Remove this, eg: doesn't work with T3A
        adapt_hparams_dict = {
            'alpha': [0],
            'gamma': [1],
            'beta':  [0],
            'theta': [0.1],
        }
    adapt_hparams_dict.update(loc_dict)
    product = [x for x in itertools.product(*adapt_hparams_dict.values())]
    adapt_hparams_list = [dict(zip(adapt_hparams_dict.keys(), r)) for r in product]
    
    print(f"\n[HPARAMS] Generated {len(adapt_hparams_list)} hyperparameter configurations")

    # Perform Model selection (for reducing computation overload and wandb logging)
    if args.best_hp.lower() in ['first']:
        adapt_hparams_list[0]['hp_mode'] = "First"
        adapt_hparams_list = [adapt_hparams_list[0]]
        print(f"[HP MODE] Using FIRST configuration only")
    elif args.best_hp.lower() in ['last']:
        adapt_hparams_list[-1]['hp_mode'] = "Last"
        adapt_hparams_list = [adapt_hparams_list[-1]]
        print(f"[HP MODE] Using LAST configuration only")
    #TODO: implement leave-one-out model selection
    #TODO: Remove the second condition below
    if args.best_hp.lower() in ['true', 'iid', 'orc', 'man'] and 'domaingen' not in args.find_algo.lower():
        print("="*80)
        print(f"ORACLE/IID HP SELECTION MODE: {args.best_hp.upper()}")
        print("="*80)
        
        tmp_run = wandb.init(mode="disabled") # to avoid timeout waiting error
        train_out_evals, test_out_evals = [], []
        for name, loader, weights in evals:
            if name in ['env{}_out'.format(i) for i in args.test_envs]:
                test_out_evals.append((name, loader, weights))
            elif 'out' in name:
                train_out_evals.append((name, loader, weights))
        
        iid_evals = train_out_evals     # IID Accuracy model selection
        oracle_evals = test_out_evals   # Oracle model selection
        manual_evals = [train_loader]   # Manually chosen best HP on target domain
        
        print(f"\n[HP SELECTION] Validation splits:")
        print(f"  IID (source out_split): {len(iid_evals)} environments")
        print(f"  Oracle (target out_split): {len(oracle_evals)} environments")
        print(f"  Manual (target in_split): {len(manual_evals)} environments")
        
        new_adapt_hparams_list = []
        print("\n[HP SELECTION] Computing best hyperparameters on each split:")
        
        if args.best_hp.lower() in ['true', 'iid']:
            print(f"\n--- IID HP Selection ---")
            iid_hparams, iid_acc = get_best_hp(iid_evals, adapt_hparams_list, "Iid")
            print(f"[IID] Best accuracy: {iid_acc:.4f}")
            print(f"[IID] Best config: {iid_hparams.get('find_algo', 'N/A')}")
            iid_hparams['hp_mode'] = "Iid"
            new_adapt_hparams_list.append(iid_hparams)

        if args.best_hp.lower() in ['true', 'orc']:
            print(f"\n--- Oracle HP Selection ---")
            oracle_hparams, oracle_acc = get_best_hp(oracle_evals, adapt_hparams_list, "Orc")
            print(f"[Oracle] Best accuracy: {oracle_acc:.4f}")
            print(f"[Oracle] Best config: {oracle_hparams.get('find_algo', 'N/A')}")
            oracle_hparams['hp_mode'] = "Orc"
            new_adapt_hparams_list.append(oracle_hparams)
        
        if args.best_hp.lower() in ['true', 'man']:
            print(f"\n--- Manual HP Selection ---")
            manual_hparams, manual_acc = get_best_hp(manual_evals, adapt_hparams_list, "Man")
            print(f"[Manual] Best accuracy: {manual_acc:.4f}")
            print(f"[Manual] Best config: {manual_hparams.get('find_algo', 'N/A')}")
            manual_hparams['hp_mode'] = "Man"
            new_adapt_hparams_list.append(manual_hparams)

        adapt_hparams_list = new_adapt_hparams_list
        print(f"\n[HP SELECTION] Final list contains {len(adapt_hparams_list)} configuration(s)")

        tmp_run.finish()

    # ========================================================================
    # Main evaluation loop - test selected HP configurations
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SPLIT")
    print("="*80)
    
    for hp_idx, adapt_hparams in tqdm(enumerate(adapt_hparams_list), desc="adapt_hparams", total=len(adapt_hparams_list), disable=PRODN):
        if 'hp_mode' in adapt_hparams.keys():
            hp_str = "hp" + adapt_hparams['hp_mode']
        elif 'cmdline' in args.best_hp.lower():
            hp_str = "hpcmd"
        else:
            hp_str = "hp" + str(hp_idx)
        if CONST_ORDER:
            order_str = '_' + 'prodn'
        else:
            order_str = ''
        if args.wandb_mode == "disabled":
            wandb_name = None
        else:
            wandb_name = args.wandb_name + order_str + '_' +  alg_name + "_" + loc_alg_name + "_" + hp_str
        
        print(f"\n[EVAL] Configuration {hp_idx+1}/{len(adapt_hparams_list)}: {hp_str}")
        print(f"[EVAL] find_algo: {adapt_hparams.get('find_algo', 'N/A')}")
        
        wandb_run = wandb.init(project="stta",
                                entity="amazing-ml",
                                name=wandb_name,
                                group=args.wandb_name,
                                config=args,
                                mode=args.wandb_mode,
                                dir=args.output_dir)
        wandb.config.update(hparams)
        wandb.config.update({'hp_idx': hp_idx})
        wandb.config.update(adapt_hparams, allow_val_change=True)

        adapt_hparams['cached_loader'] = use_featurer_cache
        adapted_algorithm = adapt_algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), adapt_hparams, algorithm
        )
        # adapted_algorithm = DataParallelPassthrough(adapted_algorithm)
        adapted_algorithm.to(device)
        # wandb.watch(adapted_algorithm.model, log='all', log_freq=1)
        
        results = adapt_hparams

        for key, val in checkpoint_vals.items():
            results[key] = np.mean(val)

        # ## Usual evaluation, TODO: remove this and account it in pandas_results.py
        for name, loader, weights in evals:
            if not PRODN:
                break           # use for faster target evaluation sometimes
            print_dbg('evaluating with adapt=True on {}...'.format(name), not PRODN)
            acc, ent = accuracy_ent(adapted_algorithm, loader, weights, device, adapt=True)
            results[name+'_acc'] = acc
            results[name+'_ent'] = ent
            adapted_algorithm.reset()

        # ## TTA evaluation
        if args.track_tta.lower() != 'true':
            name, loader, weights = train_loader
            print_dbg('evaluating with adapt=True on {}...'.format(name), not PRODN)
            acc, ent = accuracy_ent(adapted_algorithm, loader, weights, device, adapt=True)
        else:
            oodtr_evals = train_loader
            oodte_evals, iidte_evals = [], []
            for name, loader, weights in evals:
                if name in ['env{}_out'.format(i) for i in args.test_envs]:
                    oodte_evals.append((name, loader, weights))
                elif 'out' in name:
                    iidte_evals.append((name, loader, weights))
            name = oodtr_evals[0]
            print_dbg('evaluating with adapt=True on {}...'.format(name), not PRODN)
            acc, ent = accuracy_ent_track(adapted_algorithm,
                                          oodtr_evals, oodte_evals, iidte_evals,
                                          device, args.ckpt_freq, adapt=True)
        results[name+'_acc'] = acc
        results[name+'_ent'] = ent

        del adapt_hparams['cached_loader']

        wandb.log(results)
        results_keys = sorted(results.keys())

        if results_keys != last_results_keys:
            misc.print_row(results_keys, colwidth=12)
            last_results_keys = results_keys
        misc.print_row([results[key] for key in results_keys],
            colwidth=12)

        results.update({
            'hparams': hparams,
            'args': vars(args)    
        })
        
        # ====================================================================
        # BUG FIX #9: Add oracle_exp_id to results JSON
        # ====================================================================
        # CRITICAL: Must add oracle_exp_id explicitly for pandas aggregation
        # oracle_exp_id is in args_in, not in args (which is from train results)
        # ====================================================================
        if hasattr(args_in, 'oracle_exp_id') and args_in.oracle_exp_id:
            results['oracle_exp_id'] = args_in.oracle_exp_id
        
        # ====================================================================
        # CRITICAL BUG FIX #4: Add hp_mode suffix to results filename
        # ====================================================================
        # ORIGINAL: All modes wrote to same file (results_ALG_METHOD.jsonl)
        # FIXED: Oracle/IID modes write to separate files with suffix
        #   - Oracle: results_PseudoLabel_LEASTDifflrLayer_orc.jsonl
        #   - IID:    results_PseudoLabel_LEASTDifflrLayer_iid.jsonl
        #   - Manual: results_PseudoLabel_LEASTDifflrLayer.jsonl
        # This prevents contamination between HP selection modes
        # ====================================================================
        if args.best_hp.lower() in ['orc', 'iid']:
            hp_suffix = f"_{args.best_hp.lower()}"
        else:
            hp_suffix = ""
        
        epochs_path = os.path.join(args.output_dir, 
                                  f'results_{alg_name}_{loc_alg_name}{hp_suffix}.jsonl')
        
        # ================================================================
        # CRITICAL BUG FIX #3: File mode logic
        # ================================================================
        # ORIGINAL: if hp_idx == 0 and not os.path.exists(epochs_path)
        # PROBLEM: If old file exists, uses append mode even for first config!
        # FIXED: Always overwrite for first config, append for subsequent
        # ================================================================
        if hp_idx == 0:
            mode = 'w'  # Always overwrite for first config (clears old data)
            print(f"[RESULTS] Creating/overwriting file: {os.path.basename(epochs_path)}")
        else:
            mode = 'a'  # Append for subsequent configs
            print(f"[RESULTS] Appending to file: {os.path.basename(epochs_path)}")
        
        # Write results
        with open(epochs_path, mode) as f:
            f.write(json.dumps(results, sort_keys=True) + "\n")

        wandb_run.finish()
        if not PRODN:
            break   # use for faster target evaluation sometimes

    # create done file
    with open(os.path.join(args.output_dir, 'done_{}_{}_{}'.format(alg_name, loc_alg_name, args.best_hp.lower())), 'w') as f:
        f.write('done')
    
    print(f"\n[COMPLETE] Created done marker: done_{alg_name}_{loc_alg_name}_{args.best_hp.lower()}")
    print(f"[COMPLETE] Results saved to: results_{alg_name}_{loc_alg_name}{hp_suffix}.jsonl")

    # Sleep for 10 seconds to prevent race condition crash
    if not PRODN:
        time.sleep(10)
