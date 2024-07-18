# The code is modified from domainbed.scripts.train

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
import PIL
import torch
import torchvision
import torch.utils.data
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


def get_best_hp(evals_list, hparams_list, hp_idx):
        best_acc = -1
        best_hparams = {}

        track_path = os.path.join(args.output_dir, 'hpSearch_{}_{}_ord{}_{}.jsonl'.format(alg_name, loc_alg_name, CONST_ORDER, hp_idx))
        if os.path.exists(track_path) and RESUSE_BEST_HP:
            adapt_hparams_keys = list(hparams_list[0].keys()) + ['hp_acc']
            with open(track_path, 'r') as f:
                for line in f:
                    # Read the hparams and acc into a dict
                    hparams_results = json.loads(line[:-1])
                    
                    # Check if all the necessary keys are in the hparams
                    assert all([k in hparams_results.keys() for k in adapt_hparams_keys])
                    
                    # Get the necessary hparams and acc
                    adapt_hparams = {k: v for k, v in hparams_results.items() if k in adapt_hparams_keys}
                    mean_acc = hparams_results['hp_acc']

                    # Save the best hparams and acc
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        best_hparams = copy.deepcopy(adapt_hparams)

            print("Reusing best hp from {}".format(track_path))
            return best_hparams, best_acc

        assert len(evals_list) > 0
        assert len(hparams_list) > 0

        for adapt_hparams in tqdm(hparams_list, desc='get_best_hp', disable=PRODN):
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
                # if hp_idx == 'Man':
                #     # For measuring the effect of sample complexity on powerset results
                #     acc, ent, acc_dict = accuracy_ent_interim(adapted_algorithm, loader, weights, device, 5, adapt=True)
                #     low_data_dict = {name + '_step' + str(k) + '_acc' :v for k, v in acc_dict.items()}
                #     hparams_results.update(low_data_dict)
                # else:
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
            
            # print(adapt_hparams, score, mean_acc)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_hparams = copy.deepcopy(adapt_hparams)
            with open(track_path, 'a') as f:
                f.write(json.dumps(hparams_results, sort_keys=True) + "\n")
        return best_hparams, best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--adapt_algorithm', type=str, default="T3A")
    parser.add_argument('--find_algo', type=str, default="None")
    parser.add_argument('--fix_algo', type=str, default="None",
                        choices=['None', 'Trainsubset', 'Difflr'])
    parser.add_argument('--granularity', type=str, default="None",
                        choices=['None', 'Block', 'Subblock', 'Layer'],)
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--best_hp', type=str, default='True')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--adapt_hparams_dict', type=str, default="{}",
        help='JSON-serialized adaptation hparams dict')
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
        loc_alg_name = args.find_algo + args.fix_algo + args.granularity
        loc_dict = {
            'find_algo': [args.find_algo],
            'fix_algo': [args.fix_algo],
            'granularity': [args.granularity],
        }
    if 'least' in args.find_algo.lower():
        # Get the td variant and store as args
        td_variant = least.get_td_variant(args.find_algo)
        for k, v in td_variant.items():
            setattr(args, "LEAST_"+k, v)
        
    if os.path.exists(os.path.join(args.output_dir, 'done_{}_{}'.format(alg_name, loc_alg_name))):
        print("{} with {} has already excecuted".format(alg_name, loc_alg_name))

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
    if CONST_ORDER:
        print("Get all train loaders")
        train_loaders = [list(FastDataLoader(
            dataset=env,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS))
            for i, (env, env_weights) in enumerate(out_splits)
            if i in args.test_envs]
    else:
        train_loaders = [FastDataLoader(
            dataset=env,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(out_splits)
            if i in args.test_envs]
    
    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    if CONST_ORDER:
        print("Get all eval loaders")
        eval_loaders = [list(FastDataLoader(
            dataset=env,
            batch_size=args.test_batch_size,
            num_workers=dataset.N_WORKERS))
            for env, _ in tqdm((in_splits + out_splits + uda_splits), desc="eval_loaders", disable=PRODN)]
    else:
        eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=args.test_batch_size,
            num_workers=dataset.N_WORKERS)
            for env, _ in (in_splits + out_splits + uda_splits)]
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

    # Evaluate base model
    # print("Base model's results")
    # results = {}
    # evals = zip(eval_loader_names, eval_loaders, eval_weights)
    # for name, loader, weights in evals:
    #     print('evaluating with adapt=None on {}...'.format(name))
    #     acc, ent = accuracy_ent(algorithm, loader, weights, device, adapt=None)
    #     results[name+'_acc'] = acc
    #     results[name+'_ent'] = ent
    # results_keys = sorted(results.keys())
    # misc.print_row(results_keys, colwidth=12)
    # misc.print_row([results[key] for key in results_keys], colwidth=12)

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

    # Perform Model selection (for reducing computation overload and wandb logging)
    #TODO: implement leave-one-out model selection
    #TODO: Remove the second condition below
    if args.best_hp.lower() in ['true', 'iid', 'orc', 'man'] and 'domaingen' not in args.find_algo.lower():
        print("Get best HParams:")
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
        
        new_adapt_hparams_list = []
        print("Best Acc on each split:")
        if args.best_hp.lower() in ['true', 'iid']:
            iid_hparams, iid_acc = get_best_hp(iid_evals, adapt_hparams_list, "Iid")
            print("\tIID: {:.2f}".format(iid_acc))
            iid_hparams['hp_mode'] = "Iid"
            new_adapt_hparams_list.append(iid_hparams)

        if args.best_hp.lower() in ['true', 'orc']:
            oracle_hparams, oracle_acc = get_best_hp(oracle_evals, adapt_hparams_list, "Orc")
            print("\tOrc: {:.2f}".format(oracle_acc))
            oracle_hparams['hp_mode'] = "Orc"
            new_adapt_hparams_list.append(oracle_hparams)
        
        if args.best_hp.lower() in ['true', 'man']:
            manual_hparams, manual_acc = get_best_hp(manual_evals, adapt_hparams_list, "Man")
            print("\tMan: {:.2f}".format(manual_acc))
            manual_hparams['hp_mode'] = "Man"
            new_adapt_hparams_list.append(manual_hparams)

        adapt_hparams_list = new_adapt_hparams_list

        tmp_run.finish()

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

        # ## Usual evaluation
        for name, loader, weights in evals:
            if not PRODN:
                break           # use for faster target evaluation sometimes
            print_dbg('evaluating with adapt=True on {}...'.format(name), not PRODN)
            acc, ent = accuracy_ent(adapted_algorithm, loader, weights, device, adapt=True)
            results[name+'_acc'] = acc
            results[name+'_ent'] = ent
            adapted_algorithm.reset()

        name, loader, weights = train_loader
        print_dbg('evaluating with adapt=True on {}...'.format(name), not PRODN)
        acc, ent = accuracy_ent(adapted_algorithm, loader, weights, device, adapt=True)
        results[name+'_acc'] = acc
        results[name+'_ent'] = ent

        del adapt_hparams['cached_loader']
        # if 'src_info_dict' in results.keys():
        #     del results['src_info_dict']
        # if 'stats_path' in results.keys():
        #     del results['stats_path']
        # if 'grad_mask' in results.keys():
        #     del results['grad_mask']
        # if 'mean_grad_mask' in results.keys():
        #     del results['mean_grad_mask']

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
        # save file
        epochs_path = os.path.join(args.output_dir, 'results_{}_{}.jsonl'.format(alg_name, loc_alg_name))
        with open(epochs_path, 'a') as f:
            f.write(json.dumps(results, sort_keys=True) + "\n")

        wandb_run.finish()
        if not PRODN:
            break   # use for faster target evaluation sometimes

    # create done file
    with open(os.path.join(args.output_dir, 'done_{}_{}'.format(alg_name, loc_alg_name)), 'w') as f:
        f.write('done')

    # Sleep for 10 seconds to prevent race condition crash
    if not PRODN:
        time.sleep(10)
