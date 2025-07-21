# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

import argparse
import copy
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed import command_launchers

import tqdm
import shlex

# TDB (Tiny DomainBed) environment mapping for hyperparameter ablation
TDB_ENVIRONMENTS = {
    'PACS': [1, 3],              # Cartoon, Sketch
    'VLCS': [2],                 # SUN09  
    'TerraIncognita': [0, 1, 2], # L100, L38, L43
}

def get_args_hash(train_args):
    tmp_train_args = {
        key:value for key, value in train_args.items() \
                    if key in [
                        'dataset', 'algorithm', 'test_envs',
                        'holdout_fraction', 'hparams_seed',
                        'task', 'trial_seed', 'seed'
                    ]
    }
    hparams_dict = eval(train_args['hparams'])
    hparams_args = {
        key:value for key, value in hparams_dict.items() \
                    if key in [
                        'backbone'
                    ]
    }
    tmp_train_args.update(hparams_args)
    args_str = json.dumps(tmp_train_args, sort_keys=True)
    args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
    return args_hash, args_str

def get_loc_alg_name(adapt_algorithm, find_algo, fix_algo, granularity):
    if adapt_algorithm in['T3A', 'TentPreBN', 'TentClf', 'PLClf']:
        loc_alg_name = "Nolocadapt"
    else:
        # Loc adapt, only for adapt_algorithms modifying the featurizer
        if find_algo == 'None' or fix_algo == 'None' or granularity == 'None':
            loc_alg_name = "Nolocadapt"
        else:
            loc_alg_name = find_algo + fix_algo + granularity
    return loc_alg_name

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_hash, args_str = get_args_hash(train_args)
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['OMP_NUM_THREADS=1', 'python', '-m', 'domainbed.scripts.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')


class SAJob:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    PRETRAINED = 'Pretrained'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir, ft_mode):
        args_hash, args_str = get_args_hash(train_args)
        self.output_dir = os.path.join(sweep_output_dir, args_hash)
        self.ft_mode = ft_mode
        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        #TODO: make this automatic!
        command = [
            'python', '-m', 'domainbed.scripts.supervised_adaptation',
            '--input_dir', self.train_args['output_dir'], 
            '--data_dir', self.train_args['data_dir'],
            '--ft_mode', ft_mode
        ]
        if 'wandb_name' in self.train_args:
            command.append('--wandb_name {}'.format(self.train_args['wandb_name']))
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            if os.path.exists(os.path.join(self.output_dir, 'done_{}'.format(ft_mode))):
                self.state = SAJob.DONE
            else:
                self.state = SAJob.PRETRAINED
        elif os.path.exists(os.path.join(self.output_dir, 'results_{}.jsonl'.format(ft_mode))):
            self.state = SAJob.INCOMPLETE
        else:
            self.state = SAJob.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'], self.ft_mode)
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')


class UAJob:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    PRETRAINED = 'Pretrained'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir, adapt_algorithm,
                 find_algo, fix_algo, granularity):
        args_hash, args_str = get_args_hash(train_args)
        self.output_dir = os.path.join(sweep_output_dir, args_hash)
        self.adapt_algorithm = adapt_algorithm
        self.find_algo = find_algo
        self.fix_algo = fix_algo
        self.granularity = granularity
        loc_alg_name = get_loc_alg_name(adapt_algorithm, find_algo, fix_algo, granularity)
        self.loc_alg_name = loc_alg_name
        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        #TODO: make this automatic!
        command = [
            'python', '-m', 'domainbed.scripts.unsupervised_adaptation',
            '--input_dir', self.train_args['output_dir'], 
            '--adapt_algorithm', adapt_algorithm,
            '--find_algo', find_algo,
            '--fix_algo', fix_algo,
            '--granularity', granularity,
            '--data_dir', self.train_args['data_dir']
        ]
        if 'wandb_name' in self.train_args:
            command.append('--wandb_name {}'.format(self.train_args['wandb_name']))
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            if os.path.exists(os.path.join(self.output_dir, 'done_{}_{}'.format(adapt_algorithm, loc_alg_name))):
                self.state = UAJob.DONE
            else:
                self.state = UAJob.PRETRAINED
        elif os.path.exists(os.path.join(self.output_dir, 'results_{}_{}.jsonl'.format(adapt_algorithm, loc_alg_name))):
            self.state = UAJob.INCOMPLETE
        else:
            self.state = UAJob.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'], self.adapt_algorithm,
            self.find_algo, self.fix_algo, self.granularity)
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')


def all_test_env_combinations(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs.
    """
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]

def make_args_list(n_trials_from, n_trials, dataset_names, algorithms, n_hparams_from, n_hparams, steps,
    data_dir, task, holdout_fraction, single_test_envs, hparams, wandb_name):
    args_list = []
    
    # Check for TDB mode via environment variable
    tdb_only = os.environ.get('TDB_ONLY', 'False').lower() == 'true'
    
    if tdb_only:
        print("TDB Mode enabled - will only process TDB environments")
        print(f"TDB environments: {TDB_ENVIRONMENTS}")
    
    for trial_seed in range(n_trials_from, n_trials_from+n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                # TDB filtering: skip datasets not in TDB list
                if tdb_only and dataset not in TDB_ENVIRONMENTS:
                    print(f"TDB Mode: Skipping {dataset} (not in TDB)")
                    continue
                
                if single_test_envs:
                    if tdb_only and dataset in TDB_ENVIRONMENTS:
                        all_test_envs = [[i] for i in TDB_ENVIRONMENTS[dataset]]
                        print(f"TDB Mode: Using environments {TDB_ENVIRONMENTS[dataset]} for {dataset}")
                    elif tdb_only:
                        continue  # This should not happen due to check above, but safety
                    else:
                        all_test_envs = [
                            [i] for i in range(datasets.num_environments(dataset))]
                else:
                    if tdb_only:
                        # For TDB mode with multiple test envs, use TDB environments
                        tdb_envs = TDB_ENVIRONMENTS[dataset]
                        all_test_envs = []
                        # Single environments
                        for i in tdb_envs:
                            all_test_envs.append([i])
                        # Pairs of environments
                        for i in range(len(tdb_envs)):
                            for j in range(i+1, len(tdb_envs)):
                                all_test_envs.append([tdb_envs[i], tdb_envs[j]])
                    else:
                        all_test_envs = all_test_env_combinations(
                            datasets.num_environments(dataset))
                        
                for test_envs in all_test_envs:
                    for hparams_seed in range(n_hparams_from, n_hparams):
                        train_args = {}
                        train_args['dataset'] = dataset
                        train_args['algorithm'] = algorithm
                        train_args['test_envs'] = test_envs
                        train_args['holdout_fraction'] = holdout_fraction
                        train_args['hparams_seed'] = hparams_seed
                        train_args['data_dir'] = data_dir
                        train_args['task'] = task
                        train_args['trial_seed'] = trial_seed
                        train_args['seed'] = misc.seed_hash(dataset,
                            algorithm, test_envs, hparams_seed, trial_seed)
                        if steps is not None:
                            train_args['steps'] = steps
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        if wandb_name is not None:
                            test_env_name =  "te" + "".join([str(env) for env in test_envs])
                            train_args['wandb_name'] = test_env_name + "_" + wandb_name
                        args_list.append(train_args)
    
    if tdb_only:
        print(f"TDB Mode: Generated {len(args_list)} jobs total")
    
    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=[
        'launch', 'delete_incomplete', 'supervised_adaptation', 
        'unsupervised_adaptation', 'unsup_adapt'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--task', type=str, default="domain_generalization")
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials_from', type=int, default=3)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--single_test_envs', action='store_true')
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--wandb_name', type=str, default=None)
    # for unsup_adapt
    parser.add_argument('--adapt_algorithms', nargs='+', type=str, default=[""])
    parser.add_argument('--find_algos', nargs='+', type=str, default=[""])
    parser.add_argument('--fix_algos', nargs='+', type=str, default=[""])
    parser.add_argument('--granularities', nargs='+', type=str, default=[""])
    # for sup_adapt
    parser.add_argument('--ft_modes', type=str, nargs='+', default=[""])
    args = parser.parse_args()

    args_list = make_args_list(
        n_trials_from=args.n_trials_from, 
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        data_dir=args.data_dir,
        task=args.task,
        holdout_fraction=args.holdout_fraction,
        single_test_envs=args.single_test_envs,
        hparams=args.hparams,
        wandb_name=args.wandb_name
    )

    if args.command in ['launch', 'delete_incomplete']:
        jobs = [Job(train_args, args.output_dir) for train_args in args_list]

        for job in jobs:
            print(job)
        print("{} jobs: {} done, {} incomplete, {} not launched.".format(
            len(jobs),
            len([j for j in jobs if j.state == Job.DONE]),
            len([j for j in jobs if j.state == Job.INCOMPLETE]),
            len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
        )

        if args.command == 'launch':
            to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
            print(f'About to launch {len(to_launch)} jobs.')
            if not args.skip_confirmation:
                ask_for_confirmation()
            launcher_fn = command_launchers.REGISTRY[args.command_launcher]
            Job.launch(to_launch, launcher_fn)

        elif args.command == 'delete_incomplete':
            to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
            print(f'About to delete {len(to_delete)} jobs.')
            if not args.skip_confirmation:
                ask_for_confirmation()
            Job.delete(to_delete)

    elif args.command == 'supervised_adaptation':
        if not args.ft_modes[0]:
            ft_modes = [
                'clf', 'token', 'transformer', 'all'
            ]
        else:
            ft_modes = args.ft_modes

        jobs = []
        for ft_mode in ft_modes:
            jobs += [SAJob(
                train_args, args.output_dir,
                ft_mode=ft_mode) for train_args in args_list]

        for job in jobs:
            print(job)
        print("{} jobs: {} done, {} pretrained, {} incomplete, {} not launched.".format(
            len(jobs),
            len([j for j in jobs if j.state == SAJob.DONE]),
            len([j for j in jobs if j.state == SAJob.PRETRAINED]),
            len([j for j in jobs if j.state == SAJob.INCOMPLETE]),
            len([j for j in jobs if j.state == SAJob.NOT_LAUNCHED]))
        )

        to_launch = [j for j in jobs if j.state == SAJob.PRETRAINED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation() 
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command in ['unsupervised_adaptation', 'unsup_adapt']:
        if not args.adapt_algorithms[0]:
            methods = [
                'T3A', 'TentFull', 'TentNorm', 'TentPreBN','TentClf', 
                'PseudoLabel', 'PLClf', 'SHOT', 'SHOTIM'
                ]
        else:
            methods = args.adapt_algorithms

        if not args.find_algos[0]:
            find_methods = [
                'None', 'LEAST',
            ]
        else:
            find_methods = args.find_algos
        if not args.fix_algos[0]:
            fix_methods = [
                'None', 'Difflr',
            ]
        else:
            fix_methods = args.fix_algos
        if not args.granularities[0]:
            granularity_methods = [
                'None', 'Block', 'Subblock', 'Layer',
            ]
        else:
            granularity_methods = args.granularities

        jobs = []
        for method in methods:
            # for batch size: 32, 8, 200
            #TODO: uncomment for batch size 8, 200
            for find_method in find_methods:
                for fix_method in fix_methods:
                    for granularity_method in granularity_methods:
                        # skip if only one of them is None
                        # if (find_method == 'None' or fix_method == 'None' or granularity_method == 'None') and \
                        #     (find_method != fix_method or find_method != granularity_method):
                        #     continue
                        
                        # skip loc adapt for adapt_algorithms not modifying the featurizer
                        if method in ['T3A', 'TentPreBN', 'TentClf', 'PLClf'] and find_method != 'None':
                            continue
                        print(method, find_method, fix_method, granularity_method)

                        jobs += [UAJob(
                            train_args, args.output_dir,
                            adapt_algorithm=method,
                            find_algo=find_method,
                            fix_algo=fix_method,
                            granularity=granularity_method)
                            for train_args in args_list]
                        # jobs += [UAJob(
                        #     train_args, args.output_dir,
                        #     adapt_algorithm='{}-{}'.format(method, '8'),
                        #     find_algo=find_method,
                        #     fix_algo=fix_method,
                        #     granularity=granularity_method)
                        #     for train_args in args_list]
                        # jobs += [UAJob(
                        #     train_args, args.output_dir,
                        #     adapt_algorithm='{}-{}'.format(method, '200'),
                        #     find_algo=find_method,
                        #     fix_algo=fix_method,
                        #     granularity=granularity_method)
                        #     for train_args in args_list]

        for job in jobs:
            print(job)
        print("{} jobs: {} done, {} pretrained, {} incomplete, {} not launched.".format(
            len(jobs),
            len([j for j in jobs if j.state == UAJob.DONE]),
            len([j for j in jobs if j.state == UAJob.PRETRAINED]),
            len([j for j in jobs if j.state == UAJob.INCOMPLETE]),
            len([j for j in jobs if j.state == UAJob.NOT_LAUNCHED]))
        )

        to_launch = [j for j in jobs if j.state == UAJob.PRETRAINED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation() 
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)
