#!/usr/bin/env python3
"""
Oracle Hyperparameter Selection Results Aggregation Script
===========================================================

Aggregates oracle HP selection results for LEAST and LLS methods.

Key Features:
- Per-(environment, mark) granular data with selected configs
- Separate PL and SHOT columns
- TDB only computed for Tiny DomainBed mode
- Seed-based variance statistics
- Multiple output files grouped by oracle_exp_id
- Optional filtering by specific exp_ids

Usage:
  python pandas_oracle_results.py \
    --input_dir /path/to/sweep \
    --sweep_names PACS_resnet18-BN_ERM_0 ... \
    --output_prefix results_oracle \
    --hp_mode Both \
    --tdb_only (or --no_tdb_only) \
    --exp_ids 5Feb_Run_1 5Feb_Run_2  # Optional: process only these exp_ids
"""

import argparse
import json
import os
import glob
from collections import defaultdict, OrderedDict, Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback

# Tiny DomainBed environments (6 total)
TDB_ENVIRONMENTS = {
    'PACS': [1, 3],
    'VLCS': [2],
    'TerraIncognita': [0, 1, 2],
}

# Full DomainBed environments (16 total)
FULL_DB_ENVIRONMENTS = {
    'PACS': [0, 1, 2, 3],
    'VLCS': [0, 1, 2, 3],
    'TerraIncognita': [0, 1, 2, 3],
    'OfficeHome': [0, 1, 2, 3],
}

def get_environment_list(tdb_only=True):
    """Get list of environments based on mode."""
    env_dict = TDB_ENVIRONMENTS if tdb_only else FULL_DB_ENVIRONMENTS
    env_list = []
    for dataset, envs in env_dict.items():
        for env in envs:
            env_list.append((dataset, env))
    return env_list, env_dict

def normalize_algo_name(adapt_algo):
    """Normalize TTA algorithm name to standard form."""
    if adapt_algo in ['PseudoLabel', 'PLClf']:
        return 'PL'
    elif adapt_algo in ['SHOT', 'SHOTIM']:
        return 'SHOT'
    else:
        return adapt_algo

def extract_method_from_filename(filename):
    """Extract method (LEAST or LLS) from filename."""
    if 'LLS' in filename:
        return 'LLS'
    elif 'LEAST' in filename:
        return 'LEAST'
    else:
        return None

def find_all_oracle_files(base_dirs, hp_mode_filter=None, tdb_only=True):
    """Find all oracle result files."""
    all_files = []
    
    print("\n" + "="*80)
    print("DISCOVERING ORACLE RESULT FILES")
    print("="*80)
    print(f"HP mode filter: {hp_mode_filter}")
    print(f"TDB only: {tdb_only}")
    
    if tdb_only:
        valid_datasets = set(TDB_ENVIRONMENTS.keys())
    else:
        valid_datasets = set(FULL_DB_ENVIRONMENTS.keys())
    
    for base_dir in base_dirs:
        base_dir_name = os.path.basename(base_dir)
        parts = base_dir_name.split('_')
        dataset = parts[0]
        
        try:
            mark = int(parts[-1])
        except (ValueError, IndexError):
            print(f"[WARNING] Cannot extract mark from {base_dir_name}, skipping")
            continue
        
        print(f"\n[SCAN] Base directory: {base_dir}")
        print(f"[SCAN] Dataset: {dataset}, Mark: {mark}")
        
        if dataset not in valid_datasets:
            print(f"[SKIP] Dataset {dataset} not in target list")
            continue
        
        if not os.path.exists(base_dir):
            print(f"[WARNING] Directory does not exist: {base_dir}")
            continue
        
        hash_dirs = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and len(item) == 32:
                try:
                    int(item, 16)
                    hash_dirs.append(item)
                except ValueError:
                    continue
        
        print(f"[SCAN] Found {len(hash_dirs)} hash directories")
        
        for hash_dir in sorted(hash_dirs):
            hash_path = os.path.join(base_dir, hash_dir)
            
            patterns = [
                'results_*_LEAST*.jsonl',
                'results_*_LLS*.jsonl',
            ]
            
            found_files = set()
            for pattern in patterns:
                matches = glob.glob(os.path.join(hash_path, pattern))
                found_files.update(matches)
            
            found_files = [f for f in found_files if 'hpSearch' not in f and not f.endswith('done')]
            
            for result_file in found_files:
                filename = os.path.basename(result_file)
                
                parts_fn = filename.replace('.jsonl', '').split('_')
                if len(parts_fn) < 3:
                    continue
                
                adapt_algo = parts_fn[1]
                method = extract_method_from_filename(filename)
                
                if method not in ['LEAST', 'LLS']:
                    continue
                
                file_info = {
                    'dataset': dataset,
                    'mark': mark,
                    'file_path': result_file,
                    'filename': filename,
                    'hash_dir': hash_dir,
                    'adapt_algo': adapt_algo,
                    'method': method,
                }
                
                all_files.append(file_info)
    
    print(f"\n[SCAN] Total oracle files found: {len(all_files)}")
    
    method_counts = defaultdict(int)
    for f in all_files:
        key = (f['dataset'], f['method'], f['adapt_algo'], f['mark'])
        method_counts[key] += 1
    
    print("\n[SUMMARY] Files by dataset/method/algorithm/mark:")
    for (dataset, method, algo, mark), count in sorted(method_counts.items()):
        print(f"  {dataset} / {method} / {algo} / mark{mark}: {count} files")
    
    return all_files

def extract_oracle_results(file_info, hp_mode_filter=None, env_dict=None):
    """Extract results from oracle file."""
    results = []
    file_path = file_info['file_path']
    dataset = file_info['dataset']
    
    valid_envs = env_dict.get(dataset, []) if env_dict else None
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"[WARNING] Empty file: {file_path}")
            return results
        
        for line_num, line in enumerate(lines):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                args = data.get('args', {})
                
                test_envs = args.get('test_envs', [])
                if not test_envs:
                    print(f"[WARNING] No test_envs in {file_path} line {line_num}")
                    continue
                test_env = test_envs[0] if isinstance(test_envs, list) else test_envs
                
                if valid_envs is not None and test_env not in valid_envs:
                    continue
                
                hp_mode = data.get('hp_mode', 'Man')
                
                if hp_mode_filter and hp_mode_filter != 'Both':
                    if hp_mode != hp_mode_filter:
                        continue
                elif hp_mode_filter == 'Both':
                    if hp_mode not in ['Orc', 'Iid']:
                        continue
                
                selected_config = data.get('find_algo', '')
                all_configs = args.get('find_algo', '')
                
                alpha = data.get('alpha', None)
                beta = data.get('beta', None)
                gamma = data.get('gamma', None)
                
                oracle_exp_id = data.get('oracle_exp_id', None)
                
                acc_key_in = f'env{test_env}_in_acc'
                acc_key_out = f'env{test_env}_out_acc'
                
                main_acc = data.get(acc_key_in, None)
                oracle_acc = data.get(acc_key_out, None)
                
                if main_acc is None:
                    print(f"[WARNING] No {acc_key_in} in {file_path} line {line_num}")
                    continue
                
                result_dict = {
                    'dataset': dataset,
                    'test_env': test_env,
                    'mark': file_info['mark'],
                    'method': file_info['method'],
                    'adapt_algo': file_info['adapt_algo'],
                    'hp_mode': hp_mode,
                    'oracle_exp_id': oracle_exp_id,
                    'selected_config': selected_config,
                    'all_tested_configs': all_configs,
                    'main_acc': float(main_acc),
                    'oracle_acc': float(oracle_acc) if oracle_acc else None,
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'source_file': file_info['filename'],
                    'hash_dir': file_info['hash_dir'],
                }
                
                results.append(result_dict)
                
            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid JSON in {file_path} line {line_num}: {e}")
            except Exception as e:
                print(f"[ERROR] Processing {file_path} line {line_num}: {e}")
                traceback.print_exc()
    
    except Exception as e:
        print(f"[ERROR] Cannot read file {file_path}: {e}")
    
    return results

def aggregate_oracle_results(all_results, env_list, expected_n_envs=None):
    """Aggregate oracle results using seed-based statistics."""
    print("\n" + "="*80)
    print("AGGREGATING ORACLE RESULTS (SEED-BASED STATISTICS)")
    print("="*80)
    
    groups = defaultdict(lambda: {
        'mark_to_env_accs': defaultdict(dict),
        'env_mark_to_selected_config': {},
        'env_mark_to_oracle_acc': {},
        'env_mark_to_source': {},
    })
    
    for result in all_results:
        group_key = (result['method'], result['hp_mode'], result['adapt_algo'])
        mark = result['mark']
        env_key = (result['dataset'], result['test_env'])
        
        groups[group_key]['mark_to_env_accs'][mark][env_key] = result['main_acc']
        
        env_mark_key = (env_key, mark)
        groups[group_key]['env_mark_to_selected_config'][env_mark_key] = result['selected_config']
        groups[group_key]['env_mark_to_oracle_acc'][env_mark_key] = result['oracle_acc']
        groups[group_key]['env_mark_to_source'][env_mark_key] = result['source_file']
    
    print(f"\n[AGGREGATE] Created {len(groups)} groups")
    
    aggregated = []
    
    for group_key, data in groups.items():
        method, hp_mode, adapt_algo = group_key
        
        tta_algo = normalize_algo_name(adapt_algo)
        
        per_mark_means = []
        marks = sorted(data['mark_to_env_accs'].keys())
        
        for mark in marks:
            env_accs = list(data['mark_to_env_accs'][mark].values())
            mark_mean = np.mean(env_accs)
            per_mark_means.append(mark_mean)
            
            n_envs_this_mark = len(env_accs)
            if expected_n_envs is not None and n_envs_this_mark != expected_n_envs:
                print(f"  [WARNING] {method}/{hp_mode}/{tta_algo} mark{mark}: "
                      f"{n_envs_this_mark} envs != {expected_n_envs} (expected)")
        
        avg_acc = np.mean(per_mark_means)
        std_acc = np.std(per_mark_means, ddof=1) if len(per_mark_means) > 1 else 0.0
        n_marks = len(per_mark_means)
        
        env_mark_to_accuracy = {}
        for mark, env_dict in data['mark_to_env_accs'].items():
            for env_key, acc in env_dict.items():
                env_mark_to_accuracy[(env_key, mark)] = acc
        
        configs = list(data['env_mark_to_selected_config'].values())
        config_counts = Counter(configs)
        config_summary = ", ".join([f"{cfg} ({cnt})" for cfg, cnt in sorted(config_counts.items(), key=lambda x: -x[1])])
        
        result = {
            'method': method,
            'hp_mode': hp_mode,
            'tta_algo': tta_algo,
            'avg_acc': avg_acc,
            'std_acc': std_acc,
            'n_marks': n_marks,
            'per_mark_means': per_mark_means,
            'env_mark_to_accuracy': env_mark_to_accuracy,
            'env_mark_to_config': data['env_mark_to_selected_config'],
            'selected_configs_summary': config_summary,
        }
        
        aggregated.append(result)
        
        print(f"\n[GROUP] {method} / {hp_mode} / {tta_algo}")
        print(f"  Per-mark means: {[f'{m:.4f}' for m in per_mark_means]}")
        print(f"  Grand mean: {avg_acc:.4f}")
        print(f"  Std across marks: {std_acc:.4f}")
        print(f"  N marks: {n_marks}")
        print(f"  Selected configs: {config_summary}")
    
    return aggregated

def create_unified_results(aggregated_results, env_list, tdb_only=True):
    """Create unified PL, SHOT, TDB results."""
    print("\n" + "="*80)
    print("CREATING UNIFIED PL+SHOT+TDB RESULTS")
    print("="*80)
    print(f"TDB computation: {'ENABLED' if tdb_only else 'DISABLED'}")
    
    method_hp_groups = defaultdict(lambda: {'PL': None, 'SHOT': None})
    
    for result in aggregated_results:
        key = (result['method'], result['hp_mode'])
        tta = result['tta_algo']
        method_hp_groups[key][tta] = result
    
    unified_results = []
    
    for (method, hp_mode), data in method_hp_groups.items():
        pl_data = data['PL']
        shot_data = data['SHOT']
        
        has_pl = pl_data is not None
        has_shot = shot_data is not None
        
        print(f"\n[UNIFIED] {method} / {hp_mode}")
        print(f"  Has PL: {has_pl}")
        print(f"  Has SHOT: {has_shot}")
        
        if not (has_pl and has_shot):
            print(f"  [SKIP] Missing {'PL' if not has_pl else 'SHOT'}")
            continue
        
        unified = {
            'method': method,
            'hp_mode': hp_mode,
            'pl_avg': pl_data['avg_acc'],
            'pl_std': pl_data['std_acc'],
            'pl_n_marks': pl_data['n_marks'],
            'pl_env_mark_to_accuracy': pl_data['env_mark_to_accuracy'],
            'pl_env_mark_to_config': pl_data['env_mark_to_config'],
            'pl_selected_configs': pl_data['selected_configs_summary'],
            'shot_avg': shot_data['avg_acc'],
            'shot_std': shot_data['std_acc'],
            'shot_n_marks': shot_data['n_marks'],
            'shot_env_mark_to_accuracy': shot_data['env_mark_to_accuracy'],
            'shot_env_mark_to_config': shot_data['env_mark_to_config'],
            'shot_selected_configs': shot_data['selected_configs_summary'],
        }
        
        if tdb_only:
            pl_marks = pl_data['per_mark_means']
            shot_marks = shot_data['per_mark_means']
            
            if len(pl_marks) != len(shot_marks):
                print(f"  [ERROR] PL has {len(pl_marks)} marks but SHOT has {len(shot_marks)} marks")
                continue
            
            tdb_marks = [(pl + shot) / 2.0 for pl, shot in zip(pl_marks, shot_marks)]
            
            unified['tdb_avg'] = np.mean(tdb_marks)
            unified['tdb_std'] = np.std(tdb_marks, ddof=1) if len(tdb_marks) > 1 else 0.0
            unified['tdb_n_marks'] = len(tdb_marks)
            
            print(f"  PL:   {pl_data['avg_acc']:.4f} +/- {pl_data['std_acc']:.4f}")
            print(f"  SHOT: {shot_data['avg_acc']:.4f} +/- {shot_data['std_acc']:.4f}")
            print(f"  TDB:  {unified['tdb_avg']:.4f} +/- {unified['tdb_std']:.4f}")
        else:
            unified['tdb_avg'] = None
            unified['tdb_std'] = None
            unified['tdb_n_marks'] = None
            
            print(f"  PL:   {pl_data['avg_acc']:.4f} +/- {pl_data['std_acc']:.4f}")
            print(f"  SHOT: {shot_data['avg_acc']:.4f} +/- {shot_data['std_acc']:.4f}")
            print(f"  TDB:  NOT COMPUTED (Full DomainBed mode)")
        
        unified_results.append(unified)
    
    return unified_results

def create_dataframe(unified_results, env_list, tdb_only=True):
    """Create pandas DataFrame from unified results."""
    print("\n" + "="*80)
    print("CREATING OUTPUT DATAFRAME")
    print("="*80)
    
    all_marks = set()
    for result in unified_results:
        for (env_key, mark) in result['pl_env_mark_to_accuracy'].keys():
            all_marks.add(mark)
    marks = sorted(all_marks)
    
    print(f"Marks found in data: {marks}")
    
    rows = []
    
    for result in unified_results:
        row = OrderedDict([
            ('method', result['method']),
            ('hp_mode', result['hp_mode']),
            ('pl_performance', result['pl_avg']),
            ('pl_std', result['pl_std']),
            ('pl_n_marks', result['pl_n_marks']),
            ('pl_selected_configs', result['pl_selected_configs']),
            ('shot_performance', result['shot_avg']),
            ('shot_std', result['shot_std']),
            ('shot_n_marks', result['shot_n_marks']),
            ('shot_selected_configs', result['shot_selected_configs']),
        ])
        
        if tdb_only and result['tdb_avg'] is not None:
            row['tdb_performance'] = result['tdb_avg']
            row['tdb_std'] = result['tdb_std']
            row['tdb_n_marks'] = result['tdb_n_marks']
        
        for dataset, env in env_list:
            env_key = (dataset, env)
            for mark in marks:
                env_mark_key = (env_key, mark)
                prefix = f'{dataset.lower()}_env{env}_mark{mark}'
                
                pl_acc = result['pl_env_mark_to_accuracy'].get(env_mark_key, np.nan)
                pl_config = result['pl_env_mark_to_config'].get(env_mark_key, '')
                row[f'{prefix}_pl_acc'] = pl_acc
                row[f'{prefix}_pl_config'] = pl_config
                
                shot_acc = result['shot_env_mark_to_accuracy'].get(env_mark_key, np.nan)
                shot_config = result['shot_env_mark_to_config'].get(env_mark_key, '')
                row[f'{prefix}_shot_acc'] = shot_acc
                row[f'{prefix}_shot_config'] = shot_config
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        df = df.sort_values(['method', 'hp_mode'])
    
    print(f"[DATAFRAME] Created with {len(df)} rows")
    print(f"[DATAFRAME] Total columns: {len(df.columns)}")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate Oracle HP Selection Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Base directory containing sweep results')
    parser.add_argument('--sweep_names', nargs='+', type=str, required=True,
                        help='Names of sweep directories to process (including all marks)')
    parser.add_argument('--output_prefix', type=str, default='oracle_results',
                        help='Prefix for output CSV file')
    parser.add_argument('--hp_mode', type=str, required=False, choices=['Orc', 'Iid', 'Both'], default='Both',
                        help='HP selection mode to aggregate')
    parser.add_argument('--tdb_only', action='store_true', default=True,
                        help='Only process Tiny DomainBed environments (default: True)')
    parser.add_argument('--no_tdb_only', action='store_false', dest='tdb_only',
                        help='Process all DomainBed environments')
    parser.add_argument('--exp_ids', nargs='*', default=None,
                        help='Optional: Only process specific experiment IDs. Use "none" for results without exp_id. If not specified, processes all exp_ids found.')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ORACLE HYPERPARAMETER SELECTION RESULTS AGGREGATION")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Sweep names: {args.sweep_names}")
    print(f"Output prefix: {args.output_prefix}")
    print(f"HP mode: {args.hp_mode}")
    print(f"TDB only: {args.tdb_only}")
    if args.exp_ids is not None:
        print(f"Filter exp_ids: {args.exp_ids}")
    else:
        print(f"Filter exp_ids: None (process all)")
    
    env_list, env_dict = get_environment_list(tdb_only=args.tdb_only)
    expected_n_envs = len(env_list)
    
    print(f"\nEnvironments to process: {expected_n_envs}")
    for dataset, env in env_list:
        print(f"  {dataset} env{env}")
    
    sweep_dirs = [os.path.join(args.input_dir, name) for name in args.sweep_names]
    
    missing_dirs = [d for d in sweep_dirs if not os.path.exists(d)]
    if missing_dirs:
        print("\n[ERROR] Missing directories:")
        for d in missing_dirs:
            print(f"  {d}")
        return 1
    
    all_files = find_all_oracle_files(sweep_dirs, hp_mode_filter=args.hp_mode, tdb_only=args.tdb_only)
    
    if not all_files:
        print("\n[ERROR] No oracle result files found!")
        return 1
    
    print("\n" + "="*80)
    print("EXTRACTING RESULTS FROM FILES")
    print("="*80)
    
    all_results = []
    for file_info in tqdm(all_files, desc="Processing files"):
        results = extract_oracle_results(file_info, hp_mode_filter=args.hp_mode, env_dict=env_dict)
        all_results.extend(results)
    
    print(f"\n[EXTRACT] Total results extracted: {len(all_results)}")
    
    if not all_results:
        print("\n[ERROR] No results extracted!")
        return 1
    
    method_counts = defaultdict(int)
    for r in all_results:
        key = (r['dataset'], r['method'], r['adapt_algo'], r['mark'])
        method_counts[key] += 1
    
    print("\n[SUMMARY] Results by dataset/method/algorithm/mark:")
    for (dataset, method, algo, mark), count in sorted(method_counts.items()):
        print(f"  {dataset} / {method} / {algo} / mark{mark}: {count} results")
    
    results_by_exp_id = defaultdict(list)
    for result in all_results:
        exp_id = result.get('oracle_exp_id', None)
        results_by_exp_id[exp_id].append(result)
    
    print(f"\n[DISCOVERY] Found {len(results_by_exp_id)} unique experiment ID(s):")
    for exp_id in sorted(results_by_exp_id.keys(), key=lambda x: (x is None, x)):
        count = len(results_by_exp_id[exp_id])
        exp_id_display = exp_id if exp_id else "NO_EXP_ID"
        print(f"  - {exp_id_display}: {count} results")
    
    if args.exp_ids is not None:
        print("\n[FILTER] Filtering by requested exp_ids...")
        
        requested_exp_ids = set()
        for exp_id_str in args.exp_ids:
            if exp_id_str.lower() in ['none', 'no_exp_id']:
                requested_exp_ids.add(None)
            else:
                requested_exp_ids.add(exp_id_str)
        
        print(f"[FILTER] Requested exp_ids: {[e if e else 'NO_EXP_ID' for e in requested_exp_ids]}")
        
        filtered_results_by_exp_id = {k: v for k, v in results_by_exp_id.items() if k in requested_exp_ids}
        
        found_exp_ids = set(filtered_results_by_exp_id.keys())
        missing_exp_ids = requested_exp_ids - found_exp_ids
        
        if missing_exp_ids:
            print(f"[FILTER] WARNING: Requested exp_ids not found in data:")
            for missing_id in missing_exp_ids:
                display_id = missing_id if missing_id else 'NO_EXP_ID'
                print(f"  - {display_id}")
        
        if not filtered_results_by_exp_id:
            print("\n[ERROR] No results remain after filtering by exp_ids!")
            print(f"[ERROR] Requested: {[e if e else 'NO_EXP_ID' for e in requested_exp_ids]}")
            print(f"[ERROR] Available: {[e if e else 'NO_EXP_ID' for e in results_by_exp_id.keys()]}")
            return 1
        
        results_by_exp_id = filtered_results_by_exp_id
        print(f"[FILTER] After filtering: {len(results_by_exp_id)} exp_id(s) remain")
    
    output_files_created = []
    
    for exp_id, exp_results in sorted(results_by_exp_id.items(), key=lambda x: (x[0] is None, x[0])):
        exp_id_display = exp_id if exp_id else "NO_EXP_ID"
        
        print("\n" + "="*80)
        print(f"PROCESSING EXPERIMENT ID: {exp_id_display}")
        print("="*80)
        print(f"Results count: {len(exp_results)}")
        
        aggregated_results = aggregate_oracle_results(
            exp_results, env_list, expected_n_envs=expected_n_envs
        )
        
        if not aggregated_results:
            print(f"\n[WARNING] No aggregated results for exp_id: {exp_id_display}")
            continue
        
        unified_results = create_unified_results(
            aggregated_results, env_list, tdb_only=args.tdb_only
        )
        
        if not unified_results:
            print(f"\n[WARNING] No unified results for exp_id: {exp_id_display}")
            continue
        
        df = create_dataframe(unified_results, env_list, tdb_only=args.tdb_only)
        
        if df.empty:
            print(f"\n[WARNING] Empty DataFrame for exp_id: {exp_id_display}")
            continue
        
        mode_suffix = 'tdb' if args.tdb_only else 'fulldb'
        hp_suffix = args.hp_mode.lower() if args.hp_mode != 'Both' else 'both'
        
        if exp_id:
            output_file = os.path.join(
                args.input_dir,
                f'{args.output_prefix}_{exp_id}_{hp_suffix}_{mode_suffix}.csv'
            )
        else:
            output_file = os.path.join(
                args.input_dir,
                f'{args.output_prefix}_no_exp_id_{hp_suffix}_{mode_suffix}.csv'
            )
        
        df.to_csv(output_file, index=False, float_format='%.4f')
        output_files_created.append(output_file)
        print(f"\n[SAVED] {output_file}")
        
        print("\nPreview:")
        if args.tdb_only:
            preview_cols = ['method', 'hp_mode', 'pl_performance', 'shot_performance', 'tdb_performance']
        else:
            preview_cols = ['method', 'hp_mode', 'pl_performance', 'shot_performance']
        
        if all(col in df.columns for col in preview_cols):
            print(df[preview_cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("ORACLE RESULTS AGGREGATION COMPLETED")
    print("="*80)
    print(f"Processed {len(results_by_exp_id)} experiment ID(s)")
    print(f"Generated {len(output_files_created)} output file(s)")
    mode_name = 'Tiny DomainBed' if args.tdb_only else 'Full DomainBed'
    print(f"Mode: {args.hp_mode} ({mode_name} - {len(env_list)} environments)")
    print(f"Statistics: Seed-based variance (standard DG practice)")
    print(f"Granularity: Per-(environment, mark) with selected configs")
    print(f"TDB: {'COMPUTED' if args.tdb_only else 'NOT COMPUTED (Full DB mode)'}")
    
    if output_files_created:
        print("\nOutput files:")
        for output_file in output_files_created:
            print(f"  {output_file}")
    
    if not output_files_created:
        print("\n[ERROR] No output files were created - all experiment IDs failed processing!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
