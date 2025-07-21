#!/usr/bin/env python3
"""
TDB Hyperparameter Ablation Results Aggregation Script 
Aggregates results across all TDB environments with detailed tracking

This extended version adds:
1. Individual environment scores in output CSV
2. Source file paths for each score
3. Full configuration strings
4. Detailed logging for verification
"""

import argparse
import json
import os
import glob
import re
from collections import defaultdict, OrderedDict
import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback

# TDB environment mapping - these are the ONLY environments we care about
TDB_ENVIRONMENTS = {
    'PACS': [1, 3],              # Cartoon, Sketch  
    'VLCS': [2],                 # SUN09
    'TerraIncognita': [0, 1, 2], # L100, L38, L43
}

# All TDB environments flattened for easier access
ALL_TDB_ENVS = []
for dataset, envs in TDB_ENVIRONMENTS.items():
    for env in envs:
        ALL_TDB_ENVS.append((dataset, env))

print(f"[DEBUG] Total TDB environments: {len(ALL_TDB_ENVS)}")
print(f"[DEBUG] Environment list: {ALL_TDB_ENVS}")

def parse_hp_from_find_algo(find_algo, fix_algo=None, granularity=None):
    """
    Robustly extract hyperparameter name and value from find_algo string.
    
    The challenge: find_algo might be concatenated with fix_algo and granularity
    Example: "LLS_emalr:0.3DifflrLayer" where:
    - find_algo should be "LLS_emalr:0.3"
    - fix_algo is "Difflr" 
    - granularity is "Layer"
    
    Args:
        find_algo: The raw string that might contain concatenated values
        fix_algo: Known fix algorithm (helps with parsing)
        granularity: Known granularity (helps with parsing)
    
    Returns:
        tuple: (hp_name, hp_value, clean_find_algo)
    """
    if not find_algo or find_algo in ['None', 'Full', '']:
        return 'baseline', 'none', 'None'
    
    if find_algo == 'LEAST':
        return 'baseline', 'LEAST', 'LEAST'
    
    # Special handling for Nolocadapt
    if find_algo == 'Nolocadapt' or 'Nolocadapt' in find_algo:
        return 'baseline', 'nolocadapt', 'Nolocadapt'
    
    # First, try to extract the clean find_algo by removing known suffixes
    clean_find = find_algo
    
    # List of all possible suffixes to remove (order matters - longest first)
    possible_suffixes = []
    
    # Add combinations of fix_algo and granularity
    if fix_algo and fix_algo != 'None' and granularity and granularity != 'None':
        possible_suffixes.append(fix_algo + granularity)
    if fix_algo and fix_algo != 'None':
        possible_suffixes.append(fix_algo)
    if granularity and granularity != 'None':
        possible_suffixes.append(granularity)
    
    # Also add common patterns we might see
    for pattern in ['DifflrLayer', 'DifflrBlock', 'DifflrSubblock', 'NoneLayer', 'NoneBlock']:
        if pattern not in possible_suffixes:
            possible_suffixes.append(pattern)
    
    # Remove suffixes
    for suffix in possible_suffixes:
        if clean_find.endswith(suffix):
            # Check if removing this suffix would cut into a parameter value
            suffix_start = len(clean_find) - len(suffix)
            # Only remove if there's no colon in the suffix part AND
            # there's something meaningful left after removal
            if ':' not in clean_find[suffix_start:] and suffix_start > 0:
                potential_clean = clean_find[:suffix_start]
                # Verify we're not cutting a parameter value
                if not potential_clean.endswith(':'):
                    clean_find = potential_clean
                    break
    
    # Handle case where colons might have been replaced with underscores
    if '_' in clean_find and ':' not in clean_find:
        # Try to reconstruct - look for patterns like "emalr_0.3"
        clean_find = re.sub(r'(\w+)_(\d+\.?\d*)', r'\1:\2', clean_find)
    
    # Now parse the clean find_algo
    if not clean_find.startswith('LLS'):
        return 'unknown', clean_find, clean_find
    
    # Remove LLS prefix (with or without underscore)
    if clean_find.startswith('LLS_'):
        hp_string = clean_find[4:]
    elif clean_find.startswith('LLS'):
        hp_string = clean_find[3:]
    else:
        hp_string = clean_find
    
    # Parse key:value pairs
    # Use regex to find all key:value patterns
    pattern = r'(\w+):([^_\s]+)'
    matches = re.findall(pattern, hp_string)
    
    if not matches:
        # Try alternative patterns in case of different formatting
        # e.g., "emalr0.3" without colon
        alt_pattern = r'(emalr|lr|steps|patience|init|reg|regstr)(\d+\.?\d*)'
        alt_matches = re.findall(alt_pattern, hp_string)
        if alt_matches:
            matches = [(m[0], m[1]) for m in alt_matches]
        else:
            return 'unknown', hp_string, clean_find
    
    # For ablation, we typically vary one parameter at a time
    # Look for the non-baseline parameter
    baseline_params = {
        'init': 'least',
        'appl': 'avg',  # Note: might be changed to 'ema' per instructions
    }
    
    # First, look for commonly ablated parameters
    ablation_params = ['emalr', 'lr', 'steps', 'patience', 'reg', 'regstr', 'loss', 'bin', 'appl']
    
    for key, value in matches:
        if key in ablation_params:
            # This is likely our ablated parameter
            return key, value, clean_find
    
    # If no ablation params found, look for non-baseline params
    for key, value in matches:
        # Skip baseline parameters
        if key in baseline_params and value == baseline_params[key]:
            continue
        # This is our varied parameter
        return key, value, clean_find
    
    # If all parameters are baseline, return the first one
    if matches:
        return matches[0][0], matches[0][1], clean_find
    
    return 'unknown', hp_string, clean_find

def find_all_result_files(base_dirs):
    """
    Find all result files across all hash directories in the given base directories.
    
    This function:
    1. Discovers all hash directories (32-char hex strings)
    2. Searches for result files matching our TTA experiments
    3. Returns detailed information about each file found
    
    Returns:
        list: List of dictionaries with file information
    """
    all_files = []
    
    print("\n[DETAILED] Starting directory scan...")
    
    for base_dir in base_dirs:
        dataset = os.path.basename(base_dir).split('_')[0]  # Extract dataset name
        
        print(f"\n[DETAILED] Scanning base directory: {base_dir}")
        
        if not os.path.exists(base_dir):
            print(f"[WARNING] Directory does not exist: {base_dir}")
            continue
        
        # Find all hash directories (32 character hex strings)
        # Also check for 'done' file to ensure it's a valid experiment directory
        hash_dirs = []
        all_items = os.listdir(base_dir)
        print(f"[DETAILED] Found {len(all_items)} items in {dataset} directory")
        
        for item in all_items:
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and len(item) == 32:
                # Verify it's a hex string
                try:
                    int(item, 16)
                    # Check if it contains result files or done markers
                    contents = os.listdir(item_path)
                    if any(f.startswith('results_') or f.startswith('done') for f in contents):
                        hash_dirs.append(item)
                        print(f"[DETAILED]   ✓ Valid hash dir: {item}")
                except ValueError:
                    # Not a valid hex string
                    print(f"[DETAILED]   ✗ Invalid hex: {item}")
                    continue
        
        print(f"\n[{dataset}] Found {len(hash_dirs)} hash directories")
        
        # Search each hash directory
        for idx, hash_dir in enumerate(sorted(hash_dirs)):
            hash_path = os.path.join(base_dir, hash_dir)
            print(f"\n[DETAILED] Scanning hash dir {idx+1}/{len(hash_dirs)}: {hash_dir}")
            
            # List all files in the directory
            all_contents = os.listdir(hash_path)
            print(f"[DETAILED]   Total files: {len(all_contents)}")
            
            # Find all result files
            # Handle both with and without colons in filename
            result_patterns = [
                'results_PseudoLabel_*.jsonl',
                'results_T3A_*.jsonl',
                'results_TentFull_*.jsonl',
                'results_TentNorm_*.jsonl',
                'results_SHOT_*.jsonl',
                'results_SHOTIM_*.jsonl',
                'results_PLClf_*.jsonl',
                'results_TentPreBN_*.jsonl',
                'results_TentClf_*.jsonl'
            ]
            
            found_files = []
            for pattern in result_patterns:
                matches = glob.glob(os.path.join(hash_path, pattern))
                if matches:
                    print(f"[DETAILED]   Pattern '{pattern}' matched {len(matches)} files")
                found_files.extend(matches)
            
            # Also try finding files with underscores instead of colons
            # In case files were already renamed
            for f in all_contents:
                if f.startswith('results_') and f.endswith('.jsonl') and f not in ['results.jsonl']:
                    full_path = os.path.join(hash_path, f)
                    if full_path not in found_files and 'hpSearch' not in f:
                        found_files.append(full_path)
                        print(f"[DETAILED]   Additional file found: {f}")
            
            print(f"[DETAILED]   Total result files found: {len(found_files)}")
            
            for result_file in found_files:
                filename = os.path.basename(result_file)
                print(f"[DETAILED]     Processing: {filename}")
                
                # Parse filename to extract algorithm info
                # Format: results_{adapt_algorithm}_{loc_alg_name}.jsonl
                parts = filename.replace('.jsonl', '').split('_', 2)
                if len(parts) >= 3:
                    adapt_algo = parts[1]
                    loc_alg_name = parts[2]
                    
                    # Try to parse loc_alg_name
                    # It might be like "LLS_emalr:0.3DifflrLayer" or "LLS_emalr_0.3DifflrLayer"
                    # We need to separate find_algo, fix_algo, and granularity
                    
                    # Common patterns to look for
                    fix_algos = ['Difflr', 'None']
                    granularities = ['Layer', 'Block', 'Subblock', 'None']
                    
                    # Try to identify fix_algo and granularity
                    detected_fix = None
                    detected_gran = None
                    
                    for fa in fix_algos:
                        if fa in loc_alg_name and fa != 'None':
                            detected_fix = fa
                            break
                    
                    for gr in granularities:
                        if loc_alg_name.endswith(gr) and gr != 'None':
                            detected_gran = gr
                            break
                    
                    file_info = {
                        'dataset': dataset,
                        'file_path': result_file,
                        'filename': filename,
                        'hash_dir': hash_dir,
                        'adapt_algo': adapt_algo,
                        'loc_alg_name': loc_alg_name,
                        'detected_fix': detected_fix,
                        'detected_gran': detected_gran
                    }
                    
                    all_files.append(file_info)
                    print(f"[DETAILED]       Added: adapt={adapt_algo}, loc={loc_alg_name}")
                    
        print(f"\n[{dataset}] Total files collected: {len([f for f in all_files if f['dataset'] == dataset])}")
    
    return all_files

def extract_hp_results(file_info):
    """
    Extract hyperparameter and accuracy results from a single result file.
    
    This function handles various data structure variations and provides
    comprehensive error reporting.
    
    Returns:
        list: List of dictionaries with hp info and accuracies
    """
    results = []
    file_path = file_info['file_path']
    
    print(f"\n[DETAILED] Extracting from: {file_info['filename']}")
    print(f"[DETAILED]   Full path: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            print(f"[WARNING] Empty file: {file_path}")
            return results
        
        print(f"[DETAILED]   Lines in file: {len(lines)}")
            
        for line_num, line in enumerate(lines):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # Extract args - might be at different levels
                args = data.get('args', {})
                if not args and isinstance(data, dict):
                    # Sometimes args might be nested differently
                    for key in ['config', 'setup', 'parameters']:
                        if key in data and 'test_envs' in data[key]:
                            args = data[key]
                            break
                
                # Get test environment
                test_envs = args.get('test_envs', [])
                if not test_envs:
                    print(f"[WARNING] No test_envs in {file_path} line {line_num}")
                    continue
                test_env = test_envs[0] if isinstance(test_envs, list) else test_envs
                
                # Get HP mode
                hp_mode = data.get('hp_mode', 'Man')  # Default to Manual if not specified
                
                # Extract find_algo, fix_algo, granularity
                # These might be in args or in the data directly
                find_algo = args.get('find_algo', data.get('find_algo', ''))
                fix_algo = args.get('fix_algo', data.get('fix_algo', file_info.get('detected_fix', '')))
                granularity = args.get('granularity', data.get('granularity', file_info.get('detected_gran', '')))
                
                # If find_algo is empty, try to parse from loc_alg_name
                if not find_algo and 'loc_alg_name' in file_info:
                    # This is a fallback - try to extract from filename
                    loc_alg = file_info['loc_alg_name']
                    if 'LLS' in loc_alg:
                        find_algo = loc_alg  # Will be parsed later
                
                print(f"[DETAILED]   Line {line_num}: test_env={test_env}, hp_mode={hp_mode}, find_algo={find_algo}")
                
                # Parse hyperparameter
                hp_name, hp_value, clean_find = parse_hp_from_find_algo(find_algo, fix_algo, granularity)
                print(f"[DETAILED]     Parsed: hp_name={hp_name}, hp_value={hp_value}")
                
                # Find accuracy for the test environment
                acc_key = f'env{test_env}_in_acc'
                accuracy = None
                
                # Try different possible locations for accuracy
                if acc_key in data:
                    accuracy = data[acc_key]
                elif 'results' in data and acc_key in data['results']:
                    accuracy = data['results'][acc_key]
                elif 'accuracies' in data and acc_key in data['accuracies']:
                    accuracy = data['accuracies'][acc_key]
                
                if accuracy is not None:
                    result_dict = {
                        'dataset': file_info['dataset'],
                        'hp_name': hp_name,
                        'hp_value': hp_value,
                        'hp_mode': hp_mode,
                        'test_env': test_env,
                        'accuracy': float(accuracy),
                        'find_algo': find_algo,
                        'fix_algo': fix_algo,
                        'granularity': granularity,
                        'clean_find': clean_find,
                        'adapt_algo': file_info['adapt_algo'],
                        'source_file': file_info['filename'],
                        'full_path': file_info['file_path'],  # Add full path for tracking
                        'hash_dir': file_info['hash_dir']     # Add hash dir for tracking
                    }
                    results.append(result_dict)
                    print(f"[DETAILED]     ✓ Extracted: accuracy={accuracy:.4f}")
                else:
                    print(f"[WARNING] No accuracy found for {acc_key} in {file_path} line {line_num}")
                    
            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid JSON in {file_path} line {line_num}: {e}")
            except Exception as e:
                print(f"[ERROR] Processing {file_path} line {line_num}: {e}")
                traceback.print_exc()
    
    except Exception as e:
        print(f"[ERROR] Cannot read file {file_path}: {e}")
    
    print(f"[DETAILED]   Total results extracted: {len(results)}")
    return results

def aggregate_tdb_results_extended(all_results):
    """
    Extended aggregation that preserves individual environment scores and source files.
    
    This function:
    1. Groups results by (hp_name, hp_value, hp_mode)
    2. Collects accuracies from ALL TDB environments with source tracking
    3. Computes the average across all environments
    4. Preserves individual scores for output
    """
    # Group results by hp configuration and mode
    grouped = defaultdict(lambda: {
        'accuracies': [],
        'dataset_env_pairs': [],
        'env_to_accuracy': {},  # Map (dataset, env) -> accuracy
        'env_to_source': {},    # Map (dataset, env) -> source file path
        'adapt_algos': set(),
        'full_configs': set()   # Store all full configuration strings
    })
    
    total_results = 0
    print("\n[DETAILED] Starting aggregation...")
    
    for dataset, results in all_results.items():
        # Get TDB environments for this dataset
        tdb_envs = TDB_ENVIRONMENTS.get(dataset, [])
        print(f"\n[DETAILED] Processing {dataset} with {len(results)} results")
        print(f"[DETAILED]   TDB environments: {tdb_envs}")
        
        for result in results:
            if result['test_env'] in tdb_envs:
                key = (result['hp_name'], result['hp_value'], result['hp_mode'])
                env_key = (dataset, result['test_env'])
                
                grouped[key]['accuracies'].append(result['accuracy'])
                grouped[key]['dataset_env_pairs'].append(env_key)
                grouped[key]['env_to_accuracy'][env_key] = result['accuracy']
                grouped[key]['env_to_source'][env_key] = result['full_path']
                grouped[key]['adapt_algos'].add(result['adapt_algo'])
                grouped[key]['full_configs'].add(result['clean_find'])
                
                total_results += 1
                print(f"[DETAILED]   Added: {key} -> {env_key} = {result['accuracy']:.4f}")
    
    print(f"\n[DETAILED] Total TDB results collected: {total_results}")
    
    # Compute aggregated results with extended information
    aggregated = []
    
    for (hp_name, hp_value, hp_mode), data in grouped.items():
        if data['accuracies']:
            # Calculate statistics
            accuracies = data['accuracies']
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies) if len(accuracies) > 1 else 0.0
            min_acc = np.min(accuracies)
            max_acc = np.max(accuracies)
            
            print(f"\n[DETAILED] Aggregating {hp_name}={hp_value} ({hp_mode}):")
            print(f"[DETAILED]   Accuracies: {[f'{a:.4f}' for a in accuracies]}")
            print(f"[DETAILED]   Average: {avg_acc:.4f} ± {std_acc:.4f}")
            
            # Check which datasets are represented
            datasets_included = list(set(pair[0] for pair in data['dataset_env_pairs']))
            
            # Get the most common full config string
            full_config = list(data['full_configs'])[0] if data['full_configs'] else ''
            
            result_dict = {
                'hp_name': hp_name,
                'hp_value': hp_value,
                'hp_mode': hp_mode,
                'tdb_avg': avg_acc,
                'tdb_std': std_acc,
                'tdb_min': min_acc,
                'tdb_max': max_acc,
                'n_tdb_envs': len(accuracies),
                'datasets_included': sorted(datasets_included),
                'full_config': full_config,
                # Store individual environment data for later expansion
                'env_to_accuracy': data['env_to_accuracy'],
                'env_to_source': data['env_to_source']
            }
            
            aggregated.append(result_dict)
    
    return aggregated

def create_extended_dataframe(aggregated):
    """
    Create DataFrame with individual environment scores as separate columns.
    """
    rows = []
    
    for result in aggregated:
        # Start with basic columns
        row = OrderedDict([
            ('hp_name', result['hp_name']),
            ('hp_value', result['hp_value']),
            ('tdb_performance', result['tdb_avg']),
            ('tdb_std', result['tdb_std']),
            ('n_envs', result['n_tdb_envs']),
            ('full_config', result['full_config'])
        ])
        
        # Add individual environment scores and sources
        # Order: PACS_1, PACS_3, VLCS_2, TerraIncognita_0, TerraIncognita_1, TerraIncognita_2
        env_order = [
            ('PACS', 1),
            ('PACS', 3),
            ('VLCS', 2),
            ('TerraIncognita', 0),
            ('TerraIncognita', 1),
            ('TerraIncognita', 2)
        ]
        
        for dataset, env in env_order:
            env_key = (dataset, env)
            # Add accuracy column
            col_name_acc = f'{dataset.lower()}_env{env}_acc'
            row[col_name_acc] = result['env_to_accuracy'].get(env_key, np.nan)
            
            # Add source file column
            col_name_src = f'{dataset.lower()}_env{env}_source'
            source_path = result['env_to_source'].get(env_key, '')
            # Extract just the filename for readability
            source_file = os.path.basename(source_path) if source_path else ''
            row[col_name_src] = source_file
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description='Aggregate TDB HP ablation results - Extended')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Base directory containing sweep results')
    parser.add_argument('--sweep_names', nargs='+', type=str, required=True,
                        help='Names of sweep directories to process')
    parser.add_argument('--output_prefix', type=str, default='tdb',
                        help='Prefix for output files')
    args = parser.parse_args()
    
    print("="*80)
    print("TDB Hyperparameter Ablation Results Aggregation - EXTENDED VERSION")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Sweep names: {args.sweep_names}")
    print(f"Output prefix: {args.output_prefix}")
    
    # Build full paths to sweep directories
    sweep_dirs = [os.path.join(args.input_dir, name) for name in args.sweep_names]
    
    # Verify directories exist
    missing_dirs = [d for d in sweep_dirs if not os.path.exists(d)]
    if missing_dirs:
        print("\n[ERROR] Missing directories:")
        for d in missing_dirs:
            print(f"  - {d}")
        return 1
    
    # Step 1: Find all result files
    print("\n" + "="*60)
    print("Step 1: Discovering result files")
    print("="*60)
    all_files = find_all_result_files(sweep_dirs)
    print(f"\nTotal result files found: {len(all_files)}")
    
    if not all_files:
        print("[ERROR] No result files found!")
        return 1
    
    # Show summary by dataset
    for dataset in TDB_ENVIRONMENTS:
        dataset_files = [f for f in all_files if f['dataset'] == dataset]
        print(f"  {dataset}: {len(dataset_files)} files")
    
    # Step 2: Extract results from all files
    print("\n" + "="*60)
    print("Step 2: Extracting hyperparameter results")
    print("="*60)
    all_results = defaultdict(list)
    
    for file_info in tqdm(all_files, desc="Processing files"):
        results = extract_hp_results(file_info)
        if results:
            all_results[file_info['dataset']].extend(results)
    
    # Show extraction summary
    print("\nExtraction summary:")
    total_extracted = 0
    for dataset, results in all_results.items():
        print(f"  {dataset}: {len(results)} results extracted")
        total_extracted += len(results)
        
        # Show unique HP configurations
        unique_configs = set((r['hp_name'], r['hp_value']) for r in results)
        print(f"    Unique HP configs: {len(unique_configs)}")
        for hp_name, hp_value in sorted(unique_configs):
            count = len([r for r in results if r['hp_name'] == hp_name and r['hp_value'] == hp_value])
            print(f"      {hp_name}={hp_value}: {count} results")
    
    print(f"\nTotal results extracted: {total_extracted}")
    
    if total_extracted == 0:
        print("[ERROR] No results extracted from files!")
        return 1
    
    # Step 3: Aggregate with extended tracking
    print("\n" + "="*60)
    print("Step 3: Aggregating across all TDB environments (with tracking)")
    print("="*60)
    aggregated = aggregate_tdb_results_extended(all_results)
    
    if not aggregated:
        print("[ERROR] No results to aggregate!")
        return 1
    
    # Step 4: Create output files for each hp_mode
    print("\n" + "="*60)
    print("Step 4: Creating extended output files")
    print("="*60)
    
    for hp_mode in ['Iid', 'Orc', 'Man']:
        mode_results = [r for r in aggregated if r['hp_mode'] == hp_mode]
        
        if mode_results:
            # Create extended DataFrame
            df = create_extended_dataframe(mode_results)
            
            # Sort by hp_name and hp_value
            df['hp_value_numeric'] = pd.to_numeric(df['hp_value'], errors='coerce')
            df = df.sort_values(['hp_name', 'hp_value_numeric', 'hp_value'])
            df = df.drop('hp_value_numeric', axis=1)
            
            # Save extended format
            output_file = os.path.join(args.input_dir, f'{args.output_prefix}_{hp_mode.lower()}_hp_ablation_extended.csv')
            df.to_csv(output_file, index=False, float_format='%.4f')
            print(f"\nSaved {hp_mode} extended results to: {output_file}")
            
            # Show preview
            print(f"\n  Preview of {hp_mode} extended results:")
            # Show first few rows with selected columns
            preview_cols = ['hp_name', 'hp_value', 'tdb_performance', 'tdb_std', 'n_envs', 'full_config']
            print(df[preview_cols].head(5).to_string(index=False))
            
            # Show statistics
            for hp_name in df['hp_name'].unique():
                hp_df = df[df['hp_name'] == hp_name]
                if len(hp_df) > 1:
                    best_idx = hp_df['tdb_performance'].idxmax()
                    worst_idx = hp_df['tdb_performance'].idxmin()
                    best_row = hp_df.loc[best_idx]
                    worst_row = hp_df.loc[worst_idx]
                    
                    print(f"\n  {hp_name}:")
                    print(f"    Best:  {best_row['hp_value']} -> {best_row['tdb_performance']:.4f} (±{best_row['tdb_std']:.4f})")
                    print(f"    Worst: {worst_row['hp_value']} -> {worst_row['tdb_performance']:.4f} (±{worst_row['tdb_std']:.4f})")
                    print(f"    Range: {best_row['tdb_performance'] - worst_row['tdb_performance']:.4f}")
    
    print("\n" + "="*80)
    print("EXTENDED OUTPUT FORMAT")
    print("="*80)
    print("\nColumns in output CSV:")
    print("  Basic columns:")
    print("    - hp_name: The hyperparameter being varied")
    print("    - hp_value: The value of that hyperparameter")
    print("    - tdb_performance: Average accuracy across all 6 TDB environments")
    print("    - tdb_std: Standard deviation across the 6 environments")
    print("    - n_envs: Number of environments included (should be 6)")
    print("    - full_config: The complete find_algo configuration string")
    print("\n  Individual environment columns (appended at end):")
    print("    - pacs_env1_acc, pacs_env1_source")
    print("    - pacs_env3_acc, pacs_env3_source")
    print("    - vlcs_env2_acc, vlcs_env2_source")
    print("    - terraincognita_env0_acc, terraincognita_env0_source")
    print("    - terraincognita_env1_acc, terraincognita_env1_source")
    print("    - terraincognita_env2_acc, terraincognita_env2_source")
    
    print("\n" + "="*80)
    print("Extended aggregation completed successfully!")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    exit(main())
