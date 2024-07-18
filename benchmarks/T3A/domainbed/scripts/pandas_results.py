import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import re

RESULTS_PREFIX = 'results'
HP_RESULTS_PREFIX = 'hpSearch'

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def get_test_env(args):
    test_envs = args['test_envs']
    assert len(test_envs) == 1
    test_env = test_envs[0]
    return test_env

def get_names(df_res, test_env):
    train_out_names = [col for col in df_res.columns
                            if 'env' in col and
                                'out_acc' in col and
                                str(test_env) not in col]
    test_out_names = [col for col in df_res.columns
                            if 'env' in col and
                                'out_acc' in col and
                                str(test_env) in col]
    test_in_names = [col for col in df_res.columns
                            if 'env' in col and
                                'in_acc' in col and
                                str(test_env) in col]
    if len(test_out_names) == 1:
        test_out_name = test_out_names[0]
    else:
        print('Warning: No. of test_out_names is not 1!')
        test_out_name = np.nan

    if len(test_in_names) == 1:
        test_in_name = test_in_names[0]
    else:
        print('Warning: No. of test_in_names is not 1!')
        test_in_name = np.nan
    
    return train_out_names, test_out_name, test_in_name

def get_loc_alg_setting_names(loc_alg_name):
    if not loc_alg_name:
        return 'DG', 'DG', 'DG'
    
    splits = camel_case_split(loc_alg_name)
    if len(splits) == 1:
        find_algo = splits[0]
        fix_algo = splits[0]
        granularity = splits[0]
    else:
        granularity = splits[-1]
        fix_algo = splits[-2]
        find_algo = ''.join(splits[:-2])

    return find_algo, fix_algo, granularity

def fill_nans(acc_dict, test_in_name, adapt_algorithm, loc_alg_name):
    # Extract loc algo setting names
    find_algo, fix_algo, granularity = get_loc_alg_setting_names(loc_alg_name)

    # Append algo names to dictionary
    acc_dict['adapt_algorithm'].append(adapt_algorithm)
    acc_dict['find_algo'].append(find_algo)
    acc_dict['fix_algo'].append(fix_algo)
    acc_dict['granularity'].append(granularity)

    # Append target acc to dictionary
    acc_dict[test_in_name].append(np.nan)

def fill_dg(iid_dict, orc_dict, man_dict, df, train_out_names, test_out_name, test_in_name, args):
    # Get best train and test out idx
    df['train_out_mean'] = df[train_out_names].mean(skipna=False, axis=1)
    best_train_out_idx = df['train_out_mean'].idxmax()
    best_test_out_idx = df[test_out_name].idxmax()

    # Get best train and test out based test-in acc
    iidacc = df[test_in_name].iloc[best_train_out_idx]
    orcacc = df[test_in_name].iloc[best_test_out_idx]

    # TODO: Assumes that DG is encountered 1st, fix it!
    # Append acc to both dictionaries
    iid_dict[test_in_name] = [iidacc]
    orc_dict[test_in_name] = [orcacc]
    man_dict[test_in_name] = []
    
    # Append algo names to both dictionaries
    iid_dict['adapt_algorithm'] = ['DG']
    iid_dict['find_algo'] = ['DG']
    iid_dict['fix_algo'] = ['DG']
    iid_dict['granularity'] = ['DG']
    orc_dict['adapt_algorithm'] = ['DG']
    orc_dict['find_algo'] = ['DG']
    orc_dict['fix_algo'] = ['DG']
    orc_dict['granularity'] = ['DG']
    man_dict['adapt_algorithm'] = []
    man_dict['find_algo'] = []
    man_dict['fix_algo'] = []
    man_dict['granularity'] = []

    # Append seed to both dictionaries
    # iid_dict['hparams_seed'].append(args['hparams_seed'])
    # iid_dict['seed'].append(args['seed'])
    # iid_dict['trial_seed'].append(args['trial_seed'])
    # orc_dict['hparams_seed'].append(args['hparams_seed'])
    # orc_dict['seed'].append(args['seed'])
    # orc_dict['trial_seed'].append(args['trial_seed'])

def fill_row_loc(acc_dict, row, test_in_name, args):
    # Append algo names to dictionary
    acc_dict['adapt_algorithm'].append(args['adapt_algorithm'])
    if 'find_algo' in row: 
        # for loc TTA
        acc_dict['find_algo'].append(args['find_algo'])
        acc_dict['fix_algo'].append(args['fix_algo'])
        acc_dict['granularity'].append(args['granularity'])
    else:
        # for full TTA
        acc_dict['find_algo'].append('Full')
        acc_dict['fix_algo'].append('Full')
        acc_dict['granularity'].append('Full')

    # Append target acc to dictionary
    acc_dict[test_in_name].append(row[test_in_name])

def fill_meta_data(acc_dict, test_in_name, dataset, backbone, dg_name, mark, hp_mode):
    acc_dict['hp_mode'] = [hp_mode for _ in range(len(acc_dict[test_in_name]))]
    acc_dict['dataset'] = [dataset for _ in range(len(acc_dict[test_in_name]))]
    acc_dict['backbone'] = [backbone for _ in range(len(acc_dict[test_in_name]))]
    acc_dict['dg'] = [dg_name for _ in range(len(acc_dict[test_in_name]))]
    acc_dict['mark'] = [mark for _ in range(len(acc_dict[test_in_name]))]

def get_df(acc_dict):
    # Create dataframe from dictionary
    df = pd.DataFrame.from_dict(acc_dict)

    # Compute average accuracy
    df['Avg'] = df[[col for col in df.columns
                            if 'in_acc' in col]].mean(skipna=False, axis=1)

    # Sort dataframe columns
    df = df[['adapt_algorithm', 'granularity', 'find_algo', 'fix_algo',
                     'env0_in_acc', 'env1_in_acc', 'env2_in_acc', 'env3_in_acc', 'Avg',
                     'hp_mode', 'dataset', 'backbone', 'dg', 'mark']]
    
    return df

def ask_for_ignoring_incorrect_sweep_name(sweep_str):
    print('Incorrect sweep: {}'.format(sweep_str))
    response = input('Ignore it and proceed? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Stopped processing!')
        exit(0)
    print('Ignoring it!')

def get_df_final(df_full):
    # Check if dataframe is empty
    if df_full.shape[0] == 0:
        print('Warning: Empty dataframe passed to get_df_final!')
        # Return empty dataframe
        return pd.DataFrame()

    df_full['backbone_dataset_mark'] = df_full['backbone'] + '_' + df_full['dataset'] + '_' + df_full['mark']

    res_list = []
    for glist, df_group in df_full.groupby(['dg', 'adapt_algorithm', 'granularity', 'find_algo', 'fix_algo']):
        gname = '__'.join(glist)
        res_df = df_group.set_index(['backbone_dataset_mark'])[['Avg']].rename(columns={'Avg': gname}).T.reset_index().rename(columns={'index': 'method'})
        res_list.append(res_df)

    df_final = pd.concat(res_list)

    # Split the method column into 5 columns
    df_final['dg'] = df_final['method'].apply(lambda x: x.split('__')[0])
    df_final['adapt_algorithm'] = df_final['method'].apply(lambda x: x.split('__')[1])
    df_final['granularity'] = df_final['method'].apply(lambda x: x.split('__')[2])
    df_final['find_algo'] = df_final['method'].apply(lambda x: x.split('__')[3])
    df_final['fix_algo'] = df_final['method'].apply(lambda x: x.split('__')[4])

    # Drop the method column
    df_final = df_final.drop(columns=['method'])

    # Sort all columns so that the above 5 columns are in the front
    cols = df_final.columns.tolist()
    cols = cols[-5:] + cols[:-5]
    df_final = df_final[cols]

    # Sort by dg, adapt_algorithm, granularity, find_algo, fix_algo
    df_final = df_final.sort_values(by=['dg', 'adapt_algorithm', 'granularity', 'find_algo', 'fix_algo'])

    # Reset index
    df_final = df_final.reset_index(drop=True)

    return df_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pandas based collect results')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--sweep_names', nargs='+', type=str, default=[""])
    args_in = parser.parse_args()

    sweep_tables = {}

    for sweep_name in args_in.sweep_names:
        print('Processing sweep_name: {}'.format(sweep_name))
        name_splits = sweep_name.split('_')
        if len(name_splits) != 4:
            print('Error: No. of splits in sweep_name is not 4!')
            ask_for_ignoring_incorrect_sweep_name(sweep_name)
            continue
        dataset = name_splits[0]
        backbone = name_splits[1]
        dg_name = "_".join(name_splits[2:-1])
        mark = name_splits[-1]

        # Get sweep dir
        sweep_dir = os.path.join(args_in.input_dir, sweep_name)

        if not os.path.exists(sweep_dir):
            print('Error: Sweep dir not found for {}'.format(sweep_dir))
            ask_for_ignoring_incorrect_sweep_name(sweep_dir)
            continue
        
        iidacc_dict, orcacc_dict = {}, {}
        manacc_dict = {}
        # iidacc_dict['hparams_seed'] = []
        # iidacc_dict['seed'] = []
        # iidacc_dict['trial_seed'] = []
        # orcacc_dict['hparams_seed'] = []
        # orcacc_dict['seed'] = []
        # orcacc_dict['trial_seed'] = []

        # Get runs
        runs = [run for run in sorted(os.listdir(sweep_dir))
                                                if os.path.isdir(os.path.join(sweep_dir, run))]

        # Get unique list of all results fnames
        res_fnames = sorted(list(set([fname for run in runs
                                                for fname in os.listdir(os.path.join(sweep_dir, run))
                                                    if RESULTS_PREFIX in fname and\
                                                        'hp' not in fname and\
                                                        '.jsonl' in fname])))

        # Iterate over results fnames
        for run in tqdm(runs):
            run_to_testinname = {}  # dict to map a given run to test-in name
            for res_fname in res_fnames:
                # Get run dir and results path
                run_dir = os.path.join(sweep_dir, run)
                res_path = os.path.join(run_dir, res_fname)

                # check if results file exists
                if not os.path.exists(res_path):
                    print('Warning: Results file not found for {}'.format(res_path))

                    # Fill NaNs for all columns
                    print(run_to_testinname)
                    res_name_split = res_fname[:-6].split('_')
                    fill_nans(iidacc_dict, run_to_testinname[run], res_name_split[1], '_'.join(res_name_split[2:]))
                    fill_nans(orcacc_dict, run_to_testinname[run], res_name_split[1], '_'.join(res_name_split[2:]))
                    fill_nans(manacc_dict, run_to_testinname[run], res_name_split[1], '_'.join(res_name_split[2:]))

                else:
                    # print('Results file found for {}'.format(res_path))

                    # Read in results
                    df_res = pd.read_json(os.path.join(run_dir, res_fname), lines=True)

                    # Check if in_acc columns exist
                    if sum(['in_acc' in col for col in df_res.columns]) == 0:
                        print('Warning: No in_acc columns found for {}'.format(res_fname))

                        # Fill NaNs for all columns
                        print(run_to_testinname)
                        res_name_split = res_fname[:-6].split('_')
                        fill_nans(iidacc_dict, run_to_testinname[run], res_name_split[1], '_'.join(res_name_split[2:]))
                        fill_nans(orcacc_dict, run_to_testinname[run], res_name_split[1], '_'.join(res_name_split[2:]))
                        fill_nans(manacc_dict, run_to_testinname[run], res_name_split[1], '_'.join(res_name_split[2:]))

                        continue

                    # Get args
                    args = df_res['args'][0]

                    # Get test env
                    test_env = get_test_env(args)

                    # Get train and test in and out names
                    train_out_names, test_out_name, test_in_name = get_names(df_res, test_env)

                    res_name_split = res_fname[:-6].split('_')
                    loc_alg_name = '_'.join(res_name_split[2:])
                    find_algo, fix_algo, granularity = get_loc_alg_setting_names(loc_alg_name)
                    if len(res_name_split) == 1:
                        # Fill DG
                        fill_dg(iidacc_dict, orcacc_dict, manacc_dict, df_res, train_out_names, test_out_name, test_in_name, args)
                        run_to_testinname[run] = test_in_name

                    elif 'domaingen' in find_algo.lower():
                        row = df_res.iloc[0]
                        fill_row_loc(iidacc_dict, row, test_in_name, args)
                        fill_row_loc(orcacc_dict, row, test_in_name, args)
                        fill_row_loc(manacc_dict, row, test_in_name, args)

                    else:
                        if len(res_name_split) < 3:
                            print('Warning: Processing unknown results file name: {}'.format(res_fname))

                        # Collect the performance values for 3 HP tuning modes (rewriting with the latest one): TODO: Check this later properly
                        loc_iid_flag, loc_orc_flag, loc_man_flag = False, False, False
                        temp_row_dict = {}
                        for idx, row in df_res.iterrows():
                            hp_mode = row['hp_mode']
                            if hp_mode == 'Iid':
                                temp_row_dict['Iid'] = row
                                loc_iid_flag = True
                            
                            elif hp_mode == 'Orc':
                                temp_row_dict['Orc'] = row
                                loc_orc_flag = True

                            elif hp_mode == 'Man':
                                temp_row_dict['Man'] = row
                                loc_man_flag = True

                        # Store the final values in the main dictionary
                        if loc_iid_flag:
                            fill_row_loc(iidacc_dict, temp_row_dict['Iid'], test_in_name, args)
                        else:
                            print('Warning: No Iid results found for {}'.format(res_fname))
                            fill_nans(iidacc_dict, run_to_testinname[run], res_name_split[1], '_'.join(res_name_split[2:]))
                        if loc_orc_flag:
                            fill_row_loc(orcacc_dict, temp_row_dict['Orc'], test_in_name, args)
                        else:
                            print('Warning: No Orc results found for {}'.format(res_fname))
                            fill_nans(orcacc_dict, run_to_testinname[run], res_name_split[1], '_'.join(res_name_split[2:]))
                        if loc_man_flag:
                            fill_row_loc(manacc_dict, temp_row_dict['Man'], test_in_name, args)
                        else:
                            print('Warning: No Man results found for {}'.format(res_fname))
                            fill_nans(manacc_dict, run_to_testinname[run], res_name_split[1], '_'.join(res_name_split[2:]))


        # Check if dictionary is empty
        if len(iidacc_dict) == 0 or len(orcacc_dict) == 0 or len(manacc_dict) == 0:
            print('Warning: Empty dictionaries created for {}'.format(sweep_name))
            continue

        # Fill meta data
        fill_meta_data(iidacc_dict, test_in_name, dataset, backbone, dg_name, mark, 'Iid')
        fill_meta_data(orcacc_dict, test_in_name, dataset, backbone, dg_name, mark, 'Orc')
        fill_meta_data(manacc_dict, test_in_name, dataset, backbone, dg_name, mark, 'Man')

        # Create dataframe from dictionary
        df_iid = get_df(iidacc_dict)
        df_orc = get_df(orcacc_dict)
        df_man = get_df(manacc_dict)

        # Save dataframe to csv
        df_iid.to_csv(os.path.join(sweep_dir, sweep_name + '_iid.csv'), index=False)
        df_orc.to_csv(os.path.join(sweep_dir, sweep_name + '_orc.csv'), index=False)
        df_man.to_csv(os.path.join(sweep_dir, sweep_name + '_man.csv'), index=False)

        # Add dataframe to sweep tables dictionary
        sweep_tables[sweep_name + '_iid'] = df_iid
        sweep_tables[sweep_name + '_orc'] = df_orc
        sweep_tables[sweep_name + '_man'] = df_man

    # Check if sweep_tables is empty or contains single dataframe
    if len(sweep_tables) == 0:
        print('Warning: Empty sweep_tables created! No results found!')
    elif len(sweep_tables) == 2:
        print('Only single dataframe found in sweep_tables! No merging needed!')

        df_all_iid = df_iid
        df_all_orc = df_orc
        df_all_man = df_man
        
    else:
        print('Merging dataframes...')
        # Merge all dataframes in sweep_tables
        df_all_iid = pd.concat([df for dfname, df in sweep_tables.items() if 'iid' in dfname])
        df_all_orc = pd.concat([df for dfname, df in sweep_tables.items() if 'orc' in dfname])
        df_all_man = pd.concat([df for dfname, df in sweep_tables.items() if 'man' in dfname])

    # Save merged dataframes to csv
    df_all_iid.to_csv(os.path.join(args_in.input_dir, 'iid.csv'), index=False)
    df_all_orc.to_csv(os.path.join(args_in.input_dir, 'orc.csv'), index=False)
    df_all_man.to_csv(os.path.join(args_in.input_dir, 'man.csv'), index=False)

    # Get final dataframes
    df_final_iid = get_df_final(df_all_iid)
    df_final_orc = get_df_final(df_all_orc)
    df_final_man = get_df_final(df_all_man)

    # Save final dataframes to csv
    df_final_iid.to_csv(os.path.join(args_in.input_dir, 'iid_final.csv'), index=False)
    df_final_orc.to_csv(os.path.join(args_in.input_dir, 'orc_final.csv'), index=False)
    df_final_man.to_csv(os.path.join(args_in.input_dir, 'man_final.csv'), index=False)

    print('Finished {} sweep_names!'.format(len(args_in.sweep_names)))