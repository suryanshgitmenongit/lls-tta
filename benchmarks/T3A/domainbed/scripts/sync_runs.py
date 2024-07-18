import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sync offline wandb runs to dashboard')
    parser.add_argument("--input_dirs", nargs='+', type=str, default=[""])
    args = parser.parse_args()

    # Loop through the folders and sync each run
    for sweep_dir in args.input_dirs:
        for run_folder in sorted(os.listdir(sweep_dir)):
            wand_run_path = os.path.join(sweep_dir, run_folder, 'wandb')
            print('Run path: {}'.format(wand_run_path))
            
            # Change to the folder containing the offline runs
            os.chdir(wand_run_path)

            # Run the wandb sync command to sync the runs
            os.system('wandb sync ./offline-run*')