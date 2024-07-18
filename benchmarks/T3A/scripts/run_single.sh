#!/bin/bash

# bash scripts/run_single.sh

nvidia-smi
ulimit -n 8192

# Set the arrays for the input folders, adapt algorithms, granularities, find algos, and fix algos.
GPU=0
ADAPT_ALGORITHMS=(PseudoLabel SHOT)
GRANULARITIES=(Block)
FIND_ALGOS=(Layer1)

BEST_HP="Man"
INPUT_FOLDERS=( terra0_res18    terra1_res18    terra2_res18
                pacs1_res18     pacs3_res18
                vlcs3_res18)

TRACK_TTA="False"
CHECKPOINT_FREQ=50
WANDB_NAME_SUFFIX="_"
FIX_ALGOS=(Difflr)


# Set the paths for the data, output, code, and environments.
HOME_PATH=/home-local2/sasah14.extra.nobkp
DATA_DIR=${HOME_PATH}/scratch/data
OUT_PATH=${HOME_PATH}/scratch/out/stta/single_run
CODE_PATH=${HOME_PATH}/code/ttdg
ENVS_PATH=${HOME_PATH}/envs/ttdg


# Print the arrays and paths.
echo "gpu: $GPU"
echo "best hp: $BEST_HP"
echo "track tta: $TRACK_TTA"
echo "checkpoint freq: $CHECKPOINT_FREQ"
echo "wandb name suffix: $WANDB_NAME_SUFFIX"
echo "input folders: ${INPUT_FOLDERS[@]}"
echo "adapt algorithms: ${ADAPT_ALGORITHMS[@]}"
echo "granularities: ${GRANULARITIES[@]}"
echo "find algos: ${FIND_ALGOS[@]}"
echo "fix algos: ${FIX_ALGOS[@]}"
echo "data dir: ${DATA_DIR}"
echo "out path: ${OUT_PATH}"
echo "code path: ${CODE_PATH}"
echo "envs path: ${ENVS_PATH}"


# Activate the virtual environment and change to the code directory.
source ${ENVS_PATH}/bin/activate
cd ${CODE_PATH}


# A for loop that loops over the input folders, adapt algorithms, granularities, find algos, and fix algos based on the above arrays.
# For each combination of these arrays, run the unsupervised_adaptation script.
for input_folder in "${INPUT_FOLDERS[@]}"; do
    for adapt_algorithm in "${ADAPT_ALGORITHMS[@]}"; do
        for granularity in "${GRANULARITIES[@]}"; do
            for find_algo in "${FIND_ALGOS[@]}"; do
                for fix_algo in "${FIX_ALGOS[@]}"; do

                    CUDA_VISIBLE_DEVICES=$GPU \
                        python -m domainbed.scripts.unsupervised_adaptation \
                        --data_dir $DATA_DIR \
                        --fix_algo $fix_algo --granularity $granularity \
                        --track_tta $TRACK_TTA --checkpoint_freq $CHECKPOINT_FREQ --best_hp $BEST_HP \
                        \
                        --input_dir $OUT_PATH/$input_folder \
                        --wandb_name $input_folder$WANDB_NAME_SUFFIX \
                        --adapt_algorithm $adapt_algorithm \
                        --find_algo $find_algo

                    echo "more_commands.sh: Done $input_folder $adapt_algorithm $granularity $find_algo $fix_algo"
                done
            done
        done
    done
done

echo "commands.sh: Done all"