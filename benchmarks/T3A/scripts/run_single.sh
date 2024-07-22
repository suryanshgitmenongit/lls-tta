#!/bin/bash

# bash scripts/run_single.sh

nvidia-smi
ulimit -n 8192

# Set the arrays for the input folders, adapt algorithms, granularities, find algos, and fix algos.
GPU=0
ADAPT_ALGORITHMS=(PseudoLabel SHOT)
GRANULARITIES=(Block)
FIND_ALGOS=(Layer Layer1 Layer2 Layer3 Layer4)

BEST_HPS=(First Last Man)
INPUT_FOLDERS=(pacs_res18_erm0  vlcs_res18_erm0)
ENVS=(env0 env1 env2 env3)

TRACK_TTA="False"
CHECKPOINT_FREQ=1


# Set the paths for the data, output, code, and environments.
HOME_PATH=/home-local2/sasah14.extra.nobkp
DATA_DIR=${HOME_PATH}/scratch/data
OUT_PATH=${HOME_PATH}/scratch/out/estta/single_run
CODE_PATH=${HOME_PATH}/code/es-tta/benchmarks/T3A
ENVS_PATH=${HOME_PATH}/envs/ttdg


# Print the arrays and paths.
echo "gpu: $GPU"
echo "track tta: $TRACK_TTA"
echo "checkpoint freq: $CHECKPOINT_FREQ"

echo "data dir: ${DATA_DIR}"
echo "out path: ${OUT_PATH}"
echo "code path: ${CODE_PATH}"
echo "env path: ${ENVS_PATH}"

echo "best hps: ${BEST_HP[@]}"
echo "input folders: ${INPUT_FOLDERS[@]}"
echo "envs: ${ENVS[@]}"
echo "adapt algorithms: ${ADAPT_ALGORITHMS[@]}"
echo "granularities: ${GRANULARITIES[@]}"
echo "find algos: ${FIND_ALGOS[@]}"


# Activate the virtual environment and change to the code directory.
source ${ENVS_PATH}/bin/activate
cd ${CODE_PATH}


# A for loop that loops over the input folders, adapt algorithms, granularities, find algos, and fix algos based on the above arrays.
# For each combination of these arrays, run the unsupervised_adaptation script.
for input_folder in "${INPUT_FOLDERS[@]}"; do
    for env in "${ENVS[@]}"; do
        for adapt_algorithm in "${ADAPT_ALGORITHMS[@]}"; do
            for granularity in "${GRANULARITIES[@]}"; do
                for find_algo in "${FIND_ALGOS[@]}"; do
                    for best_hp in "${BEST_HPS[@]}"; do
                        CUDA_VISIBLE_DEVICES=$GPU \
                            python -m domainbed.scripts.unsupervised_adaptation \
                            --input_dir $OUT_PATH/$input_folder/$env \
                            --wandb_name "${input_folder}_${env}" \
                            --data_dir $DATA_DIR \
                            --adapt_algorithm $adapt_algorithm \
                            --granularity $granularity \
                            --find_algo $find_algo \
                            --fix_algo "Difflr" \
                            --track_tta $TRACK_TTA \
                            --ckpt_freq $CHECKPOINT_FREQ \
                            --best_hp $best_hp 

                        echo "run_single.sh: Done $input_folder $env $adapt_algorithm $granularity $find_algo $best_hp"
                    done
                done
            done
        done
    done
done

echo "run_single.sh: Done all"