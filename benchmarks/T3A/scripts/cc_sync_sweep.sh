#!/bin/bash

# bash cc_sync_runs.sh

# Specify space-separated paths to sweep folders
INPUT_DIRS=()

HOME_PATH=/home/saby
# HOME_PATH=/home-local2/sasah14.extra.nobkp   # on IID
CODE_PATH=${HOME_PATH}/code/ttdg
ENVS_PATH=${HOME_PATH}/envs/ttdg
OUT_PATH=${HOME_PATH}/scratch/out/ttdg/sweep

echo "__________________________["
echo "cc_sync_runs.sh args received:"
echo ""
echo "input_dirs: ${INPUT_DIRS}"
echo "out path: ${OUT_PATH}"
echo "code path: ${CODE_PATH}"
echo "envs path: ${ENVS_PATH}"
echo "__________________________]"

# Prepare virtualenv
module load python/3.8    # not on IID
source ${ENVS_PATH}/bin/activate
echo "__________________________["
echo "Environment activated."
echo "__________________________]"

# Run command
cd ${CODE_PATH}
for INPUT_DIR in "${INPUT_DIRS[@]}"
do
    python -m domainbed.scripts.sync_runs \
       --input_dir=${OUT_PATH}/${INPUT_DIR}
done

echo "__________________________["
echo "cc_sync_runs.sh finished."
echo "__________________________]"
