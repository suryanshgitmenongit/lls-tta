#!/bin/bash

# bash collect_results.sh

# Use the same parameters as run_sweep script
DATASET=PACS
BACKBONE=resnet18-BN
DG_ALGO=ERM
# TTA_ALGOS=""

MARK=1
SWEEP_NAME=${DATASET}_${BACKBONE}_${DG_ALGO}_${MARK}

HOME_PATH=/home-local2/sasah14.extra.nobkp
# HOME_PATH=/home/saby    # on CC
OUT_PATH=${HOME_PATH}/scratch/out/ttdg/sweep
CODE_PATH=${HOME_PATH}/code/ttdg
ENVS_PATH=${HOME_PATH}/envs/ttdg

echo "__________________________["
echo "get_results.sh args received:"
echo ""
echo "dataset: ${DATASET}"
echo "backbone: ${BACKBONE}"
echo "dg algorithm: ${DG_ALGO}"
# echo "tta algorithm(s): ${TTA_ALGOS}"
echo "mark: ${MARK}"
echo ""
echo "sweep name: ${SWEEP_NAME}"
echo "code path: ${CODE_PATH}"
echo "envs path: ${ENVS_PATH}"
echo "__________________________]"

# Prepare virtualenv
# module load python/3.8    # on CC
source ${ENVS_PATH}/bin/activate
echo "__________________________["
echo "Environment activated."
echo "__________________________]"

# Run command
cd ${CODE_PATH}
python -m domainbed.scripts.collect_results \
       --input_dir=${OUT_PATH}/${SWEEP_NAME} #\
    #    --adapt_algorithms ${TTA_ALGOS}

echo "__________________________["
echo "collect_results.sh finished."
echo "__________________________]"
