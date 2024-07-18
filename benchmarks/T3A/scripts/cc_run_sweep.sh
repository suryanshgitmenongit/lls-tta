#!/bin/bash
#SBATCH --gres=gpu:4       # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=96000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-16:00     # DD-HH:MM:SS

# sbatch cc_run_sweep.sh

nvidia-smi

# SWEEP_MODE : delete_incomplete (delete sweep folders),
#              launch (source training / DG),
#              unsupervised_adaptation (target adaptation / TTA),
#              supervised_adaptation (finetuning experiments)
# DATASET : PACS, VLCS, OfficeHome, TerraIncognita
# BACKBONE : "resnet18", *"resnet18-BN"*,  "resnet50", *"resnet50-BN"*,
#            "ViT-B16", ViT-B32, "ViT-L16", ViT-L32, ViT-H14,
#            Mixer-B16, "Mixer-L16",
#            Wide_ResNet, DINO, "DeiT", "HViT"
#            BiT-M-R50x1, "BiT-M-R50x3", BiT-M-R101x1, "BiT-M-R101x3", "BiT-M-R152x2", BiT-M-R152x4,
#            BiT-S-R50x1, BiT-S-R50x3, BiT-S-R101x1, BiT-S-R101x3, BiT-S-R152x2, BiT-S-R152x4
#            ()"model" reported in paper, *model* used for TTA baseline comparison)
# DG_ALGO: ERM, CORAL, ...
# TTA_ALGOS: T3A, SHOT, ...
#            (empty string implies all TTA algorithms mentioned in sweep)
# MARK : to identify the run type (like TRIAL_ID)
# SWEEP_MODE=launch
SWEEP_MODE=unsupervised_adaptation
DATASET=PACS
BACKBONE=resnet18-BN
DG_ALGO=ERM
# TTA_ALGOS=""
# TTA_ALGOS='TentFull TentClf'
# TTA_ALGOS='PseudoLabel PLClf'
# TTA_ALGOS='SHOT T3A'
TTA_ALGOS='SHOT'
# TTA_ALGOS='Rotn_TTA'
# TTA_ALGOS='Rotn_T3A_TTA'
# TTA_ALGOS='SimCLR_TTA'
# TTA_ALGOS='SimCLR_T3A_TTA'

# FIND_ALGOS=""
FIND_ALGOS="None"
# FIND_ALGOS="Best"
# FIND_ALGOS="Bestmulti"
# FIND_ALGOS="DomainGen"
# FIND_ALGOS="Surgrgn"
# FIND_ALGOS="Oursactgap"
# FIND_ALGOS="Png"
# FIND_ALGOS="Pngmultisharp"
# FIND_ALGOS="Pngmultismooth"
# FIND_ALGOS="PostUpdate"
# FIND_ALGOS="PostUpdate_PrevX"
# FIND_ALGOS="PostUpdate_TotalLoss"
# FIND_ALGOS="PostUpdate_PrevX_TotalLoss"

# FIX_ALGOS=""
FIX_ALGOS="None"
# FIX_ALGOS="Trainsubset"
# FIX_ALGOS="Difflr"

# GRANULARITIES=""
# GRANULARITIES="None"
GRANULARITIES="Block"
# GRANULARITIES="Subblock"
# GRANULARITIES="Layer"
MARK=0


# SWEEP_NAME : is set automatically (modify this only if using multiple dg algos or datasets)
# CMD_LAUNCH : multi_gpu, local, dummy
# N_TRIALS_FROM : 10 (fixed as per paper) 
# N_TRIALS : 1, 3 (1 used for quick experiments, 3 used in paper)
# DATA_DIR : path to parent folder of datasets or zip file
# OUT_PATH : path to parent sweep folder
SWEEP_NAME=${DATASET}_${BACKBONE}_${DG_ALGO}_${MARK}
CMD_LAUNCH=multi_gpu
N_TRIALS_FROM=10
N_TRIALS=1
HOME_PATH=/home/saby
DATA_DIR=${HOME_PATH}/scratch/data
OUT_PATH=${HOME_PATH}/scratch/out/stta/sweep
CODE_PATH=${HOME_PATH}/scratch/code/ttdg
ENVS_PATH=${HOME_PATH}/envs/ttdg
N_TRIALS_FROM=$((N_TRIALS_FROM + MARK))


echo "__________________________["
echo "run_sweep.sh args received:"
echo ""
echo "sweep_mode: ${SWEEP_MODE}"
echo "dataset: ${DATASET}"
echo "backbone: ${BACKBONE}"
echo "dg algorithm: ${DG_ALGO}"
if [ "$SWEEP_MODE" = "unsupervised_adaptation" ]; then
    echo "tta algorithm(s): ${TTA_ALGOS}"
    echo "find algorithm(s): ${FIND_ALGOS}"
    echo "fix algorithm(s): ${FIX_ALGOS}"
    echo "granularitie(s): ${GRANULARITIES}"
fi
echo "mark: ${MARK}"
echo ""
echo "sweep name: ${SWEEP_NAME}"
echo "command launcher: ${CMD_LAUNCH}"
echo "n_trials_from: ${N_TRIALS_FROM}"
echo "n_trials: ${N_TRIALS}"
echo "data dir: ${DATA_DIR}"
echo "out path: ${OUT_PATH}"
echo "code path: ${CODE_PATH}"
echo "envs path: ${ENVS_PATH}"
echo "__________________________]"

# Prepare virtualenv
module load python/3.8
source ${ENVS_PATH}/bin/activate
echo "__________________________["
echo "Environment activated."
echo "__________________________]"

# Prepare dataset
TMP_DATA_DIR=${SLURM_TMPDIR}/data/
mkdir ${TMP_DATA_DIR}
echo "Copying datasets:"
cp ${DATA_DIR}/${DATASET}.zip ${TMP_DATA_DIR}
unzip ${DATA_DIR}/${DATASET}.zip -d ${TMP_DATA_DIR}
echo "__________________________["
echo "Dataset prepared."
echo "__________________________]"

# Run command
cd ${CODE_PATH}
if [ "$SWEEP_MODE" = "launch" ]; then
    python -m domainbed.scripts.sweep ${SWEEP_MODE}\
        --data_dir=${TMP_DATA_DIR} \
        --output_dir=${OUT_PATH}/${SWEEP_NAME} \
        --command_launcher ${CMD_LAUNCH} \
        --algorithms ${DG_ALGO} \
        --datasets ${DATASET} \
        --n_hparams 1 \
        --n_trials_from ${N_TRIALS_FROM} \
        --n_trials ${N_TRIALS} \
        --single_test_envs \
        --hparams "{\"backbone\": \"${BACKBONE}\", \"batch_size\": 32}" \
        --wandb_name ${SWEEP_NAME} \
        --skip_confirmation

    ###############################################
    # Uncomment below only when trying new DG algos
    ###############################################
    # SWEEP_MODE=unsupervised_adaptation
    # python -m domainbed.scripts.sweep ${SWEEP_MODE}\
    #     --data_dir=${TMP_DATA_DIR} \
    #     --output_dir=${OUT_PATH}/${SWEEP_NAME} \
    #     --command_launcher ${CMD_LAUNCH} \
    #     --algorithms ${DG_ALGO} \
    #     --datasets ${DATASET} \
    #     --n_hparams 1 \
    #     --n_trials_from ${N_TRIALS_FROM} \
    #     --n_trials ${N_TRIALS} \
    #     --single_test_envs \
    #     --hparams "{\"backbone\": \"${BACKBONE}\"}" \
    #     --wandb_name ${SWEEP_NAME} \
    #     --skip_confirmation #\
    #     --adapt_algorithms ${TTA_ALGOS}
    ###############################################
elif [ "$SWEEP_MODE" = "unsupervised_adaptation" ]; then
    python -m domainbed.scripts.sweep ${SWEEP_MODE}\
        --data_dir=${TMP_DATA_DIR} \
        --output_dir=${OUT_PATH}/${SWEEP_NAME} \
        --command_launcher ${CMD_LAUNCH} \
        --algorithms ${DG_ALGO} \
        --datasets ${DATASET} \
        --n_hparams 1 \
        --n_trials_from ${N_TRIALS_FROM} \
        --n_trials ${N_TRIALS} \
        --single_test_envs \
        --hparams "{\"backbone\": \"${BACKBONE}\", \"batch_size\": 32}" \
        --skip_confirmation \
        --adapt_algorithms ${TTA_ALGOS} \
        --find_algos ${FIND_ALGOS} \
        --fix_algos ${FIX_ALGOS} \
        --granularities ${GRANULARITIES}
        # --wandb_name ${SWEEP_NAME}

elif [ "$SWEEP_MODE" = "supervised_adaptation" ]; then
    python -m domainbed.scripts.sweep ${SWEEP_MODE}\
    --data_dir=${TMP_DATA_DIR} \
    --output_dir=${OUT_PATH}/${SWEEP_NAME} \
    --command_launcher ${CMD_LAUNCH} \
    --algorithms ${DG_ALGO} \
    --datasets ${DATASET} \
    --n_hparams 1 \
    --n_trials_from ${N_TRIALS_FROM} \
    --n_trials ${N_TRIALS} \
    --single_test_envs \
    --hparams "{\"backbone\": \"${BACKBONE}\", \"batch_size\": 32}" \
    --skip_confirmation
    # --wandb_name ${SWEEP_NAME}
fi

# Collect results
# python -m domainbed.scripts.collect_results \
    #    --input_dir=${OUT_PATH}/${SWEEP_NAME} #\
    #    --adapt_algorithms ${TTA_ALGOS}


echo "__________________________["
echo "cc_run_sweep.sh finished."
echo "__________________________]"
