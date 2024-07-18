#!/bin/bash
#SBATCH --gres=gpu:2            # Request GPU "generic resources"
#SBATCH --cpus-per-task=16      # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=64000M            # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-06:00:00       # DD-HH:MM:SS
#SBATCH --array=0-23

# sbatch --dependency=afterok:<prev_job_id> cc_multiple_sweep.sh
nvidia-smi
# -----------------------------------------------------------------------------------
# Define run parameters as bash arrays
# -----------------------------------------------------------------------------------
# SWEEP_MODE=launch
SWEEP_MODE=unsupervised_adaptation

DATASETS=(PACS VLCS)
BACKBONES=(resnet18-BN)
DG_ALGOS=(ERM)
MARKS=(0)

if [ "$SWEEP_MODE" = "unsupervised_adaptation" ]; then
    TTA_ALGOS=(PseudoLabel SHOT)

    # 6 find algos
    # ~max time for heuristics: 6h per res18-PACS, x2 for other dataset or larger backbone
    FIND_ALGOS=(DomainGen                   None
                Autorgn
                Png                         Pngmultismooth
                UpdateMask_init:pngmulti)
    FIX_ALGOS=(Difflr)
    GRANULARITIES=(Block)

    # FIND_ALGOS=(Best Bestmulti)
    # FIX_ALGOS=(Trainsubset)
    # GRANULARITIES=(Block Subblock)

    # FIND_ALGOS=(None)
    # FIX_ALGOS=(None)
    # GRANULARITIES=(None)

    # DomainGen None Autorgn Rgn Rgnbin Oursactgap Png Pngmultisharp Pngmultismooth PostUpdate PostUpdate_PrevX PostUpdate_TotalLoss PostUpdate_PrevX_TotalLoss
    # 33 find algos : pretty extensive sweep for UpdateMask
    # ~max time for updatemask: 12h per res18-PACS, x2 for other dataset or larger backbone
    # FIND_ALGOS=(GradMask UpdateMask
    #             UpdateMask_init:curr UpdateMask_init:rgn UpdateMask_init:rgnbin UpdateMask_init:png UpdateMask_init:pngmulti
    #             UpdateMask_finit:0.5 UpdateMask_finit:1 UpdateMask_finit:rand
    #             UpdateMask_appl:prod UpdateMask_appl:avg
    #             UpdateMask_appl:ema_emalr:0.3 UpdateMask_appl:ema_emalr:0.5 UpdateMask_appl:ema_emalr:0.7
    #             UpdateMask_lr:0.1 UpdateMask_lr:0.01
    #             UpdateMask_steps:10 UpdateMask_steps:100 UpdateMask_steps:200
    #             UpdateMask_patience:1 UpdateMask_patience:5 UpdateMask_patience:10
    #             UpdateMask_loss:curr
    #             UpdateMask_reg:l1_regstr:1 UpdateMask_reg:l1_regstr:0.1 UpdateMask_reg:l1_regstr:0.01
    #             UpdateMask_reg:l2_regstr:1 UpdateMask_reg:l2_regstr:0.1 UpdateMask_reg:l2_regstr:0.01
    #             UpdateMask_bnd:none UpdateMask_bnd:scale
    #             UpdateMask_bin:1)
    # 27 find algos : imp only sweep for UpdateMask
    # FIND_ALGOS=(DomainGen                   None
    #             Autorgn                     Rgn                             Rgnbin
    #             Png                         Pngmultismooth                  
    #             UpdateMask
    #             UpdateMask_init:prev        UpdateMask_init:rgnbin
    #             UpdateMask_init:png         UpdateMask_init:pngmulti
    #             UpdateMask_init:curr        UpdateMask_init:curr_finit:0    UpdateMask_init:curr_finit:1
    #             UpdateMask_appl:curr        UpdateMask_appl:ema_emalr:0.5
    #             UpdateMask_lr:1             UpdateMask_lr:0.0001
    #             UpdateMask_steps:5          UpdateMask_steps:500
    #             UpdateMask_reg:none         UpdateMask_reg:l1_regstr:1      UpdateMask_reg:l1_regstr:0.0001
    #             UpdateMask_reg:l2_regstr:1  UpdateMask_reg:l2_regstr:0.0001
    #             UpdateMask_bnd:none)
    # FIND_ALGOS=(Layer1 Layer2 Layer3 Layer4)
    # 9 find algos : v imp only sweep for UpdateMask
    # FIND_ALGOS=(UpdateMask_init:prev        UpdateMask_init:pngmulti
    #             UpdateMask_init:curr_finit:0
    #             UpdateMask_appl:curr
    #             UpdateMask_lr:1
    #             UpdateMask_steps:5
    #             UpdateMask_reg:none         UpdateMask_reg:l1_regstr:1
    #             UpdateMask_reg:l2_regstr:0.01)
    # FIX_ALGOS=(Difflr)
    # GRANULARITIES=(Block)
fi
# -----------------------------------------------------------------------------------
# Define run identifiers
# -----------------------------------------------------------------------------------
N_TRIALS_FROM=10
N_TRIALS=1
# -----------------------------------------------------------------------------------
# Get run parameters for each array task id
# Also set n_arrays time
# -----------------------------------------------------------------------------------
# Calculate the number of hyperparameters and run times
n_datasets=${#DATASETS[@]}
n_backbones=${#BACKBONES[@]}
n_dg_algos=${#DG_ALGOS[@]}
n_marks=${#MARKS[@]}
n_total=$((n_datasets * n_backbones * n_dg_algos * n_marks))
if [ "$SWEEP_MODE" = "unsupervised_adaptation" ]; then
    n_tta_algos=${#TTA_ALGOS[@]}
    n_find_algos=${#FIND_ALGOS[@]}
    n_fix_algos=${#FIX_ALGOS[@]}
    n_granularities=${#GRANULARITIES[@]}
    n_total=$((n_total * n_tta_algos * n_find_algos * n_fix_algos * n_granularities))
fi

# Set the total_job array

# Calculate the indices of the hyperparameters and run time to use for this job
i_datasets=$((SLURM_ARRAY_TASK_ID % n_datasets))
i_backbones=$(((SLURM_ARRAY_TASK_ID / n_datasets) % n_backbones))
i_dg_algos=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones)) % n_dg_algos))
i_marks=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos)) % n_marks))
if [ "$SWEEP_MODE" = "unsupervised_adaptation" ]; then
    i_tta_algos=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos * n_marks)) % n_tta_algos))
    i_find_algos=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos * n_marks * n_tta_algos)) % n_find_algos))
    i_fix_algos=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos * n_marks * n_tta_algos * n_find_algos)) % n_fix_algos))
    i_granularities=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos * n_marks * n_tta_algos * n_find_algos * n_fix_algos)) % n_granularities))
fi

# Use the indices to select the corresponding hyperparameters and run time
DATASET=${DATASETS[$i_datasets]}
BACKBONE=${BACKBONES[$i_backbones]}
DG_ALGO=${DG_ALGOS[$i_dg_algos]}
MARK=${MARKS[$i_marks]}
if [ "$SWEEP_MODE" = "unsupervised_adaptation" ]; then
    TTA_ALGO=${TTA_ALGOS[$i_tta_algos]}
    FIND_ALGO=${FIND_ALGOS[$i_find_algos]}
    FIX_ALGO=${FIX_ALGOS[$i_fix_algos]}
    GRANULARITY=${GRANULARITIES[$i_granularities]}
fi
N_TRIALS_FROM=$((N_TRIALS_FROM + MARK))

# -----------------------------------------------------------------------------------
# Set sweep folder names and paths
# -----------------------------------------------------------------------------------
SWEEP_NAME=${DATASET}_${BACKBONE}_${DG_ALGO}_${MARK}
CMD_LAUNCH=multi_gpu

HOME_PATH=/home/saby
DATA_DIR=${HOME_PATH}/scratch/data
OUT_PATH=${HOME_PATH}/scratch/out/stta/sweep
CODE_PATH=${HOME_PATH}/scratch/code/ttdg
ENVS_PATH=${HOME_PATH}/envs/ttdg

# -----------------------------------------------------------------------------------
# Documentation for parameters above
# -----------------------------------------------------------------------------------
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
# TTA_ALGOS=(PseudoLabel T3A SHOT TentFull PLClf TentClf Rotn_TTA Rotn_T3A_TTA SimCLR_TTA SimCLR_T3A_TTA)
#            (empty string implies all TTA algorithms mentioned in sweep)
# FIND_ALGOS=(None Best)
# FIX_ALGOS=(None Trainsubset)
# GRANULARITIES=(None Block Subblock Layer)
# MARK : to identify the run type (like TRIAL_ID)
# N_TRIALS_FROM : 10 (fixed as per paper) 
# N_TRIALS : 1, 3 (1 used for quick experiments, 3 used in paper)
# SWEEP_NAME : is set automatically (modify this only if using multiple dg algos or datasets)
# CMD_LAUNCH : multi_gpu, local, dummy
# DATA_DIR : path to parent folder of datasets or zip file
# OUT_PATH : path to parent sweep folder
# -----------------------------------------------------------------------------------
# Begin...
# -----------------------------------------------------------------------------------
echo "__________________________["
echo "run_sweep.sh args received:"
echo ""
echo "total jobs should be: ${n_total}"
echo "sweep_mode: ${SWEEP_MODE}"
echo "dataset: ${DATASET}"
echo "backbone: ${BACKBONE}"
echo "dg algorithm: ${DG_ALGO}"
if [ "$SWEEP_MODE" = "unsupervised_adaptation" ]; then
    echo "tta algorithm: ${TTA_ALGO}"
    echo "find algorithm: ${FIND_ALGO}"
    echo "fix algorithm: ${FIX_ALGO}"
    echo "granularitie: ${GRANULARITY}"
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
        --adapt_algorithms ${TTA_ALGO} \
        --find_algos ${FIND_ALGO} \
        --fix_algos ${FIX_ALGO} \
        --granularities ${GRANULARITY}
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
