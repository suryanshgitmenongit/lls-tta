#!/bin/bash
#SBATCH --partition=workq              # Partition to submit to
#SBATCH --time=90:00:00               # Walltime (48 hours for LLS HP ablation)
#SBATCH --gres=gpu:2                # Request 1 GPU
#SBATCH --nodelist=asaicomputenode03  # Specify node
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=32G                     # Total memory per node
#SBATCH --array=0-8                   # 9 jobs: 3 TDB datasets × 3 emalr configs
#SBATCH --output=/dist_home/suryansh/layer-selection-tta/scratch/out/stta/sweep/slurm_%A_%a.out
#SBATCH --error=/dist_home/suryansh/layer-selection-tta/scratch/out/stta/sweep/slurm_%A_%a.err

# sbatch asai_multiple_sweep.sh
nvidia-smi

# -----------------------------------------------------------------------------------
# Define run parameters as bash arrays for emalr hyperparameter ablation
# -----------------------------------------------------------------------------------
SWEEP_MODE=unsupervised_adaptation

# TDB (Tiny DomainBed) datasets for focused ablation
DATASETS=(PACS VLCS TerraIncognita)
BACKBONES=(resnet18-BN)
DG_ALGOS=(ERM)
MARKS=(0)

# emalr configurations - only test emalr values with appl:ema as default
FIND_ALGOS=(
    LLS_emalr:0.3
    LLS_emalr:0.5
    LLS_emalr:0.7
)

# Fixed parameters for controlled ablation
TTA_ALGOS=(PseudoLabel)   # Focus on one TTA method
FIX_ALGOS=(Difflr)        # Fixed layer selection method
GRANULARITIES=(Layer)     # Fixed granularity

# -----------------------------------------------------------------------------------
# Define run identifiers
# -----------------------------------------------------------------------------------
N_TRIALS_FROM=10
N_TRIALS=1

# -----------------------------------------------------------------------------------
# Calculate combinatorial parameters and validate array size
# -----------------------------------------------------------------------------------
n_datasets=${#DATASETS[@]}        # n_datasets = 3
n_backbones=${#BACKBONES[@]}      # n_backbones = 1
n_dg_algos=${#DG_ALGOS[@]}        # n_dg_algos = 1
n_marks=${#MARKS[@]}              # n_marks = 1
n_tta_algos=${#TTA_ALGOS[@]}      # n_tta_algos = 1
n_find_algos=${#FIND_ALGOS[@]}    # n_find_algos = 3
n_fix_algos=${#FIX_ALGOS[@]}      # n_fix_algos = 1
n_granularities=${#GRANULARITIES[@]}  # n_granularities = 1

n_total=$((n_datasets * n_backbones * n_dg_algos * n_marks * n_tta_algos * n_find_algos * n_fix_algos * n_granularities))
# n_total = 3 × 1 × 1 × 1 × 1 × 3 × 1 × 1 = 9

echo "Total expected combinations: ${n_total}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"

# Validate array bounds
if [ ${SLURM_ARRAY_TASK_ID} -ge ${n_total} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) exceeds total combinations (${n_total})"
    exit 1
fi

# -----------------------------------------------------------------------------------
# Map SLURM_ARRAY_TASK_ID to specific parameter combination using modular arithmetic
# -----------------------------------------------------------------------------------
i_datasets=$((SLURM_ARRAY_TASK_ID % n_datasets))
i_backbones=$(((SLURM_ARRAY_TASK_ID / n_datasets) % n_backbones))
i_dg_algos=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones)) % n_dg_algos))
i_marks=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos)) % n_marks))
i_tta_algos=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos * n_marks)) % n_tta_algos))
i_find_algos=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos * n_marks * n_tta_algos)) % n_find_algos))
i_fix_algos=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos * n_marks * n_tta_algos * n_find_algos)) % n_fix_algos))
i_granularities=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos * n_marks * n_tta_algos * n_find_algos * n_fix_algos)) % n_granularities))

# Select parameters based on calculated indices
DATASET=${DATASETS[$i_datasets]}
BACKBONE=${BACKBONES[$i_backbones]}
DG_ALGO=${DG_ALGOS[$i_dg_algos]}
MARK=${MARKS[$i_marks]}
TTA_ALGO=${TTA_ALGOS[$i_tta_algos]}
FIND_ALGO=${FIND_ALGOS[$i_find_algos]}
FIX_ALGO=${FIX_ALGOS[$i_fix_algos]}
GRANULARITY=${GRANULARITIES[$i_granularities]}

N_TRIALS_FROM=$((N_TRIALS_FROM + MARK))

# -----------------------------------------------------------------------------------
# Set sweep folder names and paths (ASAI-specific)
# -----------------------------------------------------------------------------------
SWEEP_NAME=${DATASET}_${BACKBONE}_${DG_ALGO}_${MARK}
CMD_LAUNCH=local  # ASAI uses local launcher

HOME_PATH=/dist_home/suryansh/layer-selection-tta
DATA_DIR=${HOME_PATH}/scratch/data
OUT_PATH=${HOME_PATH}/scratch/out/stta/sweep
CODE_PATH=${HOME_PATH}/code/ttdg
ENVS_PATH=${HOME_PATH}/envs/ttdg

# -----------------------------------------------------------------------------------
# Enable TDB (Tiny DomainBed) mode for focused ablation
# -----------------------------------------------------------------------------------
export TDB_ONLY=true

# -----------------------------------------------------------------------------------
# Display configuration for this array task
# -----------------------------------------------------------------------------------
echo "__________________________["
echo "emalr Hyperparameter Ablation Configuration:"
echo ""
echo "total combinations: ${n_total}"
echo "array task id: ${SLURM_ARRAY_TASK_ID}"
echo "sweep_mode: ${SWEEP_MODE}"
echo "dataset: ${DATASET}"
echo "backbone: ${BACKBONE}"
echo "dg algorithm: ${DG_ALGO}"
echo "tta algorithm: ${TTA_ALGO}"
echo "find algorithm: ${FIND_ALGO}"
echo "fix algorithm: ${FIX_ALGO}"
echo "granularity: ${GRANULARITY}"
echo "mark: ${MARK}"
echo "TDB mode: ${TDB_ONLY}"
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

# -----------------------------------------------------------------------------------
# Prepare virtualenv (based on asai_run_sweep.sh)
# -----------------------------------------------------------------------------------
# Disable pyenv
export PYENV_VERSION=
export PATH=$(echo $PATH | tr ":" "\n" | grep -v "pyenv" | tr "\n" ":")

# Setup Python environment
source /dist_home/suryansh/miniforge3/etc/profile.d/conda.sh
conda activate es-tta

# Add to PYTHONPATH
export PYTHONPATH="${CODE_PATH}:${PYTHONPATH}"

echo "__________________________["
echo "Environment activated."
echo "__________________________]"

# Verify Python environment
PYTHON_PATH=$(which python)
echo "Using Python: $PYTHON_PATH"
$PYTHON_PATH --version

# -----------------------------------------------------------------------------------
# Change to code directory and run sweep
# -----------------------------------------------------------------------------------
cd "${CODE_PATH}"
if [ $? -ne 0 ]; then
    echo "Failed to change to code directory: ${CODE_PATH}"
    exit 1
fi

echo "Current directory: $(pwd)"
echo "__________________________["
echo "Running emalr hyperparameter sweep..."
echo "__________________________]"

# Run the sweep with emalr configuration
python -m domainbed.scripts.sweep ${SWEEP_MODE} \
    --data_dir=${DATA_DIR} \
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

# Check execution status
if [ $? -eq 0 ]; then
    echo "__________________________["
    echo "emalr hyperparameter ablation completed successfully"
    echo "Configuration: ${FIND_ALGO} on ${DATASET}"
    echo "Results saved to: ${OUT_PATH}/${SWEEP_NAME}"
    echo "__________________________]"
else
    echo "__________________________["
    echo "ERROR: emalr hyperparameter ablation failed"
    echo "Configuration: ${FIND_ALGO} on ${DATASET}"
    echo "Check logs for details"
    echo "__________________________]"
    exit 1
fi
