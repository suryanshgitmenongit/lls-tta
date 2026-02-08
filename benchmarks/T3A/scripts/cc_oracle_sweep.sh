#!/bin/bash
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000M
#SBATCH --time=0-48:00:00
#SBATCH --array=0-23
#SBATCH --output=/home/saby/scratch/Suryansh/out/estta/sweep/oracle/slurm-%A_%a.out
#SBATCH --error=/home/saby/scratch/Suryansh/out/estta/sweep/oracle/slurm-%A_%a.err
# sbatch cc_oracle_sweep.sh
nvidia-smi

# ==============================================================================
# ORACLE HYPERPARAMETER TUNING SCRIPT
# ==============================================================================
# This script implements TRUE oracle HP selection by passing ALL find_algos
# to a SINGLE job (not splitting across array tasks).
#
# USAGE:
#   1. Paste all your layer selection configs in FIND_ALGOS array below
#   2. Set HP_MODES to "Orc" or "Iid" or "Orc Iid"
#   3. Set ORACLE_EXP_ID for this experimental run
#   4. Submit once to see array size calculation
#   5. Update #SBATCH --array line above
#   6. Submit again to run
#
# IMPORTANT: 
#   - Run LEAST configs separately from LLS configs (different submissions)
#   - All configs in FIND_ALGOS will be tested together in each job
#   - Use SAME ORACLE_EXP_ID for all scripts in one comparison study
#     (e.g., LLS_PL, LLS_SHOT, LEAST_both all use exp_id="run1")
#   - Use DIFFERENT ORACLE_EXP_ID for different experimental runs
# ==============================================================================

# ==============================================================================
# CONFIGURATION SECTION - MODIFY THIS
# ==============================================================================

# -----------------------------------------------------------------------------------
# EXPERIMENT IDENTIFIER
# -----------------------------------------------------------------------------------
# Identifier for this experimental run. Use the SAME exp_id for all scripts
# that should be compared together (LLS vs LEAST).
#
# USAGE:
#   Run 1 comparison (LLS vs LEAST with config set A):
#     - Script A: LLS on PL,   exp_id="run1"
#     - Script B: LLS on SHOT, exp_id="run1"  (SAME!)
#     - Script C: LEAST both,  exp_id="run1"  (SAME!)
#
#   Run 2 comparison (LLS vs LEAST with config set B):
#     - Script D: LLS on PL,   exp_id="run2"  (DIFFERENT!)
#     - Script E: LLS on SHOT, exp_id="run2"
#     - Script F: LEAST both,  exp_id="run2"
#
# Leave empty only if running a single isolated experiment.
ORACLE_EXP_ID="5Feb_Run_1"

# -----------------------------------------------------------------------------------
# HP TUNING MODE
# -----------------------------------------------------------------------------------
# Which HP tuning modes to run (space-separated):
# - 'Orc' = Oracle (uses target out_split for HP search)
# - 'Iid' = IID (uses source out_split for HP search)
# You can run both: HP_MODES="Orc Iid"
HP_MODES="Orc Iid"

# -----------------------------------------------------------------------------------
# LAYER SELECTION CONFIGS
# -----------------------------------------------------------------------------------
# PASTE ALL YOUR CONFIGS HERE (LEAST or LLS, not both!)
#
# For LEAST oracle run: Paste all LEAST configs
# For LLS oracle run: Delete LEAST configs, paste all LLS configs
# For baseline: Use "None"
#
FIND_ALGOS=(
       "LEAST"
)

# -----------------------------------------------------------------------------------
# EXPERIMENTAL CONFIGURATION
# -----------------------------------------------------------------------------------
TDB_MODE=false  # true = Tiny DomainBed (fast), false = Full DomainBed

SWEEP_MODE=unsupervised_adaptation

# Datasets to test
DATASETS=(PACS VLCS TerraIncognita OfficeHome)

# Model configuration
BACKBONES=(resnet18-BN)
DG_ALGOS=(ERM)

# TTA algorithms
TTA_ALGOS=(PseudoLabel SHOT)

# Fixed parameters
FIX_ALGOS=(Difflr)
GRANULARITIES=(Layer)

# Trial configuration
MARKS=(0 1 2)           # 0 = seeds 10,11,12
N_TRIALS_FROM=10    # Starting seed
N_TRIALS=1          # Number of trials (1 = quick test, 3 = paper standard)

# -----------------------------------------------------------------------------------
# PATH CONFIGURATION
# -----------------------------------------------------------------------------------
HOME_PATH=/home/saby
DATA_DIR=${HOME_PATH}/scratch/Suryansh/data
OUT_PATH=${HOME_PATH}/scratch/Suryansh/out/estta/sweep
CODE_PATH=${HOME_PATH}/scratch/Suryansh/lls-tta-main/benchmarks/T3A
ENVS_PATH=${HOME_PATH}/scratch/Suryansh/envs/ttdg

CMD_LAUNCH=local

# ==============================================================================
# END OF CONFIGURATION - DO NOT MODIFY BELOW THIS LINE
# ==============================================================================

# -----------------------------------------------------------------------------------
# Calculate array job dimensions (ORACLE MODE: don't multiply by n_find_algos)
# -----------------------------------------------------------------------------------
n_datasets=${#DATASETS[@]}
n_backbones=${#BACKBONES[@]}
n_dg_algos=${#DG_ALGOS[@]}
n_marks=${#MARKS[@]}
n_tta_algos=${#TTA_ALGOS[@]}
n_fix_algos=${#FIX_ALGOS[@]}
n_granularities=${#GRANULARITIES[@]}
n_find_algos=${#FIND_ALGOS[@]}

# CRITICAL: All find_algos go to ONE job (not split across array tasks)
n_total=$((n_datasets * n_backbones * n_dg_algos * n_marks * n_tta_algos * n_fix_algos * n_granularities))

echo "=============================================================================="
echo "ORACLE SWEEP CONFIGURATION"
echo "=============================================================================="
if [ -n "${ORACLE_EXP_ID}" ]; then
    echo "Experiment ID: ${ORACLE_EXP_ID}"
else
    echo "Experiment ID: (none)"
fi
echo "HP Modes: ${HP_MODES}"
echo "TDB Mode: ${TDB_MODE}"
echo ""
echo "Experimental Configuration:"
echo "  Datasets: ${n_datasets} (${DATASETS[@]})"
echo "  Backbones: ${n_backbones} (${BACKBONES[@]})"
echo "  DG Algos: ${n_dg_algos} (${DG_ALGOS[@]})"
echo "  TTA Algos: ${n_tta_algos} (${TTA_ALGOS[@]})"
echo "  Marks: ${n_marks} (${MARKS[@]})"
echo "  Fix Algos: ${n_fix_algos} (${FIX_ALGOS[@]})"
echo "  Granularities: ${n_granularities} (${GRANULARITIES[@]})"
echo ""
echo "Layer Selection Configs (ALL passed to each job):"
echo "  Number of configs: ${n_find_algos}"
for i in "${!FIND_ALGOS[@]}"; do
    echo "    [$i] ${FIND_ALGOS[$i]}"
done
echo ""
echo "Array Job Calculation:"
echo "  Total array jobs: ${n_total}"
echo "  Update line 6 to: #SBATCH --array=0-$((n_total - 1))"
echo ""
if [ $n_total -ne $((SLURM_ARRAY_TASK_COUNT)) ]; then
    echo "WARNING: Array size mismatch!"
    echo "  Expected: ${n_total} jobs"
    echo "  Current:  ${SLURM_ARRAY_TASK_COUNT} jobs"
    echo "  Please update #SBATCH --array=0-$((n_total - 1))"
fi
echo "=============================================================================="

# -----------------------------------------------------------------------------------
# Calculate indices for this array task (NO i_find_algos calculation!)
# -----------------------------------------------------------------------------------
i_datasets=$((SLURM_ARRAY_TASK_ID % n_datasets))
i_backbones=$(((SLURM_ARRAY_TASK_ID / n_datasets) % n_backbones))
i_dg_algos=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones)) % n_dg_algos))
i_marks=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos)) % n_marks))
i_tta_algos=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos * n_marks)) % n_tta_algos))
i_fix_algos=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos * n_marks * n_tta_algos)) % n_fix_algos))
i_granularities=$(((SLURM_ARRAY_TASK_ID / (n_datasets * n_backbones * n_dg_algos * n_marks * n_tta_algos * n_fix_algos)) % n_granularities))

# Select parameters for this job
DATASET=${DATASETS[$i_datasets]}
BACKBONE=${BACKBONES[$i_backbones]}
DG_ALGO=${DG_ALGOS[$i_dg_algos]}
MARK=${MARKS[$i_marks]}
TTA_ALGO=${TTA_ALGOS[$i_tta_algos]}
FIX_ALGO=${FIX_ALGOS[$i_fix_algos]}
GRANULARITY=${GRANULARITIES[$i_granularities]}
N_TRIALS_FROM=$((N_TRIALS_FROM + MARK))

SWEEP_NAME=${DATASET}_${BACKBONE}_${DG_ALGO}_${MARK}

# -----------------------------------------------------------------------------------
# Print job configuration
# -----------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} CONFIGURATION"
echo "=============================================================================="
echo "Dataset: ${DATASET}"
echo "Backbone: ${BACKBONE}"
echo "DG Algorithm: ${DG_ALGO}"
echo "TTA Algorithm: ${TTA_ALGO}"
echo "Fix Algorithm: ${FIX_ALGO}"
echo "Granularity: ${GRANULARITY}"
echo "Mark: ${MARK}"
echo "Trials: ${N_TRIALS_FROM} to $((N_TRIALS_FROM + N_TRIALS - 1))"
echo ""
echo "ALL ${n_find_algos} layer selection configs will be tested:"
for i in "${!FIND_ALGOS[@]}"; do
    echo "  [$i] ${FIND_ALGOS[$i]}"
done
echo ""
echo "Sweep name: ${SWEEP_NAME}"
echo "Output dir: ${OUT_PATH}/${SWEEP_NAME}"
if [ -n "${ORACLE_EXP_ID}" ]; then
    echo "Experiment ID: ${ORACLE_EXP_ID}"
fi
echo "=============================================================================="

# -----------------------------------------------------------------------------------
# Setup environment
# -----------------------------------------------------------------------------------
echo ""
echo "Setting up environment..."
module load StdEnv/2020
module load python/3.8.10
source ${ENVS_PATH}/bin/activate
echo "Environment activated"

# -----------------------------------------------------------------------------------
# Prepare dataset
# -----------------------------------------------------------------------------------
echo ""
echo "Preparing dataset..."
TMP_DATA_DIR=${SLURM_TMPDIR}/data_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
mkdir -p ${TMP_DATA_DIR}
cp ${DATA_DIR}/${DATASET}.zip ${TMP_DATA_DIR}
unzip -q ${DATA_DIR}/${DATASET}.zip -d ${TMP_DATA_DIR}
echo "Dataset prepared in ${TMP_DATA_DIR}"

# Set TDB environment variable
export TDB_ONLY=${TDB_MODE}
echo "TDB_ONLY set to: ${TDB_ONLY}"

# -----------------------------------------------------------------------------------
# Fix MIG UUID for PyTorch
# -----------------------------------------------------------------------------------
if [[ "${CUDA_VISIBLE_DEVICES}" == MIG-* ]]; then
    echo ""
    echo "Fixing MIG UUID for PyTorch compatibility..."
    echo "  Original: ${CUDA_VISIBLE_DEVICES}"
    export CUDA_VISIBLE_DEVICES=0
    echo "  Fixed to: ${CUDA_VISIBLE_DEVICES}"
fi

# -----------------------------------------------------------------------------------
# Run oracle sweep for each HP mode
# -----------------------------------------------------------------------------------
cd ${CODE_PATH}

for HP_MODE in ${HP_MODES}; do
    echo ""
    echo "=============================================================================="
    echo "RUNNING HP MODE: ${HP_MODE}"
    echo "=============================================================================="
    echo "This job will:"
    echo "  1. Test all ${n_find_algos} configs on out_split (20%)"
    echo "  2. Select best config based on out_split accuracy"
    echo "  3. Evaluate best config on in_split (80%)"
    echo "  4. Save results with ${HP_MODE} suffix"
    echo ""
    
    if [ -n "${ORACLE_EXP_ID}" ]; then
        echo "Using experiment ID: ${ORACLE_EXP_ID}"
        
        python -m domainbed.scripts.sweep ${SWEEP_MODE} \
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
            --find_algos "${FIND_ALGOS[@]}" \
            --fix_algos ${FIX_ALGO} \
            --granularities ${GRANULARITY} \
            --best_hp ${HP_MODE} \
            --oracle_exp_id ${ORACLE_EXP_ID}
    else
        echo "No experiment ID specified"
        
        python -m domainbed.scripts.sweep ${SWEEP_MODE} \
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
            --find_algos "${FIND_ALGOS[@]}" \
            --fix_algos ${FIX_ALGO} \
            --granularities ${GRANULARITY} \
            --best_hp ${HP_MODE}
    fi
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "HP mode ${HP_MODE} completed successfully"
    else
        echo ""
        echo "HP mode ${HP_MODE} failed with exit code ${EXIT_CODE}"
    fi
    echo "=============================================================================="
done

echo ""
echo "=============================================================================="
echo "JOB COMPLETED"
echo "=============================================================================="
echo "HP modes run: ${HP_MODES}"
echo "Configs tested: ${n_find_algos}"
echo "Check output files in: ${OUT_PATH}/${SWEEP_NAME}"
echo "=============================================================================="
