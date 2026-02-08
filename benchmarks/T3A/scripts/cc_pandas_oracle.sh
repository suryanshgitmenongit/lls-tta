#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000M
#SBATCH --time=0-01:30:00
#SBATCH --output=/home/saby/scratch/Suryansh/out/estta/sweep/pandas_oracle_%j.out
#SBATCH --error=/home/saby/scratch/Suryansh/out/estta/sweep/pandas_oracle_%j.err

################################################################################
# Oracle Hyperparameter Selection - Results Aggregation Script
################################################################################
#
# This script aggregates oracle hyperparameter selection results for LEAST/LLS
# using SEED-BASED VARIANCE (standard domain generalization practice).
#
# USAGE:
#   sbatch cc_pandas_oracle.sh
#
# CONFIGURATION:
#   Edit the variables in the configuration section below
#
################################################################################

echo "================================================================================"
echo "ORACLE HP SELECTION - RESULTS AGGREGATION"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

#-------------------------------------------------------------------------------
# CONFIGURATION SECTION - EDIT THESE VARIABLES
#-------------------------------------------------------------------------------

# Base directory containing all sweep results
INPUT_DIR="/home/saby/scratch/Suryansh/out/estta/sweep"

# HP selection mode: "Orc" (Oracle), "Iid" (IID validation), or "Both"
HP_MODE="Both"

# TDB mode: "true" for Tiny DomainBed (6 envs), "false" for Full DB (16 envs)
TDB_MODE="false"

# Output file prefix (will be appended with exp_id, hp_mode and tdb/fulldb suffix)
OUTPUT_PREFIX="7Feb_oracle_aggregation"

# Experiment IDs to process (space-separated list)
# Leave EMPTY to process ALL exp_ids found in data
# Use "none" to process only results without exp_id
# Examples:
#   EXP_IDS=""                                  # Process all exp_ids
#   EXP_IDS="5Feb_Run_1"                        # Process only this exp_id
#   EXP_IDS="5Feb_Run_1 5Feb_Run_2"             # Process these two exp_ids
#   EXP_IDS="none"                              # Process only old results without exp_id
EXP_IDS="5Feb_Run_1 5Feb_Run_2 5Feb_Run_3 5Feb_Run_4 "
#-------------------------------------------------------------------------------
# AUTOMATIC SWEEP_NAMES CONFIGURATION
#-------------------------------------------------------------------------------

# Generate sweep names for ALL MARKS
# We must include all random seeds (marks 0, 1, 2) to compute
# proper seed-based statistics. Each mark is an independent experimental trial.
# Without all marks, we cannot compute variance across seeds.

# Datasets based on TDB mode
if [ "${TDB_MODE}" = "true" ]; then
    DATASETS=(PACS VLCS TerraIncognita)
    echo "TDB Mode enabled - processing 3 datasets (6 environments)"
else
    DATASETS=(PACS VLCS TerraIncognita OfficeHome)
    echo "Full DB Mode enabled - processing 4 datasets (16 environments)"
fi

# All marks (random seeds)
MARKS=(0 1 2)

# Generate sweep names: dataset_backbone_algo_mark
SWEEP_NAMES=""
for dataset in "${DATASETS[@]}"; do
    for mark in "${MARKS[@]}"; do
        SWEEP_NAMES+="${dataset}_resnet18-BN_ERM_${mark} "
    done
done

echo "Generated sweep names for ${#DATASETS[@]} datasets x ${#MARKS[@]} marks = $((${#DATASETS[@]} * ${#MARKS[@]})) directories"

#-------------------------------------------------------------------------------
# PATH CONFIGURATION
#-------------------------------------------------------------------------------

HOME_PATH=/home/saby
CODE_PATH=${HOME_PATH}/scratch/Suryansh/lls-tta-main/benchmarks/T3A
ENVS_PATH=${HOME_PATH}/scratch/Suryansh/envs/ttdg
SCRIPT_PATH=${CODE_PATH}/domainbed/scripts/pandas_oracle_results.py

echo ""
echo "Configuration:"
echo "  Input Directory:  ${INPUT_DIR}"
echo "  HP Mode:          ${HP_MODE}"
echo "  TDB Mode:         ${TDB_MODE}"
echo "  Output Prefix:    ${OUTPUT_PREFIX}"
if [ -n "${EXP_IDS}" ]; then
    echo "  Filter exp_ids:   ${EXP_IDS}"
else
    echo "  Filter exp_ids:   ALL (no filter)"
fi
echo "  Sweep Names:      ${SWEEP_NAMES}"
echo ""
echo "Paths:"
echo "  Code path:        ${CODE_PATH}"
echo "  Script path:      ${SCRIPT_PATH}"
echo "  Envs path:        ${ENVS_PATH}"
echo ""

#-------------------------------------------------------------------------------
# LOAD MODULES AND ACTIVATE ENVIRONMENT
#-------------------------------------------------------------------------------

echo "Loading Compute Canada modules..."
module load StdEnv/2020
module load python/3.8.10

echo "Activating virtual environment..."
source ${ENVS_PATH}/bin/activate

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment at ${ENVS_PATH}"
    exit 1
fi

echo "Environment activated successfully."
echo "Python path: $(which python)"
echo ""

#-------------------------------------------------------------------------------
# CHANGE TO CODE DIRECTORY
#-------------------------------------------------------------------------------

cd ${CODE_PATH}
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to change to code directory: ${CODE_PATH}"
    exit 1
fi

echo "Current directory: $(pwd)"
echo ""

#-------------------------------------------------------------------------------
# VALIDATE SCRIPT EXISTS
#-------------------------------------------------------------------------------

if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: pandas_oracle_results.py not found at ${SCRIPT_PATH}"
    echo "Please ensure the script is placed in domainbed/scripts/"
    exit 1
fi

echo "Found pandas_oracle_results.py at ${SCRIPT_PATH}"
echo ""

#-------------------------------------------------------------------------------
# BUILD TDB FLAG
#-------------------------------------------------------------------------------

if [ "${TDB_MODE}" = "true" ]; then
    TDB_FLAG="--tdb_only"
else
    TDB_FLAG="--no_tdb_only"
fi

#-------------------------------------------------------------------------------
# RUN AGGREGATION SCRIPT
#-------------------------------------------------------------------------------

echo "================================================================================"
echo "RUNNING AGGREGATION"
echo "================================================================================"
echo ""

# Build command with conditional exp_ids parameter
if [ -n "${EXP_IDS}" ]; then
    python ${SCRIPT_PATH} \
        --input_dir "${INPUT_DIR}" \
        --sweep_names ${SWEEP_NAMES} \
        --output_prefix "${OUTPUT_PREFIX}" \
        --hp_mode ${HP_MODE} \
        ${TDB_FLAG} \
        --exp_ids ${EXP_IDS}
else
    python ${SCRIPT_PATH} \
        --input_dir "${INPUT_DIR}" \
        --sweep_names ${SWEEP_NAMES} \
        --output_prefix "${OUTPUT_PREFIX}" \
        --hp_mode ${HP_MODE} \
        ${TDB_FLAG}
fi

exit_code=$?

echo ""
echo "================================================================================"
if [ ${exit_code} -eq 0 ]; then
    echo "AGGREGATION COMPLETED SUCCESSFULLY"
else
    echo "AGGREGATION FAILED"
fi
echo "================================================================================"
echo "End time: $(date)"
echo "Exit code: ${exit_code}"

#-------------------------------------------------------------------------------
# DISPLAY OUTPUT FILES
#-------------------------------------------------------------------------------

if [ ${exit_code} -eq 0 ]; then
    if [ "${TDB_MODE}" = "true" ]; then
        MODE_SUFFIX="tdb"
    else
        MODE_SUFFIX="fulldb"
    fi
    
    if [ "${HP_MODE}" = "Both" ]; then
        HP_SUFFIX="both"
    else
        HP_SUFFIX="${HP_MODE,,}"
    fi
    
    # Build search pattern for output files
    # Pattern: PREFIX_*_HP_MODE.csv (where * is exp_id or "no_exp_id")
    OUTPUT_PATTERN="${INPUT_DIR}/${OUTPUT_PREFIX}_*_${HP_SUFFIX}_${MODE_SUFFIX}.csv"
    
    echo ""
    echo "Output files created:"
    
    # Find all matching files
    OUTPUT_FILES=$(ls -1 ${OUTPUT_PATTERN} 2>/dev/null)
    
    if [ -n "${OUTPUT_FILES}" ]; then
        FILE_COUNT=$(echo "${OUTPUT_FILES}" | wc -l)
        echo "  Found ${FILE_COUNT} file(s):"
        echo ""
        
        # Display each file with size
        while IFS= read -r file; do
            if [ -f "${file}" ]; then
                FILE_SIZE=$(du -h "${file}" | cut -f1)
                FILE_NAME=$(basename "${file}")
                echo "  - ${FILE_NAME}"
                echo "    Path: ${file}"
                echo "    Size: ${FILE_SIZE}"
                echo ""
            fi
        done <<< "${OUTPUT_FILES}"
    else
        echo "  WARNING: No output files found matching pattern:"
        echo "  ${OUTPUT_PATTERN}"
        echo ""
        echo "  This may indicate:"
        echo "  - No oracle_exp_id was set (old results without exp_id)"
        echo "  - All experiment IDs failed processing"
        echo "  - Requested exp_ids not found in data"
        echo "  - Output directory path is incorrect"
    fi
fi

echo ""
echo "================================================================================"

exit ${exit_code}
