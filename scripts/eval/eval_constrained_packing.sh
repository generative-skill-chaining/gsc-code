#!/bin/bash

set -e

function run_cmd {
    echo ""
    echo "${CMD}"
    ${CMD}
}

function eval_tamp_diffusion {
    args=""
    args="${args} --env-config ${ENV_CONFIG}"
    if [ ${#POLICY_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS[@]}"
    fi
    if [ ${#DIFFUSION_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --diffusion-checkpoints ${DIFFUSION_CHECKPOINTS[@]}"
    fi
    args="${args} --seed ${SEED}"
    args="${args} --max-depth 4"
    args="${args} --timeout 10"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --num-eval 10"
        args="${args} --path ${PLANNER_OUTPUT_PATH}_debug"
        args="${args} --verbose 1"
    else
        args="${args} --num-eval 50"
        args="${args} --path ${PLANNER_OUTPUT_PATH}"
        args="${args} --verbose 0"
    fi
    CMD="python scripts/eval/eval_constrained_packing.py ${args}"
    run_cmd
}

function run_planners {
    for planner in "${PLANNERS[@]}"; do

        POLICY_CHECKPOINTS=()
        for policy_env in "${POLICY_ENVS[@]}"; do
            POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${policy_env}/${CKPT}.pt")
        done

        DIFFUSION_CHECKPOINTS=()
        for policy_env in "${POLICY_ENVS[@]}"; do
            DIFFUSION_CHECKPOINTS+=("diffusion_models/${exp_name}/${policy_env}/")
        done

        eval_tamp_diffusion
    done
}

SEED=100

# Setup.
DEBUG=0
input_path="models"
output_path="plots"

# Evaluate planners.
PLANNERS=(
    "diffusion"
)

# Experiments.

# Pybullet.
exp_name="official"
PLANNER_CONFIG_PATH="configs/pybullet/planners"
ENVS=(
    "constrained_packing/task0"
)
POLICY_ENVS=("pick" "place" "pull" "push")
CKPT="best_model"
ENV_KWARGS="--closed-loop 1"

# Run planners.
POLICY_INPUT_PATH="${input_path}/${exp_name}"

for env in "${ENVS[@]}"; do
    ENV_CONFIG="configs/pybullet/envs/official/domains/${env}.yaml"
    PLANNER_OUTPUT_PATH="${output_path}/${exp_name}/tamp_experiment/${env}"
    run_planners
done