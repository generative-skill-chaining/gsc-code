#!/bin/bash

set -e

function run_cmd {
    echo ""
    echo "${CMD}"
    ${CMD}
    }

function train_diffusion {
    args=""
    args="${args} --checkpoint ${POLICY_CHECKPOINT}"
    args="${args} --dataset-checkpoint ${DATASET_CHECKPOINT}"
    args="${args} --diffusion-checkpoint ${DIFFUSION_CHECKPOINT}"
    args="${args} --train-classifier"
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path plots/${EXP_NAME}_debug"
        args="${args} --num-episodes 1"
        args="${args} --verbose 1"
    else
        args="${args} --path plots/${EXP_NAME}"
        args="${args} --verbose 1"
        args="${args} --num-episodes ${NUM_EPISODES}"
    fi
    if [[ -n "${DEBUG_RESULTS}" ]]; then
        args="${args} --debug-results ${DEBUG_RESULTS}"
    fi
    if [[ -n "${ENV_CONFIG}" ]]; then
        args="${args} --env-config ${ENV_CONFIG}"
    fi
    CMD="python scripts/train/train_diffusion_transformer_w_classifier.py ${args}"
    run_cmd
}

# Setup.

DEBUG=0
NUM_EPISODES=50000

# Evaluate policies.

policy_envs=(
    "pick"
    "place"
    "pull"
    "push"
)
experiments=(
    "official"
)
ckpts=(
    "best_model"
    )

for exp_name in "${experiments[@]}"; do
    for ckpt in "${ckpts[@]}"; do
        for policy_env in "${policy_envs[@]}"; do
            EXP_NAME="${exp_name}/${ckpt}"
            POLICY_CHECKPOINT="models/${exp_name}/${policy_env}/${ckpt}.pt"
            DATASET_CHECKPOINT="datasets/${policy_env}.pkl"
            DIFFUSION_CHECKPOINT="diffusion_models/${exp_name}/${policy_env}/"
            train_diffusion
        done
    done
done
