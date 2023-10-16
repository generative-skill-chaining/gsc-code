#!/bin/bash

set -e

function run_cmd {
    echo ""
    echo "${CMD}"
    ${CMD}
}

function eval_diffusion {
    args=""
    args="${args} --checkpoint ${POLICY_CHECKPOINT}"
    args="${args} --diffusion-checkpoint ${DIFFUSION_CHECKPOINT}"
    args="${args} --env-config ${ENV_CONFIG}"
    args="${args} --seed ${SEED}"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path plots/${EXP_NAME}_debug"
        args="${args} --num-episodes 1"
        args="${args} --verbose 1"
    else
        args="${args} --path plots/${EXP_NAME}"
        args="${args} --verbose 0"
        args="${args} --num-episodes ${NUM_EPISODES}"
    fi
    if [[ -n "${DEBUG_RESULTS}" ]]; then
        args="${args} --debug-results ${DEBUG_RESULTS}"
    fi
    CMD="python scripts/eval/eval_diffusion_transformer.py ${args}"
    run_cmd
}

# Setup.

DEBUG=0
NUM_EPISODES=5

# Evaluate policies.

SEED=0

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
            DIFFUSION_CHECKPOINT="diffusion_models/${exp_name}/${policy_env}/"
            ENV_CONFIG="configs/pybullet/envs/official/primitives/${policy_env}_eval.yaml"
            eval_diffusion
        done
    done
done
