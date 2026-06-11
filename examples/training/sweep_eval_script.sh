#!/bin/bash

CAP=/data/user_data/vijayv/tir5/checkpoints/Qwen3-1.7B-Base_capybara
TASK=/data/user_data/vijayv/tir5/forger_training_runs_capybara_low_lr_6_epochs/basic_math/Qwen_Qwen3-1.7B-Base/base_task
CAPTASK=/data/user_data/vijayv/tir5/forger_training_runs_capybara_low_lr_6_epochs/basic_math/Qwen_Qwen3-1.7B-Base/capybara_task
PY=/home/vijayv/miniconda3/envs/trl/bin/python

BASE_MODEL="Qwen/Qwen3-1.7B-Base"  # or "Qwen/Qwen3-1.7B-Instruct"
$PY trajectory.py $BASE_MODEL "base"


mapfile -t C < <(ls -d $CAP/checkpoint-* | sed 's/.*checkpoint-//' | sort -n)
mapfile -t T < <(ls -d $TASK/checkpoint-* | sed 's/.*checkpoint-//' | sort -n)
mapfile -t P < <(ls -d $CAPTASK/checkpoint-* | sed 's/.*checkpoint-//' | sort -n)
echo "cap mapfile ${#C[@]}"
echo "task mapfile ${#T[@]}"
echo "captask mapfile ${#P[@]}"
for i in "${!C[@]}"; do $PY trajectory.py "$CAP/checkpoint-${C[$i]}" "capyLO-ep$((i+1))"; done
for i in "${!P[@]}"; do $PY trajectory.py "$CAPTASK/checkpoint-${P[$i]}" "capytaskLO-ep$((i+1))"; done
for i in "${!T[@]}"; do $PY trajectory.py "$TASK/checkpoint-${T[$i]}" "taskLO-ep$((i+1))"; done
