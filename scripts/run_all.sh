#!/bin/bash
# Run every dataset x model x seed under one tag; other flags go to every run.
#   bash scripts/run_all.sh --datasets "a b c" --tag NAME [--seeds "1 2 3"] [--models "..."] [--benign-weight 5 ...]
set -e
cd "$(dirname "$0")/.."

datasets=""
tag=""
seeds="1 2 3"
models="transformer mamba2 mamba3"
extra=()

while [ $# -gt 0 ]; do
    case "$1" in
        --datasets|--dataset) datasets="$2"; shift 2 ;;
        --tag)     tag="$2";     shift 2 ;;
        --seeds)   seeds="$2";   shift 2 ;;
        --models)  models="$2";  shift 2 ;;
        *)         extra+=("$1"); shift ;;
    esac
done

if [ -z "$datasets" ] || [ -z "$tag" ]; then
    echo "usage: bash scripts/run_all.sh --datasets \"a b c\" --tag NAME [--seeds \"1 2 3\"] [--models \"...\"] [model flags]"
    exit 1
fi
if compgen -G "results/*_$tag" > /dev/null; then
    echo "Runs tagged '$tag' already exist in results/. Pick a new tag or move the old runs first."
    exit 1
fi
if ! python3 -c "import torch" 2> /dev/null; then
    echo "python3 can't import torch. Activate the venv first: source .venv/bin/activate"
    exit 1
fi
for dataset in $datasets; do
    if [ ! -f "data/processed/$dataset/meta.json" ]; then
        echo "data/processed/$dataset not found. Run first: python3 run.py prepare --dataset $dataset"
        exit 1
    fi
done

echo "datasets=[$datasets] tag=$tag models=[$models] seeds=[$seeds] extra flags=[${extra[*]}]"
for dataset in $datasets; do
    for model in $models; do
        for seed in $seeds; do
            echo
            echo "=== $dataset, $model, seed $seed ==="
            python3 run.py experiment --dataset "$dataset" --model "$model" --seed "$seed" --tag "$tag" "${extra[@]}"
        done
    done
done
echo "Done."
