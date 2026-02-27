 #!/bin/bash
export LMUData="/mnt/public/usr/sunzhichao/benchmark/bp_refcoco"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

MODELS=(
    "InternVL2_5-8B_ivcp"
)


datasets=(
    "RefCOCO_testA"
    "RefCOCO_testB"
    "RefCOCO_val"
    "RefCOCO+_testA"
    "RefCOCO+_testB"
    "RefCOCO+_val"
    "RefCOCOg_test"
    "RefCOCOg_val"
)

for MODEL in "${MODELS[@]}"; do
    for dataset in "${datasets[@]}"; do

        echo "====================================================="
        echo "开始评估模型: $MODEL 数据集: $dataset"
        echo "====================================================="
        
        torchrun --nproc-per-node=4 run.py --data "$dataset" --model "$MODEL" --verbose
        
        echo "评估完成: $MODEL - $dataset"
        echo "====================================================="
        echo ""
    done
done

echo "所有数据集评估完成!"
