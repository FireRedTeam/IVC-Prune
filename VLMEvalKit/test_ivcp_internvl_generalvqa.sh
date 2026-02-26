 #!/bin/bash
export LMUData="/mnt/public/usr/sunzhichao/benchmark"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

MODELS=(
    "InternVL2_5-8B_ivcp"
)


datasets=(
    "SEEDBench_IMG"
    # "MMBench_DEV_EN_V11"
    # "MMStar"
    # "RealWorldQA"
    # "MME"
    # "POPE"

)

# 为每个数据集运行评估
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
