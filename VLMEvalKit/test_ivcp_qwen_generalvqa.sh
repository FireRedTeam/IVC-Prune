 #!/bin/bash
export LMUData="/mnt/public/usr/sunzhichao/benchmark"


MODELS=(
    "Qwen2.5-VL-7B-Instruct_IVCP"
    "Qwen2.5-VL-3B-Instruct_IVCP"
)


datasets=(
    "SEEDBench_IMG"
    "MMBench_DEV_EN_V11"
    "MMStar"
    "RealWorldQA"
    "MME"
    "POPE"

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
