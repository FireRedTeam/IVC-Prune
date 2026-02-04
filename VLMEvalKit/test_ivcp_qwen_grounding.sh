 #!/bin/bash
export LMUData="/mnt/public/usr/sunzhichao/benchmark"
# 定义模型名称

MODELS=(
    "Qwen2.5-VL-7B-Instruct_Grounding_IVCP"
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

# 为每个数据集运行评估
for MODEL in "${MODELS[@]}"; do
    for dataset in "${datasets[@]}"; do

        echo "====================================================="
        echo "开始评估模型: $MODEL 数据集: $dataset"
        echo "====================================================="
        
        # 运行评估命令
        torchrun --nproc-per-node=4  run.py --data "$dataset" --model "$MODEL" --verbose
        
        # 打印分隔线
        echo "评估完成: $MODEL - $dataset"
        echo "====================================================="
        echo ""
    done
done

echo "所有数据集评估完成!"
