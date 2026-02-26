 #!/bin/bash
export LMUData="/PATH/"


MODELS=(
    "Qwen2.5-VL-7B-Instruct_Grounding_IVCP"
)


datasets=(

    "RefCOCO_testA_for_qwen25vl"
    "RefCOCO_testB_for_qwen25vl"
    "RefCOCO_val_for_qwen25vl"
    "RefCOCO+_testA_for_qwen25vl"
    "RefCOCO+_testB_for_qwen25vl"
    "RefCOCO+_val_for_qwen25vl"
    "RefCOCOg_test_for_qwen25vl"
    "RefCOCOg_val_for_qwen25vl"
)

# 为每个数据集运行评估
for MODEL in "${MODELS[@]}"; do
    for dataset in "${datasets[@]}"; do

        echo "====================================================="
        echo "开始评估模型: $MODEL 数据集: $dataset"
        echo "====================================================="
        
        python run.py --data "$dataset" --model "$MODEL" --verbose
        
        echo "评估完成: $MODEL - $dataset"
        echo "====================================================="
        echo ""
    done
done

echo "所有数据集评估完成!"
