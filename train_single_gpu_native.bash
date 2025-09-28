#!/bin/bash
# ========================================================
# 单卡训练脚本（无 DeepSpeed）
# 直接使用 Python + PyTorch 训练 Qwen-VL 模型
# ========================================================

# -------------------------------
# 1. GPU 配置（可选：指定使用哪个 GPU）
# -------------------------------
# 如果有多张卡，可以指定使用某一张（例如 0 号 GPU）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# -------------------------------
# 2. 模型配置
# -------------------------------
# 支持 Hugging Face ID 或本地路径
llm=/home/msi/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct
# 示例本地路径：
# llm="/home/aorus/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct"

# -------------------------------
# 3. 训练超参数
# -------------------------------
batch_size=1                         # 根据显存调整（2~4 推荐）
grad_accum_steps=1                   # 保持总 batch size ≈ 16
lr=2e-7                              # 学习率

# -------------------------------
# 4. 数据集配置
# -------------------------------
data_path=samm_data                  # 数据集名称或路径

# -------------------------------
# 5. 输出配置
# -------------------------------
run_name="qwen2vl-baseline-native"   # 实验名称
output_dir=./output                  # 模型保存目录

# -------------------------------
# 6. 训练参数构建
# -------------------------------
args="
    --model_name_or_path ${llm} \
    --dataset_use ${data_path} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size * 2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# -------------------------------
# 7. 启动训练（直接使用 Python）
# -------------------------------
echo "🚀 启动单卡训练（无 DeepSpeed）..."
echo "使用 GPU: cuda:${CUDA_VISIBLE_DEVICES}"
echo "模型: ${llm}"
echo "输出目录: ${output_dir}"
echo "实验名称: ${run_name}"

python ./qwenvl/train/train_qwen.py ${args}

# -------------------------------
# 8. 错误处理
# -------------------------------
if [ $? -ne 0 ]; then
    echo "❌ 训练失败，请检查日志"
    exit 1
fi

echo "🎉 训练完成"