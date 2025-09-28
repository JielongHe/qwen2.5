#!/bin/bash

# 分布式配置
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}


NPROC_PER_NODE=4

export CUDA_VISIBLE_DEVICES=0,1,2,3

# DeepSpeed 配置
deepspeed=./scripts/zero3.json

# 模型与数据
llm=Qwen/Qwen2.5-VL-3B-Instruct
data_path=samm_data

# 超参（注意：batch size 实际由 DeepSpeed 控制更可靠）
lr=2e-7
batch_size=2
grad_accum_steps=8

# 输出
run_name="qwen2vl-baseline-multi-gpu"
output_dir=./output

# 训练参数
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path \"${llm}\" \
    --dataset_use ${data_path} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy \"no\" \
    --save_strategy \"steps\" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type \"cosine\" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# 启动多卡训练
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}