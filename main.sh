#!/bin/bash
# train.sh

# 定义日志文件名，包含当前日期和时间
LOG_FILE="logs/train_$(date +'%Y%m%d_%H%M%S').log"

echo "Starting train at $(date)" | tee -a $LOG_FILE




python main_experiment.py --dataset TCGA_BRCA --backbone ctp --exp_name initial_run --use_fixed_splits --split_num 0  2>&1 | tee -a $LOG_FILE

echo "Finished at $(date)" | tee -a $LOG_FILE



