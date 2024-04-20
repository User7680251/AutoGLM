# # 构建数据集
# bash build_dataset/build_dataset.sh

# 微调模型
export HF_ENDPOINT=https://hf-mirror.com
SAT_HOME=/root/.sat_models 
bash finetune/finetune_visualglm.sh

# # metrics
# bash metrics/metrics.sh

# # web ui 
# bash web_ui/web_ui.sh