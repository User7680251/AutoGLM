# # 构建数据集
# bash build_dataset/build_dataset.sh

# 微调模型
export HF_ENDPOINT=https://hf-mirror.com
SAT_HOME=/root/.sat_models 
# bash finetune/1.sh
# bash finetune/2.sh
# bash finetune/3.sh
# bash finetune/4.sh
bash finetune/5.sh
bash finetune/6.sh

# # metrics
# bash metrics/metrics.sh

# # web ui 
# bash web_ui/web_ui.sh