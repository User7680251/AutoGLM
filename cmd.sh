# # 验证模型 demo
# SAT_HOME=/root/.sat_models 
# export HF_ENDPOINT=https://hf-mirror.com
# python cli_demo.py 

# # 微调模型 demo
# SAT_HOME=/root/.sat_models 
# bash finetune/finetune_visualglm.sh

# # 验证微调后模型 demo
# python cli_demo.py --from_pretrained /root/VisualGLM-6B/checkpoints/finetune-visualglm-6b-04-05-13-58 --prompt_zh 这张图片的背景里有什么内容？

# 微调模型 data
SAT_HOME=/root/.sat_models 
bash finetune/finetune_visualglm.sh

# # 验证微调后模型 data
# python cli_demo.py --from_pretrained /home/VisualGLM-6B/checkpoints/finetune-visualglm-6b-04-05-14-23
