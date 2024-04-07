# # 验证模型 demo
# SAT_HOME=/root/.sat_models 
# export HF_ENDPOINT=https://hf-mirror.com
# python cli_demo.py 

# # 微调模型 demo
# SAT_HOME=/root/.sat_models 
# bash finetune/finetune_visualglm.sh

# # 验证微调后模型 demo
# python cli_demo.py --from_pretrained /root/VisualGLM-6B/checkpoints/finetune-visualglm-6b-04-05-13-58 --prompt_zh 这张图片的背景里有什么内容？

# # 微调模型 data
# SAT_HOME=/root/.sat_models 
# bash finetune/finetune_visualglm.sh

# # 验证微调后模型 data
# export HF_ENDPOINT=https://hf-mirror.com
# python cli_demo.py --from_pretrained /home/AutoGLM/checkpoints/ptuning_CODA1500 --prompt_zh 任务：图片中为汽车在行驶,请观察路况并输出按照给定格式输出驾驶动作和推理过程。

# # GUI 
# SAT_HOME=/root/.sat_models 
# unset http_proxy
# unset https_proxy
# python web_demo.py --share

# api
export HF_ENDPOINT=https://hf-mirror.com
python api.py --from_pretrained /home/AutoGLM/checkpoints/lora_CODA100_1