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
# python cli_demo.py --from_pretrained /home/AutoGLM/checkpoints/finetune-visualglm-6b-04-05-23-15 --prompt_zh 任务：图片中为汽车在行驶,请观察路况并输出按照给定格式输出驾驶动作和推理过程。规则：在输出驾驶动作时，紧急制动使用{EM},减速观察使用{DS},正常行驶使用{NS}。推理过程请使用一个[]标记。输出：

# # GUI 
# SAT_HOME=/root/.sat_models 
# unset http_proxy
# unset https_proxy
# python web_demo.py --share