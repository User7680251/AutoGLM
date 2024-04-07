# # 微调模型 data
# SAT_HOME=/root/.sat_models 
# bash finetune/finetune_visualglm.sh

# # 验证微调后模型 data
# export HF_ENDPOINT=https://hf-mirror.com
# python cli_demo.py --from_pretrained /home/AutoGLM/checkpoints/ptuning_CODA1500 --prompt_zh 任务：图片中为汽车在行驶,请观察路况并输出按照给定格式输出驾驶动作和推理过程。

# api
export HF_ENDPOINT=https://hf-mirror.com
python api.py --from_pretrained /home/AutoGLM/checkpoints/lora_CODA100_1

# streamlit web ui 
streamlit run utils/webui.py --server.address 0.0.0.0 --server.enableXsrfProtection=false