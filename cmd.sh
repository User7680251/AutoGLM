# # 微调模型 data
# SAT_HOME=/root/.sat_models 
# bash finetune/finetune_visualglm.sh

# # api
# export HF_ENDPOINT=https://hf-mirror.com
# python api.py --from_pretrained /home/AutoGLM/checkpoints/qlora_CODA100_1

# # streamlit web ui 
# streamlit run utils/webui.py --server.address 0.0.0.0 --server.enableXsrfProtection=false

# get metrics
bash metrics/metrics.sh