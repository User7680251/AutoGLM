# api
export HF_ENDPOINT=https://hf-mirror.com
pretrained_model="/home/AutoGLM/checkpoints/lora_CODA100_1"
python api.py --from_pretrained $pretrained_model & api_pid=$!
sleep 30

# streamlit web ui 
streamlit run web_ui/we_bui.py --server.address 0.0.0.0 --server.enableXsrfProtection=false
kill $api_pid