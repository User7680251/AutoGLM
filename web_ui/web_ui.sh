# api
export HF_ENDPOINT=https://hf-mirror.com
pretrained_model="/home/AutoGLM/checkpoints/lora"
python api.py --from_pretrained $pretrained_model & api_pid=$!
sleep 30

# streamlit web ui 
streamlit run web_ui/web_ui.py --server.address 0.0.0.0 --server.enableXsrfProtection=false
kill $api_pid