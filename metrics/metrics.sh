export HF_ENDPOINT=https://hf-mirror.com

json_path="/home/AutoGLM/data/test_set.json"
pretrained_model="/home/AutoGLM/checkpoints/lora"
save_name="lora"

python /home/AutoGLM/api.py --from_pretrained $pretrained_model & api_pid=$!
sleep 30
python /home/AutoGLM/metrics/metrics.py --from_pretrained $pretrained_model --json_path $json_path --save_name $save_name
kill $api_pid