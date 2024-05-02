export HF_ENDPOINT=https://hf-mirror.com

json_path="data/json/2_reason_answer/test.json"
pretrained_model="/home/AutoGLM/checkpoints/2"
save_name="2"

python /home/AutoGLM/api.py --from_pretrained $pretrained_model & api_pid=$!
sleep 30
python /home/AutoGLM/metrics/metrics.py --from_pretrained $pretrained_model --json_path $json_path --save_name $save_name
kill $api_pid