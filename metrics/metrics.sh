export HF_ENDPOINT=https://hf-mirror.com
json_path="/home/AutoGLM/data/coda_sample/CODA/sample/output1.json"

pretrained_model="/home/AutoGLM/checkpoints/lora_CODA100_1"
save_name="lora"
python /home/AutoGLM/api.py --from_pretrained $pretrained_model & api_pid=$!
sleep 30
python /home/AutoGLM/metrics/metrics.py --from_pretrained $pretrained_model --json_path $json_path --save_name $save_name
kill $api_pid
sleep 30

pretrained_model="/home/AutoGLM/checkpoints/ptuning_CODA1500"
save_name="ptuning"
python /home/AutoGLM/api.py --from_pretrained $pretrained_model & api_pid=$!
sleep 30
python /home/AutoGLM/metrics/metrics.py --from_pretrained $pretrained_model --json_path $json_path --save_name $save_name
kill $api_pid
sleep 30

pretrained_model="/home/AutoGLM/checkpoints/qlora_CODA100_1"
save_name="qlora"
python /home/AutoGLM/api.py --from_pretrained $pretrained_model & api_pid=$!
sleep 30
python /home/AutoGLM/metrics/metrics.py --from_pretrained $pretrained_model --json_path $json_path --save_name $save_name
kill $api_pid

python /home/AutoGLM/metrics/metrics.py --json_path "/home/AutoGLM/data/coda_sample/CODA/sample/rev.json"