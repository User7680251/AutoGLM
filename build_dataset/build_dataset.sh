# build data from qwen
export DASHSCOPE_API_KEY=sk-1c73b81413294b62ae84f3fc91afa299
python build_dataset/1_label_from_qwen.py

# check data by human
python build_dataset/2_check.py

# split dataset
python build_dataset/3_split_dataset.py

# reverse label
python build_dataset/4_label_reverse.py --input_json_path data/train_set.json
python build_dataset/4_label_reverse.py --input_json_path data/test_set.json

# answer only
python build_dataset/5_label_answer_only.py --input_json_path data/train_set.json
python build_dataset/5_label_answer_only.py --input_json_path data/test_set.json