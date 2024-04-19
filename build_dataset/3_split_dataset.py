import json
import random

# json文件的路径
json_file_path = '/home/AutoGLM/data/CODA-val-1500/CODA/base-val-1500/output.json'

# 读取JSON数据集
with open(json_file_path, 'r') as file:
    dataset = json.load(file)

# 随机打乱数据集
random.shuffle(dataset)

# 划分测试集和训练集
test_set_size = 100
test_set = dataset[:test_set_size]
train_set = dataset[test_set_size:]

# 保存训练集和测试集到不同的JSON文件，并保持缩进格式
with open('train_set.json', 'w', encoding='utf-8') as train_file:
    json.dump(train_set, train_file, ensure_ascii=False, indent=4)

with open('test_set.json', 'w', encoding='utf-8') as test_file:
    json.dump(test_set, test_file, ensure_ascii=False, indent=4)

print(f"训练集大小: {len(train_set)}")
print(f"测试集大小: {len(test_set)}")
