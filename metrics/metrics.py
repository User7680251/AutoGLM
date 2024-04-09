import json
import requests
import base64
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from bert_score import score
import os
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default="/home/AutoGLM/runs/test.json", help='pretrained ckpt')
parser.add_argument("--save_name", type=str, default="qlora", help='pretrained ckpt')
parser.add_argument("--from_pretrained", type=str, default="/home/AutoGLM/checkpoints/qlora_CODA100_1", help='pretrained ckpt')
args = parser.parse_args()

# FastAPI应用的URL
url = "http://127.0.0.1:8080"

# 将图片文件转换为Base64编码的字符串
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    return base64.b64encode(image_data).decode('utf-8')

# 步骤1: 从JSON文件中读取数据
json_path = args.json_path
with open(json_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

# 提取标签列表
labels = [item['label'] for item in dataset]

# 禁用代理
proxies = {
    "http": None,
    "https": None,
}

# 设置重试次数和间隔
max_retries = 10
retry_interval = 6  # 单位：秒

# 存储响应的列表
responses = []

# 遍历数据集并发送请求
for item in tqdm(dataset, desc="Sending requests"):
    # 构造请求数据
    data = {
        "text": "任务：图片中为汽车在行驶,请观察路况并输出按照给定格式输出驾驶动作和推理过程。",
        "image": image_to_base64(item['img']),
        "history": None
    }
    
    # 初始化重试次数
    retries = 0
    while retries < max_retries:
        try:
            # 发送POST请求
            response = requests.post(url, json=data, proxies=proxies)
            
            # 检查响应状态码
            if response.status_code == 200:
                # 解析JSON响应
                parsed_json = response.json()
                
                # 提取result字段
                result = parsed_json.get('result', None)
                responses.append(result)
                break  # 如果成功，则跳出循环
            else:
                # 如果响应状态码不是200，则等待一段时间后重试
                time.sleep(retry_interval)
                retries += 1
        except requests.exceptions.RequestException as e:
            # 打印异常信息
            print(f"Request failed: {e}")
            time.sleep(retry_interval)
            retries += 1

    # 如果达到最大重试次数仍失败，则打印错误信息
    if retries == max_retries:
        print(f"Error: Max retries reached for item {item}")

# 准确率计算
refs = [label[label.find('{') + 1:label.find('}')] for label in labels]  # 提取标签中的文本部分
hyps = [response[response.find('{') + 1:response.find('}')] for response in responses]  # 提取响应中的文本部分

accuracy_count = 0
for ref, hyp in zip(refs, hyps):
    if ref == hyp:
        accuracy_count += 1

# 计算准确率
accuracy = accuracy_count / len(refs)
print(f"准确率: {accuracy:.6f}")

# 使用bert-score计算分数
refs = [label[label.find('[') + 1:label.find(']')] for label in labels]  # 提取标签中的文本部分
hyps = [response[response.find('[') + 1:response.find(']')] for response in responses]  # 提取响应中的文本部分

# 计算Score
P, R, F1 = score(hyps, refs, lang="zh", verbose=True)
bleu_scores = []

for ref, hyp in tqdm(zip(refs, hyps), desc="Calculating scores"):
    # BLEU分数计算
    reference = [list(ref)]  # 将参考翻译转换为单词列表的列表
    hypothesis = list(hyp)  # 将候选翻译转换为单词列表
    bleu = sentence_bleu(reference, hypothesis)
    bleu_scores.append(bleu)

# 打印平均分数
avg_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"平均BLEU分数: {avg_bleu}")

# 打印平均分数
print(f"平均BERTScore Precision: {P.mean():.6f}")
print(f"平均BERTScore Recall: {R.mean():.6f}")
print(f"平均BERTScore F1: {F1.mean():.6f}")

# 首先，我们将要打印的指标整理成字符串
metrics = """
{args.save_name}
{args.from_pretrained}
{args.json_path}
准确率: {accuracy:.6f}
平均BLEU分数: {avg_bleu:.6f}
平均BERTScore Precision: {P:.6f}
平均BERTScore Recall: {R:.6f}
平均BERTScore F1: {F1:.6f}
"""

metrics = metrics.format(args=args, accuracy=accuracy, avg_bleu=avg_bleu, P=P.mean(), R=R.mean(), F1=F1.mean())

# 接下来，我们将这个字符串写入到一个文本文件中，但首先要检查这个文件是否已经存在
file_path = "/home/AutoGLM/runs/metrics.txt"

# 检查文件是否存在
file_exists = os.path.exists(file_path)

# 如果文件存在，我们需要读取原有的内容，并追加新的内容
if file_exists:
    with open(file_path, "r", encoding="utf-8") as file:
        # 读取原有内容
        existing_content = file.read()
else:
    existing_content = ""

# 准备要写入的新内容
new_content = existing_content + metrics

# 写入新内容到文件
with open(file_path, "w", encoding="utf-8") as file:
    file.write(new_content)

file_exists, file_path
