import json
import requests
import base64
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from tqdm import tqdm


# FastAPI应用的URL
url = "http://127.0.0.1:8080"

# 将图片文件转换为Base64编码的字符串
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    return base64.b64encode(image_data).decode('utf-8')

# 步骤1: 从JSON文件中读取数据
json_path = "/home/AutoGLM/test.json"
with open(json_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

# 提取标签列表
labels = [item['label'] for item in dataset]

# 禁用代理
proxies = {
    "http": None,
    "https": None,
}

# 发送POST请求到模型API并获取响应
responses = []
for item in tqdm(dataset, desc="Sending requests"):
    # 构造请求的数据
    data = {
        "text": "任务：图片中为汽车在行驶,请观察路况并输出按照给定格式输出驾驶动作和推理过程。",
        "image": image_to_base64(item['img']),
        "history": None
    }

    # 发送POST请求
    response = requests.post(url, json=data, proxies=proxies)

    # 确保响应成功
    if response.status_code == 200:
        # 解析JSON响应
        parsed_json = response.json()
        
        # 提取result字段
        result = parsed_json.get('result', None)
        responses.append(result)
    else:
        print(f"Error: {response.status_code}")
        responses.append("")  # 或者其他适当的错误处理

# 步骤3: 计算BLEU和ROUGE分数
rouge = Rouge()

bleu_scores = []
rouge_scores = []

for ref, hyp in tqdm(zip(labels, responses), desc="Calculating scores"):
    # BLEU分数计算
    reference = [ref.split()]  # 将参考翻译转换为单词列表的列表
    hypothesis = hyp.split()  # 将候选翻译转换为单词列表
    bleu = sentence_bleu(reference, hypothesis)
    bleu_scores.append(bleu)

    # ROUGE分数计算
    scores = rouge.get_scores(hyp, ref)
    rouge_scores.append(scores[0])  # scores是一个字典列表，我们只需要第一个字典

# 打印平均分数
avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_rouge1 = sum([s['rouge-1']['f'] for s in rouge_scores]) / len(rouge_scores)
avg_rouge2 = sum([s['rouge-2']['f'] for s in rouge_scores]) / len(rouge_scores)
avg_rougeL = sum([s['rouge-l']['f'] for s in rouge_scores]) / len(rouge_scores)

print(f"平均BLEU分数: {avg_bleu}")
print(f"平均ROUGE-1分数: {avg_rouge1}")
print(f"平均ROUGE-2分数: {avg_rouge2}")
print(f"平均ROUGE-L分数: {avg_rougeL}")
