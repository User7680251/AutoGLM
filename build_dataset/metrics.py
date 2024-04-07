import json
import requests
import base64
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from bert_score import score


# FastAPI应用的URL
url = "http://127.0.0.1:8080"

# 将图片文件转换为Base64编码的字符串
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    return base64.b64encode(image_data).decode('utf-8')

# 步骤1: 从JSON文件中读取数据
json_path = "/home/AutoGLM/data/coda_sample/CODA/sample/output1.json"
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