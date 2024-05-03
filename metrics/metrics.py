import json
import requests
import base64
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from bert_score import score
import torch
import time
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


def calculate_metrics(labels, responses):
    # 初始化分数变量
    accuracy = 0
    avg_bleu = 0
    avg_bert_precision = 0
    avg_bert_recall = 0
    avg_bert_f1 = 0

    # 初始化用于计算BLEU和BERTScore的列表
    refs_reasons = []
    hyps_reasons = []

    ref_actions = []
    hyp_actions = []

    # 定义动作到索引的映射
    action_to_index = {'NS': 0, 'DS': 1, 'EM': 2}

    # 初始化混淆矩阵
    confusion_matrix = torch.zeros(3, 3, dtype=torch.int64).cuda()

    # 提取标签和响应中的动作和推理文本
    for label, response in zip(labels, responses):
        try:
            # 尝试提取动作
            ref_action = label[label.find('{') + 1:label.find('}')]
            hyp_action = response[response.find('{') + 1:response.find('}')]

            # 尝试提取推理文本
            ref_reason = label[label.find('[') + 1:label.find(']')]
            hyp_reason = response[response.find('[') + 1:response.find(']')]

            # 将提取的推理文本添加到列表中
            refs_reasons.append(ref_reason)
            hyps_reasons.append(hyp_reason)

            ref_actions.append(ref_action)
            hyp_actions.append(hyp_action)

        except ValueError:
            # 如果提取失败，则跳过该样本，并在最后计算平均分时给出0分
            continue

    # 如果成功提取了任何样本，计算准确率
    if refs_reasons and hyps_reasons:
        # 计算BERTScore
        P, R, F1 = score(hyps_reasons, refs_reasons, lang="zh", verbose=True)
        avg_bert_precision = P.mean()
        avg_bert_recall = R.mean()
        avg_bert_f1 = F1.mean()

        # 计算BLEU分数
        bleu_scores = [sentence_bleu([list(ref)], list(hyp)) for ref, hyp in zip(refs_reasons, hyps_reasons)]
        avg_bleu = sum(bleu_scores) / len(bleu_scores)

        ref_index = action_to_index[ref_action]
        hyp_index = action_to_index[hyp_action]
        
        confusion_matrix[ref_index, hyp_index] += 1

        # 从混淆矩阵中计算true positive, false positive, false negative
    true_positive = torch.diag(confusion_matrix)
    false_positive = confusion_matrix.sum(dim=0) - true_positive
    false_negative = confusion_matrix.sum(dim=1) - true_positive

    # 计算召回率和精确率
    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)

    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall)

    # 计算宏观平均的召回率、精确率和F1分数
    macro_recall = recall.mean()
    macro_precision = precision.mean()
    macro_f1 = f1.mean()

    # 计算准确率
    accuracy = true_positive.sum() / confusion_matrix.sum()

    return {
        "accuracy": torch.tensor(accuracy, dtype=torch.float32).cuda(),
        "macro_recall": torch.tensor(macro_recall, dtype=torch.float32).cuda(),
        "macro_precision": torch.tensor(macro_precision, dtype=torch.float32).cuda(),
        "macro_f1": torch.tensor(macro_f1, dtype=torch.float32).cuda(),
        "avg_bleu": torch.tensor(avg_bleu, dtype=torch.float32).cuda(),
        "avg_bert_precision": avg_bert_precision.cuda(),
        "avg_bert_recall": avg_bert_recall.cuda(),
        "avg_bert_f1": avg_bert_f1.cuda()
    }, confusion_matrix

def plot_confusion_matrix(confusion_matrix, action_names, filename="temp.jpg"):
    # 将混淆矩阵转换为NumPy数组
    confusion_matrix = confusion_matrix.cpu().numpy()

    # 创建一个新的图形
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=action_names, yticklabels=action_names,
           title="Confusion Matrix",
           ylabel='True Label',
           xlabel='Predicted Label')

    # 在每个单元格中添加文本
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > (confusion_matrix.max() / 2) else "black")

    # 保存图形
    fig.savefig(filename)

def run_metrics():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="/home/AutoGLM/data/test_set.json", help='pretrained ckpt')
    parser.add_argument("--save_name", type=str, default="lora", help='pretrained ckpt')
    parser.add_argument("--from_pretrained", type=str, default="/home/AutoGLM/checkpoints/lora_CODA100", help='pretrained ckpt')
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
            "text": item['prompt'],
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
                    print(result)
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


    # 调用metric函数计算指标
    metrics, confusion_matrix = calculate_metrics(labels, responses)

    # 定义动作名称
    action_names = ['NS', 'DS', 'EM']

    # 可视化并保存混淆矩阵
    plot_confusion_matrix(confusion_matrix, action_names)

    # 首先，我们将要打印的指标整理成字符串
    metrics_str = """
    {args.save_name}
    {args.from_pretrained}
    {args.json_path}
    准确率: {accuracy:.6f}
    召回率: {recall:.6f}
    精确率: {precision:.6f}
    F1分数: {f1:.6f}
    平均BLEU分数: {avg_bleu:.6f}
    平均BERTScore Precision: {avg_bert_precision:.6f}
    平均BERTScore Recall: {avg_bert_recall:.6f}
    平均BERTScore F1: {avg_bert_f1:.6f}
    """

    # 将指标转换为CPU张量以打印
    accuracy = metrics["accuracy"].cpu().item()
    recall = metrics["macro_recall"].cpu().item()
    precision = metrics["macro_precision"].cpu().item()
    f1 = metrics["macro_f1"].cpu().item()
    avg_bleu = metrics["avg_bleu"].cpu().item()
    avg_bert_precision = metrics["avg_bert_precision"].cpu().item()
    avg_bert_recall = metrics["avg_bert_recall"].cpu().item()
    avg_bert_f1 = metrics["avg_bert_f1"].cpu().item()

    metrics_str = metrics_str.format(args=args, accuracy=accuracy, recall=recall, precision = precision, f1=f1,
                                    avg_bleu=avg_bleu, avg_bert_precision=avg_bert_precision,
                                    avg_bert_recall=avg_bert_recall, avg_bert_f1=avg_bert_f1)

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
    new_content = existing_content + metrics_str

    # 写入新内容到文件
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(new_content)

    file_exists, file_path

    # 打印指标
    print(metrics_str)


if __name__ == '__main__':
    run_metrics()