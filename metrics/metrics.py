import json
import requests
import base64
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from bert_score import score
import torch


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
        except ValueError:
            # 如果提取失败，则跳过该样本，并在最后计算平均分时给出0分
            continue

    # 如果成功提取了任何样本，计算准确率
    if refs_reasons and hyps_reasons:
        # 准确率计算
        accuracy_count = sum(ref == hyp for ref, hyp in zip(refs_reasons, hyps_reasons))
        accuracy = accuracy_count / len(refs_reasons)

        # 计算BERTScore
        P, R, F1 = score(hyps_reasons, refs_reasons, lang="zh", verbose=True)
        avg_bert_precision = P.mean()
        avg_bert_recall = R.mean()
        avg_bert_f1 = F1.mean()

        # 计算BLEU分数
        bleu_scores = [sentence_bleu([list(ref)], list(hyp)) for ref, hyp in zip(refs_reasons, hyps_reasons)]
        avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return {
        "accuracy": torch.tensor(accuracy, dtype=torch.float32).cuda(),
        "avg_bleu": torch.tensor(avg_bleu, dtype=torch.float32).cuda(),
        "avg_bert_precision": avg_bert_precision.cuda(),
        "avg_bert_recall": avg_bert_recall.cuda(),
        "avg_bert_f1": avg_bert_f1.cuda()
    }