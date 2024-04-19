import json
import requests
import base64
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from bert_score import score


def calculate_metrics(labels, responses):
    # 提取标签中的动作和推理文本
    refs_actions = [label[label.find('{') + 1:label.find('}')] for label in labels]
    refs_reasons = [label[label.find('[') + 1:label.find(']')] for label in labels]
    
    # 提取响应中的动作和推理文本
    hyps_actions = [response[response.find('{') + 1:response.find('}')] for response in responses]
    hyps_reasons = [response[response.find('[') + 1:response.find(']')] for response in responses]

    # 准确率计算
    accuracy_count = sum(ref == hyp for ref, hyp in zip(refs_actions, hyps_actions))
    accuracy = accuracy_count / len(refs_actions)

    # 计算BERTScore
    P, R, F1 = score(hyps_reasons, refs_reasons, lang="zh", verbose=True)

    # 计算BLEU分数
    bleu_scores = []
    for ref, hyp in zip(refs_reasons, hyps_reasons):
        bleu = sentence_bleu([list(ref)], list(hyp))
        bleu_scores.append(bleu)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return {
        "accuracy": accuracy,
        "avg_bleu": avg_bleu,
        "avg_bert_precision": P.mean(),
        "avg_bert_recall": R.mean(),
        "avg_bert_f1": F1.mean()
    }
