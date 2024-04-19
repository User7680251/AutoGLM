import json
import requests
import base64
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from bert_score import score
import os
import argparse
import time


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


def func(data_iterator, model, args, timers):
    import torch
    from model import is_chinese, get_infer_setting, generate_input, chat
    from transformers import AutoTokenizer
    from finetune_visualglm import get_batch

    tokenizer = AutoTokenizer.from_pretrained("PiaoYang/chatglm-6b", trust_remote_code=True)
    input_para = {
        "max_length": 2048,
        "min_length": 50,
        "temperature": 0.8,
        "top_p": 0.4,
        "top_k": 100,
        "repetition_penalty": 1.2
    }
    # Get the batch.
    timers('batch generator').start()
    tokens, labels, image, pre_image = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    is_zh = is_chinese(tokens)
    input_data = generate_input(tokens, None, history, input_para)
    gen_kwargs = input_data['gen_kwargs']
    with torch.no_grad():
        answer, history, _ = chat(None, model, tokenizer, tokens, history=history, image=image, \
                            max_length=gen_kwargs['max_length'], top_p=gen_kwargs['top_p'], \
                            top_k = gen_kwargs['top_k'], temperature=gen_kwargs['temperature'], english=not is_zh)