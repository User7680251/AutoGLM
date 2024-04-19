import json
import requests
import base64
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from bert_score import score
import os
import argparse
import time
import torch
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss

from finetune_visualglm import get_batch


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


def eval_step(data_iterator, model, args, timers):
    """Evaluation step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, image, pre_image = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    # Forward pass without calculating loss
    with torch.no_grad():
        logits = model(input_ids=tokens, image=image, pre_image=pre_image)[0]

    dtype = logits.dtype
    lm_logits = logits.to(torch.float32)

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    lm_logits = lm_logits.to(dtype)
    loss = loss.to(dtype)

    # Convert logits to predicted tokens
    predicted_tokens = torch.argmax(logits, dim=-1)

    # Decode tokens into responses (complete sentences)
    tokenizer = AutoTokenizer.from_pretrained("PiaoYang/chatglm-6b", trust_remote_code=True)
    responses = []
    for predicted_token_sequence in predicted_tokens:
        response = tokenizer.decode(predicted_token_sequence, skip_special_tokens=True)
        responses.append(response)

    # Calculate metrics using the provided function
    metrics = calculate_metrics(labels, responses)

    return loss, {'loss': loss, **metrics} 