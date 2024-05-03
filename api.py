import os
import json
import uvicorn
from fastapi import FastAPI, Request
from model import is_chinese, get_infer_setting, generate_input, chat
import datetime
import torch
import argparse
from sat.model import AutoModel
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from finetune_visualglm import FineTuneVisualGLMModel

gpu_number = 0
parser = argparse.ArgumentParser()
parser.add_argument("--from_pretrained", type=str, default="visualglm-6b", help='pretrained ckpt')
args = parser.parse_args()

# load model
model, model_args = AutoModel.from_pretrained(
    args.from_pretrained,
    args=argparse.Namespace(
    fp16=True,
    skip_init=True,
    use_gpu_initialization=True if (torch.cuda.is_available()) else False,
    device='cuda' if (torch.cuda.is_available()) else 'cpu',
))
model = model.eval()

model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

tokenizer = AutoTokenizer.from_pretrained("PiaoYang/chatglm-6b", trust_remote_code=True)

app = FastAPI()
@app.post('/')
async def visual_glm(request: Request):
    json_post_raw = await request.json()
    print("Start to process request")

    json_post = json.dumps(json_post_raw)
    request_data = json.loads(json_post)
    input_text, input_image_encoded, history = request_data['text'], request_data['image'], request_data['history']
    input_para = {
        "max_length": 2048,
        "min_length": 50,
        "temperature": 1.9,
        "top_p": 0.4,
        "top_k": 100,
        "repetition_penalty": 1.2
    }
    input_para.update(request_data)

    is_zh = is_chinese(input_text)
    input_data = generate_input(input_text, input_image_encoded, history, input_para)
    input_image, gen_kwargs =  input_data['input_image'], input_data['gen_kwargs']
    with torch.no_grad():
        answer, history, _ = chat(None, model, tokenizer, input_text, history=history, image=input_image, \
                            max_length=gen_kwargs['max_length'], top_p=gen_kwargs['top_p'], \
                            top_k = gen_kwargs['top_k'], temperature=gen_kwargs['temperature'], english=not is_zh)
        
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    response = {
        "result": answer,
        "history": history,
        "status": 200,
        "time": time
    }
    return response


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)