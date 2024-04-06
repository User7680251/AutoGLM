import os
import json
from dashscope import MultiModalConversation
from tqdm import tqdm
import time


def process_images_in_folder(folder_path, output_file, max_retries=3, batch_size=10):
    """Process all images in a folder with retry mechanism, display a progress bar, and save the results to a JSON file."""
    results = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    batch_counter = 0  # 初始化批次计数器
    
    for filename in tqdm(image_files, desc="Processing images", unit="image"):
        file_path = os.path.join(folder_path, filename)
        label = None
        retries = 0
        
        while retries < max_retries and label is None:
            try:
                messages = create_messages(file_path)
                response = MultiModalConversation.call(model='qwen-vl-max', messages=messages)
                label = extract_text_content(response)
            except Exception as e:
                tqdm.write(f"Error processing {file_path}: {e}. Retrying...")
                retries += 1
                time.sleep(1)
        
        if label is None:
            tqdm.write(f"Failed to process {file_path} after {max_retries} retries.")
        else:
            result = {
                "img": file_path,
                "prompt": "任务：图片中为汽车在行驶，请观察路况并输出按照给定格式输出驾驶动作和推理过程。规则：在输出驾驶动作时，仅输出一种动作，使用以下标识符{}进行标记：紧急制动使用{EM} ，减速观察使用{DS} ，正常行驶使用{NS}。推理过程请使用一个[]标记。举例：{EM}[前方路面有障碍物，需要立即进行紧急制动，以避免碰撞]。输出：",
                "label": label
            }
            results.append(result)
            batch_counter += 1  # 增加批次计数器

        # 当达到批次大小或者已经是最后一批时，保存结果到JSON文件
        if batch_counter == batch_size or (filename == image_files[-1] and results):
            save_results_to_json(output_file, results)
            results = []  # 重置结果列表
            batch_counter = 0  # 重置批次计数器

def save_results_to_json(output_file, results):
    """Save the results to a JSON file."""
    with open(output_file, 'a', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False, indent=4)
            f.write('\n')

def create_messages(file_path):
    """Create messages for the MultiModalConversation call."""
    return [{
        'role': 'system',
        'content': [{
            'text': 'You are a helpful assistant.'
        }]
    }, {
        'role':
        'user',
        'content': [
            {
                'image': file_path
            },
            {
                'text': '任务：图片中为汽车在行驶，请观察路况并输出按照给定格式输出驾驶动作和推理过程。规则：在输出驾驶动作时，仅输出一种动作，使用以下标识符{}进行标记：紧急制动使用{EM} ，减速观察使用{DS} ，正常行驶使用{NS}。推理过程请使用一个[]标记。举例：{EM}[前方路面有障碍物，需要立即进行紧急制动，以避免碰撞]。输出：'
            },
        ]
    }]

def extract_text_content(response):
    """Extract text content from the response."""
    return response['output']['choices'][0]['message']['content'][0]['text']

if __name__ == '__main__':
    folder_path = '/home/AutoGLM/data/CODA-val-1500/CODA/base-val-1500/images/'
    output_file = 'data/CODA-val-1500/CODA/base-val-1500/output.json'
    process_images_in_folder(folder_path, output_file)