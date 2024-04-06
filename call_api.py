import requests
import base64

# 将图片文件转换为Base64编码的字符串
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    return base64.b64encode(image_data).decode('utf-8')

# FastAPI应用的URL
url = "http://127.0.0.1:8080"

# 图片的本地地址
image_path = "data/coda_sample/CODA/sample/images/000036_1616177260599.jpg"

# 将图片编码为Base64
image_encoded = image_to_base64(image_path)

# 构造请求的数据
data = {
    "text": "任务：图片中为汽车在行驶，请观察路况并输出按照给定格式输出驾驶动作和推理过程。规则：在输出驾驶动作时，仅输出一种动作，使用以下标识符{}进行标记：紧急制动使用{EM} ，减速观察使用{DS} ，正常行驶使用{NS}。推理过程请使用一个[]标记。举例：{EM}[前方路面有障碍物，需要立即进行紧急制动，以避免碰撞]。输出：",
    "image": image_encoded,
    "history":None
}

# 禁用代理
proxies = {
    "http": None,
    "https": None,
}

# 发送POST请求
response = requests.post(url, json=data, proxies=proxies)

# 确保响应状态码是200
if response.status_code == 200:
    # 解析JSON响应
    parsed_json = response.json()
    
    # 提取result字段
    result = parsed_json.get('result', None)

    # 打印result
    print(result)
else:
    print("Error: Server returned status code", response.status_code)
