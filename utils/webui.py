import streamlit as st
import requests
import base64

# 将图片文件转换为Base64编码的字符串
def image_to_base64(image_file):
    if image_file is not None:
        image_data = image_file.read()
        return base64.b64encode(image_data).decode('utf-8')
    return None

# FastAPI应用的URL
url = "http://127.0.0.1:8080"

st.title("AutoGLM")

# 创建一个文件上传组件
uploaded_file = st.file_uploader("上传行车图片", type=["jpg", "png"])

if uploaded_file is not None:
    # 显示上传的图片
    st.image(uploaded_file, caption='', use_column_width=True)

    # 将图片编码为Base64
    image_encoded = image_to_base64(uploaded_file)

    # 构造请求的数据
    data = {
        "text": "任务：图片中为汽车在行驶,请观察路况并输出按照给定格式输出驾驶动作和推理过程。",
        "image": image_encoded,
        "history": None
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

        # 显示结果
        st.write("AutoGLM返回结果:")
        st.write(result)
    else:
        st.write("Error: Server returned status code", response.status_code)

# 运行Streamlit应用
# 在命令行中运行以下命令：streamlit run utils/webui.py --server.address 0.0.0.0 --server.enableXsrfProtection=false
