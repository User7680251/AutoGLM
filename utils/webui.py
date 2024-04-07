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

st.title("图片上传和API调用演示")

# 创建一个文件上传组件
uploaded_file = st.file_uploader("上传图片", type=["jpg", "png"])

if uploaded_file is not None:
    # 显示上传的图片
    st.image(uploaded_file, caption='上传的图片', use_column_width=True)

    # 将图片编码为Base64
    image_encoded = image_to_base64(uploaded_file)

    # 构造请求的数据
    data = {
        "text": "任务：图片中为汽车在行驶，请观察路况并输出按照给定格式输出驾驶动作和推理过程。规则：在输出驾驶动作时，仅输出一种动作，使用以下标识符{}进行标记：紧急制动使用{EM} ，减速观察使用{DS} ，正常行驶使用{NS}。推理过程请使用一个[]标记。举例：{EM}[前方路面有障碍物，需要立即进行紧急制动，以避免碰撞]。输出：",
        "image": image_encoded,
        "history": None
    }

    # 发送POST请求
    response = requests.post(url, json=data)

    # 确保响应状态码是200
    if response.status_code == 200:
        # 解析JSON响应
        parsed_json = response.json()
        
        # 提取result字段
        result = parsed_json.get('result', None)

        # 显示结果
        st.write("API返回的结果:")
        st.write(result)
    else:
        st.write("Error: Server returned status code", response.status_code)

# 运行Streamlit应用
# 在命令行中运行以下命令：streamlit run app.py
