import streamlit as st
import base64

# 本地图片的路径
image_path = '/home/AutoGLM/web_ui/car_dashboard.png'

# 读取图片文件并编码为Base64
with open(image_path, 'rb') as image_file:
    image_data = image_file.read()
    encoded_string = base64.b64encode(image_data).decode()

# 创建Base64图片URL
encoded_image_url = f"data:image/png;base64,{encoded_string}"

# 自定义CSS样式代码
css = f"""
<style>
    body {{
        margin: 0; /* 清除默认外边距 */
        padding: 0; /* 清除默认内边距 */
        height: 100vh; /* 设置高度为视口的100%，使背景图片能够铺满全屏 */
        background-image: url('{encoded_image_url}'); /* 替换为你的图片URL */
        background-position: center center; /* 将背景图片居中 */
        background-repeat: no-repeat; /* 不重复背景图片 */
        background-attachment: fixed; /* 固定背景图片，使其不随页面滚动而移动 */
        background-size: cover; /* 覆盖整个容器，使背景图片能够铺满全屏 */
    }}
</style>
"""

# 使用st.markdown将CSS样式添加到页面中
st.write(css, unsafe_allow_html=True)

#streamlit run web_ui/web_ui_2_0.py --server.address 0.0.0.0 --server.enableXsrfProtection=false