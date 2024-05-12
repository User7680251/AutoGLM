import json
from PIL import Image
import matplotlib.pyplot as plt

# 加载JSON数据
json_file = 'data/json/2_reason_answer/test.json'  # 替换为您的JSON文件路径
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历数据集，保存每张图片和对应的标签
for item in data:
    # 打开图片
    img_path = item['img']
    img = Image.open(img_path)
    
    # 保存图片到当前目录下的temp.jpg
    plt.imshow(img)
    
    plt.axis('off')  # 不显示坐标轴
    plt.savefig('temp.jpg')  # 保存图片到当前目录
    
    print(item['label'])

    # 获取新的标签
    new_label = input("请输入新的标签（回车以保留原始标签）: ")
    
    if new_label:
        item['label'] = new_label

    # 保存修改后的数据到JSON文件
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("已修改为：", item['label'])

