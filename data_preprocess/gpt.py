import os
import sys
sys.path.append(os.getcwd())  # 将当前工作目录添加到系统路径中，以便导入其他模块

import io
from PIL import Image  # 导入PIL库用于图像处理
import openai  # 导入OpenAI API
import yaml    # 导入YAML库用于读取配置文件
from typing import Optional  # 导入Optional类型提示

from data_preprocess.prompt import prompt_score_system, prompt_score_user  # 导入预定义的提示模板


def encode_image(image_path: str) -> str:
    """
    将图像编码为base64字符串，用于OpenAI API的图像输入
    
    参数:
        image_path: 图像文件的路径
        
    返回:
        编码后的图像URL字符串，格式为data:image/png;base64,...
    """
    import base64
    
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("ascii")

    image_url = f"data:image/png;base64," + encoded_image

    return image_url



def get_message(text_list, image_path_list) -> list:
    """
    构建OpenAI API的消息格式，交替包含文本和图像
    
    参数:
        text_list: 文本列表
        image_path_list: 图像路径列表
        
    返回:
        格式化的消息列表，用于API请求
    """
    content = []
    image_index = 0
    for text in text_list:
        if image_index < len(image_path_list):
            image = encode_image(image_path_list[image_index])
            image_index += 1
            content.append({"type": "text", "text": text})
            content.append({"type": "image_url", "image_url": {"url": image}})
        else:
            content.append({"type": "text", "text": text})

    message = [{
        "role": "user",
        "content": content
    }]

    return message


def get_gpt_4o(messages, config_path: str = "configs/gpt_config.yaml") -> Optional[str]:
    """
    调用OpenAI GPT-4o API进行推理
    
    参数:
        messages: 格式化的消息列表
        config_path: 配置文件路径，包含API密钥和模型参数
        
    返回:
        API响应的文本内容，如果出错则返回None
    """
    try:
        # 读取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        openai.api_key = config['api_key']  # 设置API密钥
         
        # 获取模型配置参数，如果不存在则使用默认值
        model = config.get('model', 'gpt-4o')
        temperature = config.get('temperature', 0.7)
        max_tokens = config.get('max_tokens', 1000)

        # 调用API
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API error: {str(e)}")
        return None
    
class GPTScorer:
    """
    使用GPT模型对任务执行情况进行评分的类
    """
    def __init__(self):
        pass


    def get_score(self, ann):
        """
        获取对特定任务步骤的评分
        
        参数:
            ann: 包含任务信息、动作描述和图像的注释字典
            
        返回:
            GPT模型的评分结果
        """
        task = ann["task"]  # 获取任务描述

        # 构建历史动作描述，在每个描述前添加<image>标记
        history_action_desc = ""
        for action_desc in ann["action_desc_list"]:
            history_action_desc += f"\n<image>\n{action_desc}"
        
        # 使用预定义的提示模板格式化输入
        task_describe = prompt_score_system + prompt_score_user.format(task, history_action_desc, f"\n<image>\n{ann['action_desc_list'][ann['step_id']]}")
        
        # 分割文本和图像
        texts, images = task_describe.split("<image>")[:-1], ann["add_point_image_list"] + [ann["add_point_image_list"][ann["step_id"]]]
        
        # 构建消息并调用API
        messages = get_message(texts, images)
        response = get_gpt_4o(messages)
        
        return response.choices[0].message.content

def process_image(image_path):
    """
    处理图像：调整大小为原来的1/4
    
    参数:
        image_path: 图像文件路径
        
    返回:
        处理后的PIL图像对象
    """
    image = Image.open(image_path, 'r')
    image = image.resize((image.width // 4, image.height // 4))  # 将图像尺寸缩小为原来的1/4
    
    # 将图像保存到内存缓冲区
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    # 从缓冲区重新加载图像
    buffer.seek(0)
    image_reloaded = Image.open(buffer)
    return image_reloaded
