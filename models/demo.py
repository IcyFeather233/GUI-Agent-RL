import os
import sys
sys.path.append(os.getcwd())  # 将当前工作目录添加到系统路径
import torch
import gradio as gr  # 导入Gradio用于创建Web界面
from accelerate import Accelerator  # 导入Accelerator用于模型加速
import argparse
import spaces
from models.agent import AutoUIAgent
from train_rl import DigiRLTrainer


@spaces.GPU()  # 装饰器，指定在GPU上运行
def predict(text, image_path):
    """
    预测函数，处理输入的文本和图像
    Args:
        text: 用户输入的文本
        image_path: 图像文件路径
    Returns:
        模型生成的动作序列
    """
    # 处理图像特征，只使用最后1408维特征
    image_features = image_features = torch.stack([trainer.image_process.to_feat(image_path)[..., -1408:]])

    # 使用agent获取预测结果，将图像特征转换为bfloat16格式
    raw_actions = trainer.agent.get_action([text], image_features.to(dtype=torch.bfloat16))
    
    return raw_actions[0]


def main(model_name):
    """
    主函数，负责初始化模型和启动Web界面
    Args:
        model_name: 模型名称或路径
    """
    global trainer  # 声明全局变量trainer
    
    # 初始化Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    print("### load AutoUIAgent")

    # 初始化AutoUIAgent，设置模型参数
    agent = AutoUIAgent(
        device=device,
        accelerator=accelerator,
        do_sample=True,  # 启用采样
        temperature=1.0,  # 设置温度参数
        max_new_tokens=128,  # 设置最大生成token数
        policy_lm="checkpoints/Auto-UI-Base",  # 策略模型路径
        critic_lm="checkpoints/critic_1218/merge-520",  # 评论家模型路径
    )

    # 初始化DigiRL训练器
    trainer = DigiRLTrainer(
        agent=agent,
        accelerator=accelerator,
        tokenizer=agent.tokenizer
    )

    # 如果不是使用基础autoui模型，则加载指定的检查点
    if model_name != "autoui":
        print(f"### loading the checkpoint: {model_name}")
        trainer.load(model_name)

    # 创建Gradio界面
    demo = gr.Interface(
        fn=predict,  # 设置预测函数
        inputs=[
            # 添加文本输入框
            gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.'),
            # 添加图像输入框
            gr.Image(type="filepath", label="Image Prompt", value=None),
        ],
        outputs="text"  # 设置输出类型为文本
    )

    # 启动Gradio界面
    demo.launch(share=True, show_error=True)  # share=True允许生成公共URL，show_error=True显示详细错误信息


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)  # 添加模型路径参数

    # 解析命令行参数
    args = parser.parse_args()
    
    print(f"### model_name: {args.model_path}")
    main(args.model_path)  # 调用主函数
