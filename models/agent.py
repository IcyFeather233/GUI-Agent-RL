import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration, Blip2Model
from models.T5_model import T5ForMultimodalGeneration
from PIL import Image


class Agent(torch.nn.Module):
    """
    Agent类：实现了一个多模态智能体，能够处理文本和图像输入并生成响应
    继承自PyTorch的Module类，可以作为神经网络模型使用
    """
    def __init__(self, device, accelerator, policy_lm, critic_lm, do_sample, temperature, max_new_tokens):
        """
        初始化Agent对象
        
        参数:
            device: 运行模型的设备（CPU或GPU）
            accelerator: 用于模型加速的对象
            policy_lm: 策略语言模型的路径
            critic_lm: 评论家语言模型的路径
            do_sample: 是否使用采样生成文本
            temperature: 生成文本的温度参数，控制随机性
            max_new_tokens: 生成的最大新token数量
        """
        super(Agent, self).__init__()

        print(f"### load policy lm: {policy_lm}")
        # 加载策略模型（T5多模态生成模型）
        self.model = T5ForMultimodalGeneration.from_pretrained(policy_lm, torch_dtype=torch.bfloat16).to(device)
        
        print(f"### load critic && trajectory critic: {critic_lm}")
        # 加载评论家模型（Qwen2VL条件生成模型）
        self.critic = Qwen2VLForConditionalGeneration.from_pretrained(critic_lm, torch_dtype=torch.bfloat16).to(device)
        self.critic_processor = AutoProcessor.from_pretrained(critic_lm)

        # 初始化tokenizer，用于文本处理
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True)
        self.tokenizer.truncation_side = 'left'  # 设置截断方向为左侧
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置填充token为结束token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.dropout = torch.nn.Dropout(p=0.5)  # 设置dropout层，防止过拟合
        self.softmax = torch.nn.Softmax(dim=-1)  # softmax层，用于概率计算
        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
    

    def prepare(self):
        """
        使用accelerator准备模型，用于分布式训练或混合精度训练
        """
        self.model = self.accelerator.prepare(self.model)
        self.critic = self.accelerator.prepare(self.critic)


    def get_log_prob(self, text, image_features, target):
        """
        计算给定文本和图像特征生成目标文本的对数概率
        
        参数:
            text: 输入文本
            image_features: 图像特征
            target: 目标文本
            
        返回:
            对数概率
        """
        # 只使用图像特征的最后1408维
        image_features = image_features[..., -1408:]
        # 将文本转换为token ID
        text_ids = self.tokenizer(text, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        target_ids = self.tokenizer(target, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        # 使用模型计算输出
        outputs = self.model(
            input_ids = text_ids["input_ids"],
            image_ids = image_features,
            attention_mask = text_ids["attention_mask"],
            labels = target_ids["input_ids"]
        )
        
        # 计算预测概率
        prediction_probs = self.softmax(outputs.logits)
        # 提取目标token的预测概率
        selected_prediction_probs = torch.take_along_dim(prediction_probs, target_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        # 限制概率范围，避免数值问题
        selected_prediction_probs = torch.clamp(selected_prediction_probs, min=0.001, max=0.99)
        
        # 返回对数概率，并使用attention_mask过滤填充部分
        return torch.log(selected_prediction_probs) * target_ids["attention_mask"]
    
    
    def get_action(self, texts, image_features):
        """
        根据输入文本和图像特征生成动作（文本响应）
        
        参数:
            texts: 输入文本列表
            image_features: 图像特征
            
        返回:
            生成的文本响应列表
        """
        # 只使用图像特征的最后1408维
        image_features = image_features[..., -1408:]
        
        with torch.no_grad():  # 不计算梯度，用于推理
            # 将文本转换为token ID
            text_ids = self.tokenizer(texts, return_tensors='pt', padding=True, max_length=512, truncation=True).to(self.device)
            image_features = image_features.to(self.device)
            # 使用模型生成文本
            outputs = self.accelerator.unwrap_model(self.model).generate(
                **text_ids, image_ids=image_features,
                max_new_tokens=self.max_new_tokens, 
                do_sample=self.do_sample, 
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id
            ).cpu()
        
        # 解码生成的token ID为文本
        raw_actions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # 移除开头的换行符（如果有）
        for _ in range(3):
            raw_actions = [a[1:] if a.startswith('\n') else a for a in raw_actions]
        
        return raw_actions


class ImageFeatureExtractor:
    """
    图像特征提取器类：用于从图像中提取特征向量
    """
    def __init__(self):
        """
        初始化图像特征提取器
        加载BLIP2模型用于图像特征提取
        """
        self.model = Blip2Model.from_pretrained("./checkpoints/blip2-opt-2.7b").to("cuda")
        self.processor = AutoProcessor.from_pretrained("./checkpoints/blip2-opt-2.7b")

    def to_feat(self, image_path: str):
        """
        从图像路径提取特征
        
        参数:
            image_path: 图像文件路径
            
        返回:
            图像特征向量
        """
        with torch.no_grad():  # 不计算梯度，用于推理
            # 打开并处理图像
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to("cuda")
            
            # 提取图像特征
            image_features = self.model.get_image_features(**inputs).pooler_output[0]
            image_features = image_features.detach().cpu()  # 分离张量并移至CPU
            
        return image_features
