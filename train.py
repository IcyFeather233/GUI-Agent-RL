import argparse
import yaml
import os
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from qwen_vl_utils import process_vision_info
import logging
import random

import utils
from dataset import ReplayBuffer, DummyDataset
from models.agent import ImageFeatureExtractor, Agent
from eval_tools.metrix import compute_matrix
from data_preprocess.utils import update_trajectory


class Trainer:
    """训练器类，用于管理模型训练过程"""
    def __init__(self,
        agent,          # 智能体实例
        accelerator,    # 加速器实例，用于分布式训练
        tokenizer,      # 分词器
        config         # 配置参数字典
    ):
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        # 初始化优化器，用于更新语言模型参数
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=float(config["lm_lr"]))

        self.criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.grad_accum_steps = config["grad_accum_steps"]  # 梯度累积步数
        self.gamma = config["gamma"]  # 折扣因子

        self.epochs = config["epochs"]  # 训练轮数

        self.step = 0  # 当前训练步数
        self.tau = config["tau"]  # 软更新参数
        self.max_grad_norm = config["max_grad_norm"]  # 梯度裁剪阈值
        self.accelerator = accelerator
        self.softmax = torch.nn.Softmax(dim = -1)  # softmax层
        self.image_process = ImageFeatureExtractor()  # 图像特征提取器


    def prepare(self):
        """准备优化器，使其支持分布式训练"""
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)


    def actor_loss(
        self,
        critic_images,      # 评论家网络的图像输入
        critic_inputs,      # 评论家网络的文本输入
        policy_inputs,      # 策略网络的输入
        policy_outputs,     # 策略网络的输出
        policy_images,      # 策略网络的图像输入
        validation=False,   # 是否为验证模式
        **kwargs
    ):
        """计算演员(Actor)的损失函数"""
        # 获取模型的数据类型和设备信息
        dtype = self.accelerator.unwrap_model(self.agent.model).dtype
        device = self.accelerator.unwrap_model(self.agent.model).device
        
        # 构建消息列表，包含文本和图像信息
        messages = []
        for critic_input, critic_image in zip(critic_inputs, critic_images):
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "text", "text": critic_input},
                    {"type": "image", "image": critic_image, "max_pixels": 56000}
                ]
            }])

        # 使用评论家处理器应用聊天模板
        texts = self.agent.critic_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 处理视觉信息
        vision_inputs = []
        for message in messages:
            vision_input, _ = process_vision_info(message)
            vision_inputs.append(vision_input)

        # 准备评论家网络的输入
        inputs = self.agent.critic_processor(
            text=texts,
            images=vision_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # 生成评论家网络的输出
        generated_ids = self.agent.critic.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 解码生成的token获取Q值
        q_values = self.agent.critic_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 将Q值转换为张量
        q_values = [int(val) for val in q_values]
        q_values = torch.tensor(q_values, dtype=dtype, requires_grad=True).to(device)

        q_values = q_values / 2  # 缩放Q值

        # 处理策略网络的图像特征
        policy_image_features = []
        for policy_image in policy_images:
            policy_image_feature = self.image_process.to_feat(image_path=policy_image).to(device, dtype=dtype)
            policy_image_features.append(policy_image_feature)
        policy_image_features = torch.stack(policy_image_features)

        # 计算策略网络的对数概率
        log_prob = self.agent.get_log_prob(policy_inputs, policy_image_features, policy_outputs).sum(dim=1).flatten()

        # 计算策略梯度损失
        pg_loss = - torch.mean(log_prob * q_values)

        # 如果不是验证模式，进行反向传播
        if not validation:
            self.accelerator.backward(pg_loss)

        return pg_loss.detach().cpu().item(), torch.mean(q_values).detach().cpu().item()


    def update_policy(self, buffer, is_validation, batch_size):
        """更新策略网络
        
        Args:
            buffer: 经验回放缓冲区
            is_validation: 是否为验证模式
            batch_size: 批次大小
        """
        logs = []  # 记录训练日志

        self.step += 1
        # 从buffer中采样数据
        data = [buffer.sample(1) for _ in range(self.grad_accum_steps * batch_size)]

        # 处理采样数据的格式
        for d in data:
            for k, v in d.items():
                d[k] = v[0]

        # 定义数据加载需要的键
        keys = ["ep_id", "step_id", "policy_inputs", "policy_outputs", "policy_images", "critic_inputs", "critic_images"]
        dataloader = self.accelerator.prepare(DataLoader(DummyDataset(data, keys), batch_size=batch_size, shuffle=False))

        # 训练或验证过程
        self.lm_optimizer.zero_grad()  # 清空梯度
        losses, q_values = [], []
        if is_validation:  # 验证模式
            with torch.no_grad():
                for batch in dataloader:
                    loss, q_value = self.actor_loss(**batch, validation=True)
                    losses.append(loss)
                    q_values.append(q_value)
            logging.info(f"[val] step: {self.step}\tloss: {sum(losses) / len(losses):.2f}\tQ-values: {sum(q_values) / len(q_values):.4f}")
            logs.append({"step": self.step, "val loss": sum(losses) / len(losses), "val Q value": sum(q_values) / len(q_values)})
        else:  # 训练模式
            for batch in dataloader:
                loss, q_value = self.actor_loss(**batch)
                losses.append(loss)
                q_values.append(q_value)
            logging.info(f"step: {self.step}\tloss: {sum(losses) / len(losses):.2f}\tQ-values: {sum(q_values) / len(q_values):.4f}")
            logs.append({"step": self.step, "train loss": sum(losses) / len(losses), "train Q value": sum(q_values) / len(q_values)})

            # 梯度裁剪和优化器步进
            self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.lm_optimizer.step()

        return logs


    def infer(self, data, batch_size):
        """模型推理函数
        
        Args:
            data: 输入数据
            batch_size: 批次大小
        """
        # 获取模型的数据类型和设备信息
        dtype = self.accelerator.unwrap_model(self.agent.model).dtype
        device = self.accelerator.unwrap_model(self.agent.model).device
        
        # 准备数据加载器
        keys = ["ep_id", "step_id", "policy_input", "policy_output", "policy_image"]
        dataloader = DataLoader(DummyDataset(data, keys), batch_size=batch_size, shuffle=False)
        dataloader = self.accelerator.prepare(dataloader)

        results = []
        for batch in dataloader:
            ep_ids, step_ids = batch["ep_id"], batch["step_id"]
            texts, groundtruths, image_paths = batch["policy_input"], batch["policy_output"], batch["policy_image"]
            
            # 处理图像特征
            image_features = []
            for image_path in image_paths:
                image_feature = self.image_process.to_feat(image_path=image_path).to(device, dtype=dtype)
                image_features.append(image_feature)
            image_features = torch.stack(image_features)

            # 获取模型预测结果
            outputs = self.agent.get_action(texts, image_features)

            # 整理结果
            for (output, groundtruth, ep_id, step_id) in zip(outputs, groundtruths, ep_ids, step_ids):
                results.append({"output": output, "groundtruth": groundtruth, "ep_id": ep_id, "step_id": step_id.item()})

        return results


    def save(self, path):
        """保存模型状态"""
        self.accelerator.save_state(path, safe_serialization=False)


    def load(self, path):
        """加载模型状态"""
        self.accelerator.load_state(path)


def train(
    agent,          # 智能体实例
    accelerator,    # 加速器实例
    config         # 配置参数
):
    """训练主函数"""
    # 初始化训练器
    trainer = Trainer(
        agent=agent,
        accelerator=accelerator,
        tokenizer=agent.tokenizer,
        config=config
    )

    batch_size = config["batch_size"]
    # 读取训练数据
    all_trajectories = utils.read_jsonl(config["data_path"])

    # 准备模型和训练器
    agent.prepare()
    trainer.prepare()

    print(f"### all trajectories: {len(all_trajectories)}")

    # 划分训练集和验证集
    logs = []
    train_trajectories = all_trajectories[:int(len(all_trajectories) * 0.95)]
    val_trajectories = all_trajectories[int(len(all_trajectories) * 0.95):]
    random.shuffle(train_trajectories)
    sample_num = config["batch_size"] * config["grad_accum_steps"]

    # 开始训练循环
    for epoch in range(config["epochs"]):
        print(f"### epoch {epoch}")
        for train_step in range(len(train_trajectories) // sample_num):
            # 采样训练数据
            sample_trajectories = train_trajectories[train_step * sample_num: (train_step + 1) * sample_num]

            # 进行推理并更新轨迹
            results = trainer.infer(sample_trajectories, batch_size)
            sample_trajectories = update_trajectory(sample_trajectories, results)
            
            # 创建经验回放缓冲区
            replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(sample_trajectories))
            for d in sample_trajectories:
                replay_buffer.insert(**d)

            # 更新策略
            logs.extend(trainer.update_policy(replay_buffer, is_validation=False, batch_size=batch_size))

        # 验证阶段
        results = trainer.infer(val_trajectories, batch_size)
        val_trajectories = update_trajectory(val_trajectories, results)
        validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(val_trajectories))
        for d in val_trajectories:
            validation_buffer.insert(**d)
        logs.extend(trainer.update_policy(validation_buffer, is_validation=True, batch_size=batch_size))

        # 保存模型和训练日志
        if accelerator.is_main_process:
            save_path = config["save_path"]
            print("### saving")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(os.path.join(save_path, f"epoch_{epoch}")):
                os.mkdir(os.path.join(save_path, f"epoch_{epoch}"))
            trainer.save(os.path.join(save_path, f"epoch_{epoch}"))
            utils.write_jsonl(logs, os.path.join(save_path, f"epoch_{epoch}", "train_log.jsonl"))
            utils.plot_loss(os.path.join(save_path, f"epoch_{epoch}"), keys=["train loss", "train Q value", "val loss", "val Q value"])


def evaluation(
    agent,          # 智能体实例
    accelerator,    # 加速器实例
    config         # 配置参数
):
    """评估函数"""
    # 设置结果保存路径
    result_dir = f"checkpoints/{config['test_task']}_result"
    result_wpath = os.path.join(result_dir, f"{config['test_task']}_{config['model_name']}_results.jsonl")

    print(f"### result path: {result_wpath}")
    
    # 读取评估数据
    anns = utils.read_jsonl(config['eval_data'])
    
    # 创建位置字典
    position_dict = {}
    for ann in anns:
        position_dict[f"{ann['ep_id']}_{ann['step_id']}"] = ann["position"]

    # 如果结果文件不存在，进行推理
    if not os.path.exists(result_wpath):
        trainer = Trainer(
            agent=agent,
            accelerator=accelerator,
            tokenizer=agent.tokenizer,
            config=config,
        )

        results = trainer.infer(anns, config["batch_size"])
        utils.write_jsonl(results, result_wpath)
    else:
        results = utils.read_jsonl(result_wpath)

    # 创建结果目录
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # 计算评估指标
    for file in os.listdir(result_dir):
        result_wpath = os.path.join(result_dir, file)
        results = utils.read_jsonl(result_wpath)
        print(f"================{result_wpath.split('/')[2]}================")
        compute_matrix(results, position_dict)


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--eval', action='store_true')  
    args = parser.parse_args()

    # 根据任务类型选择配置文件
    if args.task == "general":
        config = "configs/train_policy_general.yaml"
    else:
        config = "configs/train_policy_webshopping.yaml"

    # 读取配置文件
    with open(config, 'r') as file:
        config = yaml.safe_load(file)
        
    # 打印配置信息
    print(f"### config:")
    for k, v in config.items():
        print(f"\t{k}: {v}")
    
    # 初始化加速器
    accelerator = Accelerator()

    print("### Agent")

    # 初始化智能体
    agent = Agent(
        device=accelerator.device,
        accelerator=accelerator,
        temperature=config["temperature"],
        do_sample=config["do_sample"],
        policy_lm=config["policy_lm"],
        critic_lm=config["critic_lm"],
        max_new_tokens=config["max_new_tokens"]
    )

    # 根据参数选择评估或训练
    if args.eval:
        evaluation(agent=agent, accelerator=accelerator, config=config)
    else:
        train(agent=agent, accelerator=accelerator, config=config)


