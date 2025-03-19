import utils
from torch.utils.data import Dataset
import numpy as np


class DummyDataset(Dataset):
    """
    虚拟数据集类，继承自PyTorch的Dataset类
    用于创建一个简单的数据集，只保留指定的键值对
    """
    def __init__(self, anns, keys):
        """
        初始化虚拟数据集
        
        参数:
            anns: 原始数据注释列表
            keys: 需要保留的键列表
        """
        self.anns = []
        for ann in anns:
            self.anns.append({k:v for k, v in ann.items() if k in keys})

    def __len__(self):
        """返回数据集的大小"""
        return len(self.anns)

    def __getitem__(self, idx):
        """根据索引获取数据集中的项目"""
        return self.anns[idx]


class ReplayBuffer:
    """
    经验回放缓冲区类
    用于存储和采样训练数据，常用于强化学习
    """
    def __init__(self, batch_size=2, capacity=10000):
        """
        初始化回放缓冲区
        
        参数:
            batch_size: 批次大小，默认为2
            capacity: 缓冲区容量，默认为10000
        """
        self.max_size = capacity
        self.current_size = 0   
        self.batch_size = batch_size

        # 初始化各种数据存储为None，稍后会创建
        self.critic_images = None  # 评论家网络的图像输入
        self.critic_inputs = None  # 评论家网络的文本输入
        self.policy_outputs = None  # 策略网络的输出
        self.policy_inputs = None  # 策略网络的输入
        self.policy_images = None  # 策略网络的图像输入
        self.action_lists = None  # 动作列表
        self.tasks = None  # 任务描述
        self.step_ids = None  # 步骤ID

    def sample(self, batch_size=None):
        """
        从缓冲区随机采样一批数据
        
        参数:
            batch_size: 采样批次大小，如果为None则使用默认值
            
        返回:
            包含采样数据的字典
        """
        rand_indices = np.random.randint(0, self.current_size, size=batch_size) % self.max_size
        return {
            "critic_images": self.critic_images[rand_indices],
            "critic_inputs": self.critic_inputs[rand_indices],
            "policy_outputs": self.policy_outputs[rand_indices],
            "policy_inputs": self.policy_inputs[rand_indices],
            "policy_images": self.policy_images[rand_indices],
            "action_lists": self.action_lists[rand_indices],
            "tasks": self.tasks[rand_indices],
            "step_ids": self.step_ids[rand_indices]
        }

    def __len__(self):
        """返回缓冲区当前大小"""
        return self.current_size

    def insert(
        self,
        policy_output,
        policy_input,
        policy_image,
        action_list,
        task,
        step_id, 
        critic_image="",
        critic_input="",
        **kwargs
    ):
        """
        向缓冲区插入新的数据项
        
        参数:
            policy_output: 策略网络的输出
            policy_input: 策略网络的输入
            policy_image: 策略网络的图像输入
            action_list: 动作列表
            task: 任务描述
            step_id: 步骤ID
            critic_image: 评论家网络的图像输入，默认为空字符串
            critic_input: 评论家网络的文本输入，默认为空字符串
            **kwargs: 其他可能的参数
        """
        # 如果缓冲区为空，初始化所有数组
        if self.critic_images is None:
            self.critic_images = np.array([''] * self.max_size, dtype="object")
            self.critic_inputs = np.array([''] * self.max_size, dtype="object")
            self.policy_outputs = np.array([''] * self.max_size, dtype="object")
            self.policy_inputs = np.array([''] * self.max_size, dtype="object")
            self.policy_images = np.array([''] * self.max_size, dtype="object")
            self.action_lists = np.array([''] * self.max_size, dtype="object")
            self.tasks = np.array([''] * self.max_size, dtype="object")
            self.step_ids = np.array([''] * self.max_size, dtype="object")

        # 将新数据插入到当前位置（循环使用缓冲区空间）
        self.critic_images[self.current_size % self.max_size] = critic_image
        self.critic_inputs[self.current_size % self.max_size] = critic_input
        self.policy_outputs[self.current_size % self.max_size] = policy_output
        self.policy_inputs[self.current_size % self.max_size] = policy_input
        self.policy_images[self.current_size % self.max_size] = policy_image
        self.action_lists[self.current_size % self.max_size] = action_list
        self.tasks[self.current_size % self.max_size] = task
        self.step_ids[self.current_size % self.max_size] = step_id

        # 增加当前大小计数
        self.current_size += 1


class AutoGUIDataset():
    """
    自动GUI数据集类
    用于处理自动化GUI交互任务的数据
    """
    def __init__(self, config, finish_task):
        """
        初始化AutoGUI数据集
        
        参数:
            config: 配置字典，包含数据路径等信息
            finish_task: 已完成任务的字典
        """
        # 定义查询格式，包含前一个动作和目标
        self.query_format = "Previous Action:\n{}\nGoal:\n{}"
        # 读取原始数据注释
        origin_anns = utils.read_jsonl(config["data_path"])

        # 筛选未完成的任务
        self.anns, tasks = [], []
        for ann in origin_anns:
            if ann["task"] not in tasks and ann["task"] not in finish_task.keys():
                self.anns.append({"task": ann["task"], "task_id": ann["ep_id"]})
                tasks.append(ann["task"])

        print(f"\tlen of tasks: {len(self.anns)}")
        

    def __len__(self):
        """返回数据集的大小"""
        return len(self.anns)


    def __getitem__(self, idx):
        """
        根据索引获取数据集中的项目
        
        返回:
            任务ID、任务描述和查询格式的元组
        """
        return self.anns[idx]["task_id"], self.anns[idx]["task"], self.query_format


def create_dataset(config, finish_task):
    """
    创建数据集的工厂函数
    
    参数:
        config: 配置字典
        finish_task: 已完成任务的字典
        
    返回:
        创建的数据集实例
    
    异常:
        如果指定的模型名称不支持，则抛出NotImplementedError
    """
    if config["model_name"] == "autogui":
        dataset = AutoGUIDataset(config, finish_task)
        return dataset
    else:
        raise NotImplementedError(f"dataset == {dataset}")

    