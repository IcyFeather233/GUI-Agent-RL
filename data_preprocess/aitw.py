# 导入操作系统相关功能，用于文件路径操作
import os
# 导入系统模块，用于修改Python路径
import sys
# 将当前工作目录添加到Python路径中，确保可以导入项目中的其他模块
sys.path.insert(0, os.getcwd())

# 导入tqdm库，用于在控制台显示进度条
from tqdm import tqdm
# 导入JSON处理库，用于读写JSON格式数据
import json
# 导入线程模块，用于多线程处理
import threading
# 导入随机数模块，可能用于数据打乱或采样
import random
# 导入并发处理模块中的线程池执行器和结果收集函数
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入自定义工具模块
import utils
# 从数据预处理工具模块导入动作类型字典和转换函数
from data_preprocess.utils import action_type_dict, to_autoui, action_dict_to_class
# 导入评价系统的提示模板
from data_preprocess.prompt import prompt_critic_system, prompt_critic_user
# 导入GPT评分器类
from data_preprocess.gpt import GPTScorer


# ann表示annotation（标注）
def get_unfinish_anns(anns, rpath):
    """
    获取未完成处理的标注数据
    
    参数:
    anns: 所有标注数据列表
    rpath: 已完成处理的标注数据保存路径
    
    返回:
    未完成处理的标注数据列表
    """
    # 检查已完成标注的文件是否存在
    if os.path.exists(rpath):
        # 初始化未完成标注列表
        unfinish_anns = []
        # 读取已完成的标注数据
        finish_anns = utils.read_jsonl(rpath)
        # 创建已完成标注的唯一ID列表，格式为"ep_id_step_id"
        finish_ids = [f"{ann['ep_id']}_{ann['step_id']}" for ann in finish_anns]
        # 遍历所有标注数据
        for ann in anns:
            # 构建当前标注的唯一ID
            if f"{ann['ep_id']}_{ann['step_id']}" in finish_ids:
                # 如果ID已在完成列表中，则跳过
                pass
            else:
                # 否则添加到未完成列表
                unfinish_anns.append(ann)
        # 打印已完成和未完成标注的数量
        print(f"### finish anns: {len(finish_anns)} unfinish: {len(unfinish_anns)}")
        # 返回未完成的标注列表
        return unfinish_anns
    else:
        # 如果已完成标注文件不存在，则返回所有标注
        return anns


class AITW:
    """
    AITW数据处理类，用于处理AI Task Wizard数据集
    """
    def __init__(self, split: str, part: str):
        """
        初始化AITW数据处理器
        
        参数:
        split: 数据集划分，如'train'或'val'
        part: 数据集部分，如'general'或'webshopping'
        """
        # 设置图像目录路径
        self.image_dir = "images/aitw_images"
        # 保存数据集划分（训练集或验证集）
        self.split = split

        # 保存数据集部分（general或webshopping）
        self.part = part
        # 检查并创建标注保存目录
        if not os.path.exists(f"data/aitw_anns"):
            os.mkdir(f"data/aitw_anns")
        # 初始化GPT评分器实例
        self.gpt = GPTScorer()
    

    def get_unfold_data(self):
        """
        展开原始数据，将每个episode的每个step转换为单独的样本
        处理原始数据并生成展开的步骤数据，每个步骤包含完整的上下文信息
        """
        # 读取原始标注数据，根据split和part选择相应部分
        anns = utils.read_json(f"data/aitw_anns/aitw_{self.split}.json")[self.part]
        # 初始化存储展开步骤的列表
        steps = []
        # 遍历每个交互剧集(episode)，显示进度条
        for episode in tqdm(anns):
            # 初始化各种列表，用于存储每个episode的所有步骤信息
            action_list, action_type_list, image_list, add_point_image_list = [], [], [], []
            action_desc_list, action_desc_all_list = [], []
            # 遍历episode中的每个步骤
            for step_id, step in enumerate(episode):
                # 构建图像文件名
                image_filename = f"{step['img_filename']}.png"
                # 构建完整图像路径，并统一路径分隔符为"/"
                image_path = os.path.join(self.image_dir, image_filename).replace("\\", "/")
                # 检查图像文件是否存在
                if not os.path.exists(image_path):
                    # 如果图像不存在，打印警告并跳过此步骤
                    print(f"{image_path} image not found")
                    continue
                
                # 构建所有模型需要的动作字典
                action_dict = {
                    "action_type": action_type_dict[step["action_type_text"]],  # 动作类型编码
                    "touch_point": step["touch"],  # 触摸坐标
                    "lift_point": step["lift"],  # 抬起坐标
                    "typed_text": step["type_text"]  # 输入的文本
                }
                # 将动作字典添加到动作列表
                action_list.append(action_dict)
                # 将动作类型添加到动作类型列表
                action_type_list.append(action_dict["action_type"])
                # 将图像路径添加到图像列表
                image_list.append(image_path)
                # 将动作字典转换为动作类对象
                action = action_dict_to_class(action_dict)

                # 为不同模型输入准备动作描述
                # 简化版动作描述，包含步骤ID
                action_desc_list.append(f"step {step_id}: " + to_autoui(action, all_dict=False))
                # 完整版动作描述，包含所有细节
                action_desc_all_list.append(to_autoui(action, all_dict=True))
                
                # 在截图上可视化动作点，并添加到带标记点的图像列表
                add_point_image_list.append(utils.add_visilize2screenshot(image_path, action_list[-1], "score"))
            
            # 为每个步骤创建完整的样本数据
            for step_id, step in enumerate(episode):
                steps.append({
                    "ep_id": step["ep_id"],  # 剧集ID
                    "step_id": step_id,  # 步骤ID
                    "task": step["goal"],  # 任务目标描述
                    "action_list": action_list,  # 整个剧集的动作列表
                    "action_type_list": action_type_list,  # 整个剧集的动作类型列表
                    "image_list": image_list,  # 整个剧集的图像路径列表
                    "add_point_image_list": add_point_image_list,  # 整个剧集的带标记点图像列表
                    "position": step["annot_position"],  # 标注位置信息
                    "action_type": action_type_dict[step["action_type_text"]],  # 当前步骤的动作类型
                    "touch_point": step["touch"],  # 当前步骤的触摸坐标
                    "lift_point": step["lift"],  # 当前步骤的抬起坐标
                    "typed_text": step["type_text"],  # 当前步骤的输入文本
                    "action_desc_list": action_desc_list,  # 整个剧集的简化动作描述列表
                    "action_desc_all_list": action_desc_all_list  # 整个剧集的完整动作描述列表
                })
        
        # 将展开后的步骤数据保存为JSONL文件
        utils.write_jsonl(steps, f"data/aitw_anns/{self.part}_{self.split}.jsonl")

    
    def get_gpt_label(self):
        """
        使用GPT为每个步骤生成评价标签
        调用GPT模型为每个步骤生成评分和解释，用于训练评价模型
        """
        # 读取展开后的步骤数据
        anns = utils.read_jsonl(f"data/aitw_anns/{self.part}_{self.split}.jsonl")
        # 设置评价结果保存路径
        ann_wpath = f"data/aitw_anns/{self.part}_{self.split}_critic.jsonl"
        # 获取未处理的标注数据
        unfinish_anns = get_unfinish_anns(anns, ann_wpath)

        # 创建线程锁，用于保护文件写入操作
        write_lock = threading.Lock()

        def process_ann(ann):
            """
            处理单个标注的函数
            
            参数:
            ann: 单个标注数据
            
            返回:
            添加了评价信息的标注数据
            """
            # 使用GPT获取对当前步骤的评分
            response = self.gpt.get_score(ann)
            # 解析GPT响应，提取评分和解释
            response = utils.parse_response(response)
            # 将评分和解释保存到标注数据中
            ann["critic_output"], ann["critic_explanation"] = response["rating"], response["explanation"]

            # 构建对话历史，包含系统提示和用户输入
            conversations = [
                {"from": "human", "value": prompt_critic_system + prompt_critic_user.format(
                    ann["task"],  # 任务目标
                    "\n".join(ann["action_desc_list"][:ann["step_id"]]),  # 之前步骤的动作描述
                    ann["action_desc_list"][ann["step_id"]]  # 当前步骤的动作描述
                )},
                {"from": "gpt", "value": str(response["rating"])}  # GPT的评分响应
            ]
            # 保存对话输入到标注数据
            ann["critic_inputs"] = conversations
            # 保存评价使用的图像路径，统一路径分隔符
            ann["critic_images"] = ann["add_point_image_list"][ann["step_id"]].replace("\\", "/")

            # 返回添加了评价信息的标注数据
            return ann

        # 以追加模式打开结果文件，在线程外打开以避免冲突
        with open(ann_wpath, "a") as fout:
            # 创建线程池，最多8个工作线程并行处理
            with ThreadPoolExecutor(max_workers=8) as executor:
                # 为每个未处理的标注创建一个处理任务
                future_to_ann = {executor.submit(process_ann, ann): ann for ann in unfinish_anns}
                # 遍历完成的任务，显示进度条
                for future in tqdm(as_completed(future_to_ann), total=len(future_to_ann)):
                    # 获取对应的标注数据
                    ann = future_to_ann[future]
                    try:
                        # 获取处理结果
                        result = future.result()
                        # 使用锁保护文件写入操作
                        with write_lock:
                            # 将结果写入文件，每个结果占一行
                            fout.writelines(json.dumps(result) + "\n")
                    except Exception as exc:
                        # 捕获并打印处理过程中的异常
                        print(f'Error processing annotation {ann}: {exc}')
            # 关闭文件
            fout.close()


    def get_rl_data(self):
        """
        准备强化学习数据，包括策略输入和输出
        为每个步骤准备适合强化学习训练的数据格式
        """
        # 设置展开数据的读取路径
        ann_rpath = f"data/aitw_anns/{self.part}_{self.split}.jsonl"
        # 设置策略数据的保存路径
        ann_wpath = f"data/aitw_anns/{self.part}_{self.split}_policy.jsonl"
        # 读取展开的步骤数据
        anns = utils.read_jsonl(ann_rpath)

        # 遍历每个步骤数据，显示进度条
        for ann in tqdm(anns):
            # 构建之前步骤的动作描述，用换行符连接
            previous_actions = "\n".join(ann["action_desc_all_list"][:ann["step_id"]])
            # 保存当前步骤的图像路径
            ann["policy_image"] = ann["image_list"][ann["step_id"]]
            # 构建策略模型的输入，包括之前的动作和任务目标，移除单引号以避免解析问题
            ann["policy_input"] = f"Previous Action:\n{previous_actions}\nGoal:\n{ann['task']}".replace("'", "")
            # 构建策略模型的期望输出，包括动作计划和决策，移除单引号
            ann["policy_output"] = f"Action Plan: {ann['action_type_list'][ann['step_id']:]} ; Action Decision: {ann['action_desc_all_list'][ann['step_id']]}".replace("'", "")

        # 将处理后的策略数据保存为JSONL文件
        utils.write_jsonl(anns, ann_wpath)


if __name__ == "__main__":
    aitw_data = AITW(split="train", part="general")
    aitw_data.get_unfold_data()
    aitw_data.get_gpt_label()
    aitw_data.get_rl_data()

    aitw_data = AITW(split="val", part="general")
    aitw_data.get_unfold_data()
    aitw_data.get_gpt_label()
    aitw_data.get_rl_data()

    aitw_data = AITW(split="train", part="webshopping")
    aitw_data.get_unfold_data()
    aitw_data.get_gpt_label()
    aitw_data.get_rl_data()

    aitw_data = AITW(split="val", part="webshopping")
    aitw_data.get_unfold_data()
    aitw_data.get_gpt_label()
    aitw_data.get_rl_data()