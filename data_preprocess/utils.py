from enum import Enum
from typing import Tuple
from dataclasses import dataclass
import re

import utils
from data_preprocess.prompt import prompt_critic_system, prompt_critic_user


class ActionType(Enum):
    """
    动作类型枚举类，定义了所有可能的操作类型
    """
    Idle=0          # 空闲状态
    DualPoint=1     # 点击操作（触摸和抬起）
    Type=2          # 输入文本
    GoBack=3        # 返回操作
    GoHome=4        # 回到主页
    Enter=5         # 回车键
    TaskComplete=6  # 任务完成
    TaskImpossible=7 # 任务无法完成
    Up=8            # 向上滑动
    Down=9          # 向下滑动
    Left=10         # 向左滑动
    Right=11        # 向右滑动


@dataclass
class AndroidAction():
    """
    安卓操作的数据类，包含操作类型和相关参数
    """
    action_type: ActionType                  # 操作类型
    touch_point: Tuple[float, float] = None  # 触摸点坐标 (x, y)
    lift_point: Tuple[float, float] = None   # 抬起点坐标 (x, y)
    typed_text: str = None                   # 输入的文本内容


# 操作类型的字典映射，将字符串映射到对应的操作类型
action_type_dict = {
    "type": "TYPE",
    "click": "DUAL_POINT",
    "press back": "PRESS_BACK",
    "press home": "PRESS_HOME",
    "press enter": "PRESS_ENTER",
    "status task complete": "STATUS_TASK_COMPLETE",
    "status task impossible": "STATUS_TASK_IMPOSSIBLE",
    "scroll down": "SCROLL_DOWN",
    "scroll up": "SCROLL_UP",
    "scroll left": "SCROLL_LEFT",
    "scroll right": "SCROLL_RIGHT",
}


# 滑动操作的坐标映射，定义了不同滑动方向的起点和终点坐标
scroll_map = {
    "up": [[0.8000, 0.5000], [0.2000, 0.5000]],
    "down": [[0.2000, 0.5000], [0.8000, 0.5000]],
    "left": [[0.5000, 0.8000], [0.5000, 0.2000]],
    "right": [[0.5000, 0.2000], [0.5000, 0.8000]]
}


def extract_scroll(action):
    """
    从触摸和抬起点判断滑动方向，并更新操作类型
    
    参数:
        action: AndroidAction对象
    
    返回:
        更新了action_type的AndroidAction对象
    """
    if action.touch_point == action.lift_point:
        return action
    
    drag_delta_x, drag_delta_y = action.lift_point[0] - action.touch_point[0], action.lift_point[1] - action.touch_point[1]
    if drag_delta_y == 0:
        if drag_delta_x < 0: action.action_type = ActionType.Up
        else: action.action_type = ActionType.Down
    elif drag_delta_x == 0:
        if drag_delta_y < 0: action.action_type = ActionType.Left
        else: action.action_type = ActionType.Right
    
    return action


def update_trajectory(anns, results):
    """
    更新轨迹数据，处理模型输出结果并添加到标注中
    
    参数:
        anns: 标注数据列表
        results: 模型输出结果列表
    
    返回:
        更新后的标注数据列表
    """
    for (result, ann) in zip(results, anns):
        new_action = autoui_translate_action(result["output"])
        try:
            new_action = extract_scroll(new_action)
        except:
            print(f"error get new action: {new_action}")
        
        new_action_desc = to_autoui(new_action, all_dict=False)
        
        history_action_desc = "\n".join(ann["action_desc_list"][:ann["step_id"] - 1])
        
        ann["critic_input"] = prompt_critic_system + prompt_critic_user.format(ann["task"], history_action_desc, new_action_desc)
        ann["policy_output"] = new_action_desc
        ann["critic_image"] = utils.add_visilize2screenshot(ann["policy_image"], new_action, "policy")

    return anns


def action_dict_to_class(action_dict):
    """
    将操作字典转换为AndroidAction类对象
    
    参数:
        action_dict: 包含操作信息的字典
    
    返回:
        AndroidAction对象
    """
    action_type = action_dict["action_type"]
    
    if action_type == 'DUAL_POINT':
        action_class = AndroidAction(action_type=ActionType.DualPoint, touch_point=action_dict["touch_point"][::-1], lift_point=action_dict["lift_point"][::-1])
    elif action_type == 'TYPE':
        action_class = AndroidAction(action_type=ActionType.Type, typed_text=action_dict["typed_text"])
    elif action_type == 'SCROLL_UP':
        return AndroidAction(action_type=ActionType.Up, touch_point=(0.5, 0.5), lift_point=(0.5, 0.2))
    elif action_type == 'SCROLL_DOWN':
        return AndroidAction(action_type=ActionType.Down, touch_point=(0.5, 0.2), lift_point=(0.5, 0.5))
    elif action_type == 'SCROLL_LEFT':
        return AndroidAction(action_type=ActionType.Left, touch_point=(0.8, 0.5), lift_point=(0.2, 0.5))
    elif action_type == 'SCROLL_RIGHT':
        return AndroidAction(action_type=ActionType.Right, touch_point=(0.2, 0.5), lift_point=(0.8, 0.5))
    elif action_type == 'PRESS_HOME':
        action_class = AndroidAction(action_type=ActionType.GoHome)
    elif action_type == 'PRESS_BACK':
        action_class = AndroidAction(action_type=ActionType.GoBack)
    elif action_type == 'PRESS_ENTER':
        action_class = AndroidAction(action_type=ActionType.Enter)
    elif action_type == 'STATUS_TASK_COMPLETE':
        action_class = AndroidAction(action_type=ActionType.TaskComplete)
    elif action_type == 'STATUS_TASK_IMPOSSIBLE':
        action_class = AndroidAction(action_type=ActionType.TaskImpossible)
    else:
        print(f"Action {action_dict} not supported yet.")
        action_class = AndroidAction(action_type=ActionType.Idle)
    
    return action_class


def autoui_translate_action(raw_action):
    """
    解析模型输出的原始操作字符串，转换为AndroidAction对象
    
    参数:
        raw_action: 模型输出的原始操作字符串
    
    返回:
        AndroidAction对象
    """
    try:
        action_str = raw_action.split("Action Decision: ")[1]
        action_type, touch_point_1, touch_point_2, lift_point_1, lift_point_2, typed_text = action_str.split(", ")
        touch_point = touch_point_1 + ", " + touch_point_2
        lift_point = lift_point_1 + ", " + lift_point_2
        action_type = action_type.split(": ")[1].strip('"')
        if action_type == 'DUAL_POINT':
            touch_point_yx = touch_point.split(": ")[1].strip('[]"')
            touch_point_yx = [float(num) for num in touch_point_yx.split(", ")]
            lift_point_yx = lift_point.split(": ")[1].strip('[]"')
            lift_point_yx = [float(num) for num in lift_point_yx.split(", ")]
            action_class = AndroidAction(action_type=ActionType.DualPoint, touch_point=touch_point_yx[::-1], lift_point=lift_point_yx[::-1])
        elif action_type == 'TYPE':
            text = typed_text.split(": ")[1].strip('"')
            action_class = AndroidAction(action_type=ActionType.Type, typed_text=text)
        elif action_type == 'PRESS_HOME':
            action_class = AndroidAction(action_type=ActionType.GoHome)
        elif action_type == 'PRESS_BACK':
            action_class = AndroidAction(action_type=ActionType.GoBack)
        elif action_type == 'PRESS_ENTER':
            action_class = AndroidAction(action_type=ActionType.Enter)
        elif action_type == 'STATUS_TASK_COMPLETE':
            action_class = AndroidAction(action_type=ActionType.TaskComplete)
        elif action_type == 'TASK_IMPOSSIBLE':
            action_class = AndroidAction(action_type=ActionType.TaskImpossible)
        else:
            print(f"Action {raw_action} not supported yet.")
            action_class = AndroidAction(action_type=ActionType.Idle)
    except:
        return AndroidAction(action_type=ActionType.GoHome)
    
    return action_class


def to_autoui(act: AndroidAction, all_dict):
    """
    将AndroidAction对象转换为字符串表示
    
    参数:
        act: AndroidAction对象
        all_dict: 布尔值，决定输出格式（完整字典或简化版）
    
    返回:
        操作的字符串表示
    """
    if all_dict:
        # 完整字典格式，包含所有参数
        if act.action_type in [ActionType.DualPoint, ActionType.Up, ActionType.Down, ActionType.Left, ActionType.Right]:
            return f'"action_type": "DUAL_POINT", "touch_point": "[{act.touch_point[1]:.4f}, {act.touch_point[0]:.4f}]", "lift_point": "[{act.lift_point[1]:.4f}, {act.lift_point[0]:.4f}]", "typed_text": ""'
        elif act.action_type == ActionType.Type:
            return f'"action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "{act.typed_text}"'
        elif act.action_type == ActionType.GoBack:
            return f'"action_type": "PRESS_BACK", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
        elif act.action_type == ActionType.GoHome:
            return f'"action_type": "PRESS_HOME", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
        elif act.action_type == ActionType.Enter:
            return f'"action_type": "PRESS_ENTER", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
        elif act.action_type == ActionType.TaskComplete or act.action_type == ActionType.TaskImpossible:
            return f'"action_type": "STATUS_TASK_COMPLETE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
        else:
            print(f"Action {act} not supported yet.")
            return ""
    else:
        # 简化格式，只包含必要参数
        if act.action_type == ActionType.DualPoint:
            return f'"action_type": "DUAL_POINT", "click_point": "[{act.touch_point[1]:.4f}, {act.touch_point[0]:.4f}]"'
        elif act.action_type == ActionType.Type:
            return f'"action_type": "TYPE", "typed_text": "{act.typed_text}"'
        elif act.action_type == ActionType.GoBack:
            return f'"action_type": "PRESS_BACK"'
        elif act.action_type == ActionType.GoHome:
            return f'"action_type": "PRESS_HOME"'
        elif act.action_type == ActionType.Enter:
            return f'"action_type": "PRESS_ENTER"'
        elif act.action_type == ActionType.TaskComplete or act.action_type == ActionType.TaskImpossible:
            return f'"action_type": "STATUS_TASK_COMPLETE"'
        elif act.action_type == ActionType.Up:
            return f'"action_type": "SCROLL_UP"'
        elif act.action_type == ActionType.Down:
            return f'"action_type": "SCROLL_DOWN"'
        elif act.action_type == ActionType.Left:
            return f'"action_type": "SCROLL_LEFT"'
        elif act.action_type == ActionType.Right:
            return f'"action_type": "SCROLL_RIGHT"'
        else:
            print(f"Action {act} not supported yet.")
            return ""


