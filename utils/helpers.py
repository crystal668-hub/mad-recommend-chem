"""
===================================
辅助函数模块
功能：提供各种通用的辅助功能
===================================
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import hashlib


def load_config(config_path: str) -> Dict:
    """
    加载配置文件（支持YAML和JSON）
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Dict: 配置字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 根据文件扩展名选择解析器
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    # 替换环境变量
    config = replace_env_variables(config)
    
    return config


def replace_env_variables(data: Any) -> Any:
    """
    递归替换配置中的环境变量占位符
    格式: ${VARIABLE_NAME}
    
    Args:
        data: 配置数据（可以是dict、list或str）
    
    Returns:
        Any: 替换后的数据
    """
    if isinstance(data, dict):
        return {k: replace_env_variables(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_env_variables(item) for item in data]
    elif isinstance(data, str):
        # 检查是否是环境变量占位符
        if data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            return os.getenv(env_var, data)
        return data
    else:
        return data


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    确保目录存在，不存在则创建
    
    Args:
        directory: 目录路径
    
    Returns:
        Path: 目录路径对象
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进空格数
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    从JSON文件加载数据
    
    Args:
        file_path: 文件路径
    
    Returns:
        Any: 加载的数据
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """
    生成时间戳字符串
    
    Args:
        format: 时间格式
    
    Returns:
        str: 时间戳字符串
    """
    return datetime.now().strftime(format)


def calculate_hash(text: str, algorithm: str = "md5") -> str:
    """
    计算文本的哈希值
    
    Args:
        text: 输入文本
        algorithm: 哈希算法 ("md5", "sha1", "sha256")
    
    Returns:
        str: 哈希值
    """
    hash_func = getattr(hashlib, algorithm)()
    hash_func.update(text.encode('utf-8'))
    return hash_func.hexdigest()


def format_component_list(components: List[str]) -> str:
    """
    格式化组分列表为可读字符串
    
    Args:
        components: 组分列表
    
    Returns:
        str: 格式化后的字符串
    """
    if not components:
        return "无"
    
    return "、".join(components)


def parse_component_string(component_str: str) -> List[str]:
    """
    解析组分字符串为列表
    支持多种分隔符：英文逗号/中文逗号、顿号、分号
    
    Args:
        component_str: 组分字符串
    
    Returns:
        List[str]: 组分列表
    """
    # 替换各种分隔符为统一的分隔符
    normalized = (
        str(component_str or "")
        .replace("，", ",")
        .replace("、", ",")
        .replace("；", ",")
        .replace(";", ",")
    )
    
    # 分割并去除空白
    components = [c.strip() for c in normalized.split(',') if c.strip()]
    
    return components


def validate_components(components: List[str], expected_count: int = 5) -> tuple[bool, str]:
    """
    验证组分列表的有效性
    
    Args:
        components: 组分列表
        expected_count: 期望的组分数量
    
    Returns:
        tuple: (是否有效, 错误信息)
    """
    if not components:
        return False, "组分列表为空"
    
    if len(components) != expected_count:
        return False, f"组分数量错误：期望{expected_count}个，实际{len(components)}个"
    
    # 检查是否有重复
    if len(components) != len(set(components)):
        return False, "组分列表中存在重复项"
    
    # 检查每个组分是否非空
    for comp in components:
        if not comp or not comp.strip():
            return False, "存在空的组分"
    
    return True, "组分列表有效"


def format_duration(seconds: float) -> str:
    """
    格式化时长为可读字符串
    
    Args:
        seconds: 秒数
    
    Returns:
        str: 格式化后的时长
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}小时"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本到指定长度
    
    Args:
        text: 输入文本
        max_length: 最大长度
        suffix: 截断后缀
    
    Returns:
        str: 截断后的文本
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def create_experiment_id(components: List[str]) -> str:
    """
    为实验创建唯一ID
    
    Args:
        components: 组分列表
    
    Returns:
        str: 实验ID
    """
    # 组合组分和时间戳
    component_str = "_".join(sorted(components))
    timestamp = generate_timestamp("%Y%m%d%H%M%S")
    
    # 计算哈希以缩短ID长度
    hash_value = calculate_hash(component_str)[:8]
    
    return f"exp_{timestamp}_{hash_value}"


def print_header(title: str, width: int = 60, char: str = "=") -> None:
    """
    打印格式化的标题头
    
    Args:
        title: 标题文本
        width: 总宽度
        char: 边框字符
    """
    print()
    print(char * width)
    print(title.center(width))
    print(char * width)
    print()


def print_section(title: str, content: Any = None) -> None:
    """
    打印格式化的章节
    
    Args:
        title: 章节标题
        content: 章节内容
    """
    print(f"\n{title}")
    print("-" * len(title))
    if content:
        print(content)


def dict_to_table(data: Dict, headers: tuple = ("键", "值")) -> str:
    """
    将字典转换为表格字符串
    
    Args:
        data: 字典数据
        headers: 表头
    
    Returns:
        str: 表格字符串
    """
    if not data:
        return "（无数据）"
    
    # 计算列宽
    key_width = max(len(str(k)) for k in data.keys())
    key_width = max(key_width, len(headers[0]))
    
    value_width = max(len(str(v)) for v in data.values())
    value_width = max(value_width, len(headers[1]))
    
    # 构建表格
    lines = []
    
    # 表头
    header_line = f"| {headers[0]:<{key_width}} | {headers[1]:<{value_width}} |"
    separator = f"|{'-' * (key_width + 2)}|{'-' * (value_width + 2)}|"
    
    lines.append(separator)
    lines.append(header_line)
    lines.append(separator)
    
    # 数据行
    for key, value in data.items():
        line = f"| {str(key):<{key_width}} | {str(value):<{value_width}} |"
        lines.append(line)
    
    lines.append(separator)
    
    return "\n".join(lines)


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    深度合并两个字典
    
    Args:
        dict1: 字典1
        dict2: 字典2
    
    Returns:
        Dict: 合并后的字典
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


