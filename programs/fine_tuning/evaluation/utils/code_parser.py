"""
code_parser.py - 代码提取工具

从模型输出中提取 Python 代码，兼容多种格式：
- XML 格式: <file name="...">code</file>
- Markdown 代码块: ```python ... ```
- 纯文本代码
"""

from __future__ import annotations

import re
from typing import Optional


def extract_code(model_output: str) -> Optional[str]:
    """
    从模型输出中提取 Python 代码。

    尝试多种策略：
    1. XML <file> 标签（微调模型的输出格式）
    2. Markdown ```python``` 代码块
    3. 返回整个输出（假设是纯代码）

    Args:
        model_output: 模型的原始输出字符串

    Returns:
        提取的代码字符串，如果无法提取则返回 None
    """
    if not model_output or not model_output.strip():
        return None

    # 策略 1: XML <file> 标签
    # 匹配 <file name="xxx">code</file>
    xml_pattern = r'<file\s+name="[^"]*">\s*(.*?)\s*</file>'
    xml_matches = re.findall(xml_pattern, model_output, re.DOTALL)
    if xml_matches:
        # 返回第一个文件的内容（通常是核心模型文件）
        return xml_matches[0].strip()

    # 策略 2: Markdown 代码块
    # 匹配 ```python ... ``` 或 ``` ... ```
    md_pattern = r"```(?:python)?\s*\n(.*?)\n```"
    md_matches = re.findall(md_pattern, model_output, re.DOTALL)
    if md_matches:
        # 如果有多个代码块，优先返回最长的
        return max(md_matches, key=len).strip()

    # 策略 3: 纯文本
    # 清理可能的多余空行和头尾空白
    cleaned = model_output.strip()

    # 简单启发式：如果包含 Python 关键字，认为是代码
    python_keywords = ["import", "def ", "class ", "from ", "if __name__"]
    if any(kw in cleaned for kw in python_keywords):
        return cleaned

    # 兜底：返回清理后的文本（即使可能不是代码）
    return cleaned if cleaned else None


def extract_all_files(model_output: str) -> dict[str, str]:
    """
    从模型输出中提取所有文件（适用于多文件输出）。

    Args:
        model_output: 模型的原始输出字符串

    Returns:
        {filename: code} 字典
    """
    files = {}

    # 匹配所有 <file name="xxx">code</file>
    xml_pattern = r'<file\s+name="([^"]*)">\s*(.*?)\s*</file>'
    xml_matches = re.findall(xml_pattern, model_output, re.DOTALL)

    for filename, code in xml_matches:
        files[filename] = code.strip()

    return files


def is_likely_code(text: str) -> bool:
    """
    启发式判断文本是否像 Python 代码。

    Args:
        text: 要判断的文本

    Returns:
        True 如果看起来像代码
    """
    if not text or not text.strip():
        return False

    # 检查是否包含 Python 关键字
    python_indicators = [
        "import ",
        "from ",
        "def ",
        "class ",
        "if __name__",
        "return ",
        "self.",
        "torch.",
        "nn.",
    ]

    return any(ind in text for ind in python_indicators)
