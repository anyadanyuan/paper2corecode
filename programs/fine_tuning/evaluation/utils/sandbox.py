"""
sandbox.py - 沙盒执行逻辑

从 validate_xkg.py 复制并改进的沙盒执行功能。
支持两种模式：
- standalone=False: 允许预装环境（用于 XPR 指标）
- standalone=True: 隔离环境执行（用于 SAR 指标）
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Tuple


def run_in_sandbox(
    code: str, timeout: int = 60, standalone: bool = False
) -> Tuple[int, str, str]:
    """
    在隔离子进程中执行 Python 代码。

    Args:
        code: 要执行的 Python 代码
        timeout: 超时时间（秒）
        standalone: 是否隔离环境（True 时环境变量更严格）

    Returns:
        (exit_code, stdout, stderr)
        exit_code == 0 表示成功执行
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        # 构建环境变量
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        if standalone:
            # SAR 模式：更严格的隔离
            # 移除可能影响 import 的环境变量
            env.pop("PYTHONPATH", None)

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return result.returncode, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return -1, "", f"TimeoutExpired after {timeout}s"
    except Exception as e:
        return -1, "", str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def check_syntax(code: str) -> bool:
    """
    检查代码语法是否正确（用于 SYR 指标）。

    Args:
        code: 要检查的 Python 代码

    Returns:
        True 如果语法正确，False 否则
    """
    import ast

    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        # 其他异常（如 encoding 问题）也视为语法错误
        return False
