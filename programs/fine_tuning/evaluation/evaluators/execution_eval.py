"""
execution_eval.py - 可运行性指标评估

实现三个核心指标：
- XPR (Execution Pass Rate): 可执行率
- SAR (Standalone Runnability): 独立运行性
- SYR (Syntax Correctness Rate): 语法正确率
"""

from __future__ import annotations

import time
from typing import Dict, Any

from utils.sandbox import run_in_sandbox, check_syntax


class ExecutionEvaluator:
    """可运行性指标评估器"""

    def __init__(self, timeout: int = 60):
        """
        初始化评估器。

        Args:
            timeout: 沙盒执行超时时间（秒）
        """
        self.timeout = timeout

    def evaluate(self, code: str) -> Dict[str, Any]:
        """
        评估代码的可运行性。

        Returns:
            {
                "xpr": bool,           # 可执行（允许预装环境）
                "sar": bool,           # 独立运行（隔离环境）
                "syr": bool,           # 语法正确
                "exit_code_xpr": int,  # XPR 模式的退出码
                "exit_code_sar": int,  # SAR 模式的退出码
                "stderr_xpr": str,     # XPR 错误信息
                "stderr_sar": str,     # SAR 错误信息
                "exec_time": float,    # 执行耗时（秒）
            }
        """
        start_time = time.time()

        # 1. 语法检查（最快，最基础）
        syr = check_syntax(code)

        # 2. XPR：允许预装环境
        exit_code_xpr, stdout_xpr, stderr_xpr = run_in_sandbox(
            code, timeout=self.timeout, standalone=False
        )
        xpr = exit_code_xpr == 0

        # 3. SAR：隔离环境（更严格）
        exit_code_sar, stdout_sar, stderr_sar = run_in_sandbox(
            code, timeout=self.timeout, standalone=True
        )
        sar = exit_code_sar == 0

        exec_time = time.time() - start_time

        return {
            "xpr": xpr,
            "sar": sar,
            "syr": syr,
            "exit_code_xpr": exit_code_xpr,
            "exit_code_sar": exit_code_sar,
            "stderr_xpr": stderr_xpr,
            "stderr_sar": stderr_sar,
            "stdout_xpr": stdout_xpr,
            "stdout_sar": stdout_sar,
            "exec_time": exec_time,
        }

    def classify_error(self, stderr: str) -> str:
        """
        分类错误类型。

        用于分析失败原因分布。

        Args:
            stderr: 标准错误输出

        Returns:
            错误类型字符串
        """
        if not stderr:
            return "success"

        stderr_lower = stderr.lower()

        if "syntaxerror" in stderr_lower:
            return "syntax_error"
        elif "nameerror" in stderr_lower:
            return "name_error"
        elif "importerror" in stderr_lower or "modulenotfounderror" in stderr_lower:
            return "import_error"
        elif "timeoutexpired" in stderr_lower:
            return "timeout"
        elif "indentationerror" in stderr_lower:
            return "indentation_error"
        elif "typeerror" in stderr_lower:
            return "type_error"
        elif "attributeerror" in stderr_lower:
            return "attribute_error"
        elif "valueerror" in stderr_lower:
            return "value_error"
        elif "runtimeerror" in stderr_lower:
            return "runtime_error"
        else:
            return "other_error"
