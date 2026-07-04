"""
distill_code_repository.py — Algorithm Code Purification Pipeline (ACPP).

实现 OriGen 风格的 Code-to-Code Augmentation，分为四个模块：
    1. ASTFilter:        基于 AST 的启发式粗筛，剔除工程脚手架
    2. CodePurifier:     单次 CoT LLM 调用，将候选代码净化为规范 PyTorch
    3. SandboxRunner:    在子进程中执行 Dummy Test，失败则自反思修复
    4. 辅助函数:         clone_repo, strip_main_block, wrap_file_tags

输出格式保持与旧版本一致：<file name="core_model.py">...</file>
"""

from __future__ import annotations

import ast
import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# 模块一：文件级过滤规则（与用户指定一致）
_FILE_FILTER_RE = re.compile(r"(?i)(test|utils|data|arg|config|log|plot|setup)\.py")

# 模块二/三：CoT 输出标签
_STEP1_OPEN = "<step_1_algorithm_specification>"
_STEP1_CLOSE = "</step_1_algorithm_specification>"
_STEP2_OPEN = "<step_2_clean_pytorch_implementation>"
_STEP2_CLOSE = "</step_2_clean_pytorch_implementation>"


# ═══════════════════════════════════════════════════════════════════════════════
#  公共辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def strip_main_block(code: str) -> str:
    """移除 `if __name__ == '__main__':` 及其后续所有内容。"""
    lines = code.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("if __name__")
            and "__main__" in stripped
            and stripped.rstrip().endswith(":")
        ):
            body = "\n".join(lines[:i])
            return body.rstrip()
    return code


def wrap_file_tags(code: str, filename: str = "core_model.py") -> str:
    """用 LLaMA-Factory / SFT 训练集中使用的 <file> 标签包裹代码。"""
    return f'<file name="{filename}">\n{code}\n</file>'


@contextlib.contextmanager
def cloned_repo(
    repo_url: str, tmp_dir: Path
) -> Generator[str, None, None]:
    """
    将仓库 clone 到 {tmp_dir}/repo_XXXXX 临时目录，退出上下文时自动清理。
    产生 clone 后的本地路径。
    """
    work_dir = tempfile.mkdtemp(dir=str(tmp_dir), prefix="repo_")
    try:
        print(f"  cloning {repo_url} ...")
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GCM_INTERACTIVE"] = "never"
        env["CI"] = "true"

        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, work_dir],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        yield work_dir
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Module 1: 基于 AST 的启发式粗筛
# ═══════════════════════════════════════════════════════════════════════════════

class ASTFilter:
    """遍历仓库，基于 AST 提取候选核心代码块。"""

    _SPECIAL_FUNC_NAMES = {"forward", "loss", "criterion", "step"}

    @staticmethod
    def _is_nn_module_class(node: ast.ClassDef) -> bool:
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module":
                return True
            if isinstance(base, ast.Name) and base.id == "Module":
                return True
        return False

    @staticmethod
    def _contains_backward_call(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Attribute) and func.attr == "backward":
                    return True
        return False

    @staticmethod
    def _is_candidate_function(node: ast.FunctionDef) -> bool:
        lowered = node.name.lower()
        if any(s in lowered for s in ASTFilter._SPECIAL_FUNC_NAMES):
            return True
        return ASTFilter._contains_backward_call(node)

    @staticmethod
    def extract_candidates(repo_path: str) -> List[str]:
        """
        从 repo_path 中提取候选核心代码块。
        返回字符串列表，每个元素形如：
            # FILE: src/model.py\n<class/function source>
        """
        repo_path_obj = Path(repo_path)
        candidates: List[str] = []

        for py_file in repo_path_obj.rglob("*.py"):
            # 跳过隐藏目录、__pycache__、第三方 vendored 代码
            parts = py_file.relative_to(repo_path_obj).parts
            if any(part.startswith(".") or part == "__pycache__" for part in parts):
                continue

            if _FILE_FILTER_RE.search(py_file.name):
                continue

            try:
                source = py_file.read_text(encoding="utf-8")
            except Exception:
                continue

            # 文件过大说明糅杂严重，直接跳过
            if len(source.splitlines()) > 2000:
                continue

            try:
                tree = ast.parse(source)
            except Exception:
                continue

            rel_path = py_file.relative_to(repo_path_obj).as_posix()
            extracted_nodes: List[ast.AST] = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and ASTFilter._is_nn_module_class(node):
                    extracted_nodes.append(node)
                elif isinstance(node, ast.FunctionDef) and ASTFilter._is_candidate_function(node):
                    # 类内部的方法已通过 ClassDef 提取，避免重复
                    extracted_nodes.append(node)

            for node in extracted_nodes:
                try:
                    snippet = ast.unparse(node)
                except Exception:
                    continue

                lines = snippet.splitlines()
                if len(lines) < 10 or len(lines) > 800:
                    continue

                header = f"# FILE: {rel_path}"
                candidates.append(f"{header}\n{snippet}")

        # 优先保留完整的 nn.Module 类（通常排在前面更长也更重要）
        candidates.sort(key=lambda s: len(s), reverse=True)
        return candidates


# ═══════════════════════════════════════════════════════════════════════════════
#  Module 2+3: 语义逆向 + 正向纯净重构（单次 CoT API 调用）
# ═══════════════════════════════════════════════════════════════════════════════

class CodePurifier:
    """通过单次 CoT 调用，将候选代码先提炼为算法规格，再重构为纯净 PyTorch。"""

    def __init__(
        self,
        model_name: str = "deepseek/deepseek-v4-pro",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = base_url or OPENROUTER_BASE_URL
        self.model_name = model_name
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Place it in the .env file at project root."
            )

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    # ── Prompts ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are an expert deep learning researcher and PyTorch architect. "
            "Your task is to analyze raw PyTorch code extracted from a public repository and "
            "reconstruct it into the purest possible algorithmic implementation.\n\n"
            "You MUST follow the exact two-step output format below:\n\n"
            f"{_STEP1_OPEN}\n"
            "1. Use natural language and/or concise pseudo-code to describe the core algorithm.\n"
            "2. Clearly specify input tensor shapes, intermediate transformations, and output shapes.\n"
            "3. Completely IGNORE engineering details such as DistributedDataParallel, "
            ".to(device), wandb/tensorboard logging, argparse, exception handling, data loaders.\n"
            f"{_STEP1_CLOSE}\n\n"
            f"{_STEP2_OPEN}\n"
            "1. Write a self-contained, clean PyTorch nn.Module implementation.\n"
            "2. Imports are restricted to: torch, torch.nn, torch.nn.functional, math, typing.\n"
            "3. Do NOT include training loops, data loading, device placement, distributed code, logging.\n"
            "4. At the end, include an `if __name__ == '__main__':` block that instantiates the model, "
            "creates a dummy input tensor with an appropriate shape, runs a forward pass, "
            "and asserts that the output shape is as expected.\n"
            f"{_STEP2_CLOSE}"
        )

    @staticmethod
    def _user_prompt(candidate_code: str) -> str:
        return f"[Raw Code Candidates]:\n\n{candidate_code}\n\n" \
               f"Now produce your response in the required {_STEP1_OPEN} ... {_STEP2_CLOSE} format."

    @staticmethod
    def _reflection_system_prompt() -> str:
        return (
            "You are an expert PyTorch bug-fixing engineer. "
            "The user will provide a PyTorch script and an execution error. "
            "Fix the bug (especially tensor shape mismatches) and return ONLY the corrected complete code. "
            "Do not add explanations."
        )

    @staticmethod
    def _reflection_user_prompt(code: str, error_message: str) -> str:
        return (
            f"[Previous Code]:\n{code}\n\n"
            f"[Execution Error]:\n{error_message[-2000:]}\n\n"
            "Return the fixed complete code only."
        )

    # ── Public API ──────────────────────────────────────────────────────────────

    def purify(self, candidate_code: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(candidate_code)},
            ],
        )
        raw = response.choices[0].message.content or ""
        return self._extract_clean_code(raw)

    def reflect(self, code: str, error_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": self._reflection_system_prompt(),
                },
                {
                    "role": "user",
                    "content": self._reflection_user_prompt(code, error_message),
                },
            ],
        )
        return response.choices[0].message.content or ""

    # ── Parsing helper ──────────────────────────────────────────────────────────

    @classmethod
    def _extract_clean_code(cls, raw_response: str) -> str:
        # 优先精确匹配 step 2 标签
        m = re.search(
            rf"{re.escape(_STEP2_OPEN)}(.*?){re.escape(_STEP2_CLOSE)}",
            raw_response,
            flags=re.DOTALL,
        )
        if m:
            return m.group(1).strip()

        # fallback: 如果有 step 1 结束标签，取其后的内容；否则返回全部
        m = re.search(
            rf"{re.escape(_STEP1_CLOSE)}(.*)$",
            raw_response,
            flags=re.DOTALL,
        )
        if m:
            return m.group(1).strip()

        return raw_response.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  Module 4: 沙盒执行与带反馈的自反思
# ═══════════════════════════════════════════════════════════════════════════════

class SandboxRunner:
    """在子进程中执行 Dummy Test，若失败则让 LLM 自反思修复。"""

    @staticmethod
    def verify_and_reflect(
        code: str,
        purifier: CodePurifier,
        max_retries: int = 2,
        tmp_dir: Path = Path("data/tmp"),
        timeout: int = 15,
    ) -> Tuple[str, bool]:
        """
        执行 Dummy Test；失败时最多重试 max_retries 次。
        返回 (最终代码, 是否通过)。
        """
        tmp_dir.mkdir(parents=True, exist_ok=True)
        current = code

        for attempt in range(max_retries + 1):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", dir=str(tmp_dir), delete=False
            ) as f:
                f.write(current)
                temp_path = f.name

            try:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ""  # 避免子进程占用 GPU
                env["TOKENIZERS_PARALLELISM"] = "false"
                env["PYTHONUNBUFFERED"] = "1"

                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )

                if result.returncode == 0:
                    print("  [OK] sandbox test passed")
                    return current, True

                error = result.stderr or result.stdout or "unknown error"
                print(f"  [FAIL] sandbox test failed (attempt {attempt + 1}/{max_retries + 1})")

                if attempt >= max_retries:
                    print("  [WARN] max retries reached; skipping")
                    return current, False

                print("  [RETRY] self-reflecting ...")
                current = purifier.reflect(current, error)

            except subprocess.TimeoutExpired:
                print(f"  [FAIL] sandbox test timed out after {timeout}s")
                if attempt >= max_retries:
                    return current, False
                current = purifier.reflect(current, f"Execution timed out after {timeout} seconds.")
            except Exception as exc:
                print(f"  [FAIL] sandbox runner error: {exc}")
                return current, False
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        return current, False
