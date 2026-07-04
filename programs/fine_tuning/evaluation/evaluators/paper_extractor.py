"""
paper_extractor.py - 论文组件提取器

使用 LLM 从论文文本中提取结构化的技术组件清单。
每篇论文只调用一次，结果缓存复用。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from openai import OpenAI


_EXTRACTION_SYSTEM = (
    "You are an expert in analyzing academic papers in AI/ML. "
    "You extract technical components from paper descriptions into structured JSON."
)

_EXTRACTION_PROMPT = """\
Given the paper's abstract and method description, extract the technical components \
that should be present in a PyTorch implementation of the core model.

# Paper Text
{paper_text}

# Task
Extract the following:
1. **required_modules**: PyTorch modules that should be used (e.g., "nn.MultiheadAttention", "nn.LayerNorm")
2. **required_patterns**: Code patterns/structures (e.g., "residual_connection", "positional_encoding")
3. **required_functions**: Essential methods (e.g., "forward", "__init__")
4. **expected_dims**: Key hyperparameters (e.g., "hidden_size", "num_heads")

# Output Format
Return a valid JSON object:
```json
{{
  "required_modules": ["nn.Module1", "nn.Module2", ...],
  "required_patterns": ["pattern1", "pattern2", ...],
  "required_functions": ["function1", "function2", ...],
  "expected_dims": ["dim1", "dim2", ...]
}}
```

# Rules
- Be specific (e.g., "nn.MultiheadAttention" not just "attention")
- Include only components explicitly mentioned or strongly implied in the paper
- For patterns, use descriptive names (e.g., "residual_connection", "layer_normalization")
- Limit to 5-10 items per category
- Return ONLY the JSON object, no additional text
"""


class PaperComponentExtractor:
    """论文技术组件提取器"""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "deepseek/deepseek-v3.2",
        api_base: Optional[str] = None,
    ):
        """
        初始化提取器。

        Args:
            cache_dir: 缓存目录路径
            api_key: OpenAI API 密钥（支持 OpenRouter）
            model: 使用的模型名称
            api_base: API Base URL（OpenRouter: https://openrouter.ai/api/v1）
        """
        self.cache_dir = Path(cache_dir or "outputs/eval/component_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        # 支持通过环境变量指定模型
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if self.api_key:
            # 支持自定义 base_url（用于 OpenRouter 等兼容服务）
            client_kwargs = {"api_key": self.api_key}
            if self.api_base:
                client_kwargs["base_url"] = self.api_base
            self.client = OpenAI(**client_kwargs)
        else:
            self.client = None
            print(
                "[WARNING] No OpenAI API key provided. Component extraction will be skipped."
            )

    def extract(self, paper_text: str, paper_id: str) -> Dict[str, Any]:
        """
        从论文提取组件清单。

        先检查缓存，未命中则调用 LLM。

        Args:
            paper_text: 论文文本（Abstract + Method）
            paper_id: 论文唯一标识符（用于缓存）

        Returns:
            {
                "required_modules": [...],
                "required_patterns": [...],
                "required_functions": [...],
                "expected_dims": [...],
            }
        """
        # 检查缓存
        cache_path = self.cache_dir / f"{paper_id}.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 未命中缓存，调用 LLM
        if not self.client:
            # 没有 API key，返回空清单
            return self._empty_component_list()

        components = self._call_llm(paper_text)

        # 缓存结果
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(components, f, indent=2, ensure_ascii=False)

        return components

    def _call_llm(self, paper_text: str) -> Dict[str, Any]:
        """调用 LLM 提取组件"""
        prompt = _EXTRACTION_PROMPT.format(paper_text=paper_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # 确保稳定输出
                max_tokens=1000,
            )

            content = response.choices[0].message.content.strip()

            # 提取 JSON
            components = self._parse_json(content)
            return components

        except Exception as e:
            print(f"[ERROR] LLM extraction failed: {e}")
            return self._empty_component_list()

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """从文本中解析 JSON"""
        import re

        # 尝试提取 ```json ... ``` 代码块
        json_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            text = match.group(1)

        # 尝试解析
        try:
            data = json.loads(text)
            # 验证结构
            if not isinstance(data, dict):
                return self._empty_component_list()

            # 确保所有字段存在
            for key in [
                "required_modules",
                "required_patterns",
                "required_functions",
                "expected_dims",
            ]:
                if key not in data or not isinstance(data[key], list):
                    data[key] = []

            return data

        except json.JSONDecodeError:
            print(f"[WARNING] Failed to parse JSON from LLM output")
            return self._empty_component_list()

    def _empty_component_list(self) -> Dict[str, Any]:
        """返回空的组件清单"""
        return {
            "required_modules": [],
            "required_patterns": [],
            "required_functions": [],
            "expected_dims": [],
        }
