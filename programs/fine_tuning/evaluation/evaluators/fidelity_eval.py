"""
fidelity_eval.py - 论文忠实度指标评估

实现三个核心指标：
- PCR (Paper Component Coverage Rate): 论文组件覆盖率
- AAR (API Alignment Rate): API 调用对齐率
- CodeBERTScore: 代码语义相似度
"""

from __future__ import annotations

import re
from typing import Dict, Any, Optional, List, Set

from .paper_extractor import PaperComponentExtractor


class FidelityEvaluator:
    """论文忠实度指标评估器"""

    def __init__(
        self,
        paper_extractor: Optional[PaperComponentExtractor] = None,
    ):
        """
        初始化评估器。

        Args:
            paper_extractor: 论文组件提取器（可选）
        """
        self.paper_extractor = paper_extractor or PaperComponentExtractor()
        self.component_cache = {}  # paper_id -> components

    def evaluate(
        self,
        pred_code: str,
        paper_text: str,
        paper_id: str,
        ref_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        评估代码的忠实度。

        Args:
            pred_code: 生成的代码
            paper_text: 论文文本
            paper_id: 论文唯一标识符
            ref_code: 参考代码（可选，用于 AAR 和 CodeBERTScore）

        Returns:
            {
                "pcr": float,              # 论文组件覆盖率 (0-1)
                "aar": float,              # API 对齐率 (0-1)
                "codebertscore": float,    # CodeBERTScore (0-1)
                "component_hits": list,    # 命中的组件
                "missing_components": list,# 缺失的组件
            }
        """
        # 1. PCR
        pcr, component_hits, missing = self.compute_pcr(pred_code, paper_text, paper_id)

        # 2. AAR (需要参考代码)
        aar = self.compute_aar(pred_code, ref_code) if ref_code else 0.0

        # 3. CodeBERTScore (需要参考代码)
        cbs = self.compute_codebertscore(pred_code, ref_code) if ref_code else 0.0

        return {
            "pcr": pcr,
            "aar": aar,
            "codebertscore": cbs,
            "component_hits": component_hits,
            "missing_components": missing,
        }

    def compute_pcr(
        self,
        pred_code: str,
        paper_text: str,
        paper_id: str,
    ) -> tuple[float, List[str], List[str]]:
        """
        计算论文组件覆盖率。

        Returns:
            (pcr_score, component_hits, missing_components)
        """
        # 获取组件清单（使用缓存）
        if paper_id not in self.component_cache:
            self.component_cache[paper_id] = self.paper_extractor.extract(
                paper_text, paper_id
            )

        components = self.component_cache[paper_id]

        hits = []
        missing = []
        total = 0

        # 检查 required_modules
        for module in components.get("required_modules", []):
            total += 1
            if module in pred_code:
                hits.append(f"module:{module}")
            else:
                missing.append(f"module:{module}")

        # 检查 required_patterns
        for pattern in components.get("required_patterns", []):
            total += 1
            if self._match_pattern(pred_code, pattern):
                hits.append(f"pattern:{pattern}")
            else:
                missing.append(f"pattern:{pattern}")

        # 检查 required_functions
        for func in components.get("required_functions", []):
            total += 1
            if self._match_function(pred_code, func):
                hits.append(f"function:{func}")
            else:
                missing.append(f"function:{func}")

        # 检查 expected_dims
        for dim in components.get("expected_dims", []):
            total += 1
            if dim in pred_code:
                hits.append(f"dim:{dim}")
            else:
                missing.append(f"dim:{dim}")

        pcr = len(hits) / total if total > 0 else 0.0

        return pcr, hits, missing

    def compute_aar(self, pred_code: str, ref_code: Optional[str]) -> float:
        """
        计算 API 调用对齐率。

        AAR = |pred_apis ∩ ref_apis| / |ref_apis|
        """
        if not ref_code:
            return 0.0

        pred_apis = self._extract_apis(pred_code)
        ref_apis = self._extract_apis(ref_code)

        if not ref_apis:
            return 0.0

        intersection = pred_apis & ref_apis
        return len(intersection) / len(ref_apis)

    def compute_codebertscore(
        self,
        pred_code: str,
        ref_code: Optional[str],
    ) -> float:
        """
        计算 CodeBERTScore。

        使用 code-bert-score 库。
        """
        if not ref_code:
            return 0.0

        try:
            from code_bert_score import score

            # 计算 F1
            _, _, f1, _ = score([pred_code], [ref_code], lang="python")
            return float(f1.item())

        except ImportError:
            print("[WARNING] code-bert-score not installed. Skipping CodeBERTScore.")
            return 0.0
        except Exception as e:
            print(f"[ERROR] CodeBERTScore computation failed: {e}")
            return 0.0

    def _extract_apis(self, code: str) -> Set[str]:
        """
        从代码中提取 PyTorch API 调用。

        Returns:
            API 名称集合，如 {"torch.nn.Linear", "F.softmax", ...}
        """
        apis = set()

        # 匹配模式
        patterns = [
            r"torch\.([a-zA-Z_][a-zA-Z0-9_\.]*)",  # torch.*
            r"nn\.([a-zA-Z_][a-zA-Z0-9_]*)",  # nn.*
            r"F\.([a-zA-Z_][a-zA-Z0-9_]*)",  # F.*
            r"nn\.functional\.([a-zA-Z_][a-zA-Z0-9_]*)",  # nn.functional.*
        ]

        for pattern in patterns:
            matches = re.findall(pattern, code)
            apis.update(matches)

        return apis

    def _match_pattern(self, code: str, pattern_name: str) -> bool:
        """
        匹配代码结构模式。

        Args:
            code: 代码字符串
            pattern_name: 模式名称

        Returns:
            True 如果模式匹配
        """
        # 预定义模式正则表达式
        patterns = {
            "residual_connection": r"\w+\s*[\+\-]\s*\w+\([^)]*\)",  # x + self.layer(x)
            "positional_encoding": r"pos(?:itional)?[_\s]?enc(?:od)?|PositionalEnc",
            "softmax_attention": r"F\.softmax.*attn|softmax.*dim\s*=\s*-1",
            "layer_normalization": r"LayerNorm|layer_norm",
            "dropout": r"Dropout|dropout",
            "multi_head_attention": r"MultiheadAttention|multi_head",
            "feedforward": r"FeedForward|feed_forward|FFN",
            "transformer": r"Transformer|transformer",
            "embedding": r"Embedding|embedding",
            "linear": r"nn\.Linear|Linear\(",
        }

        if pattern_name in patterns:
            return bool(re.search(patterns[pattern_name], code, re.IGNORECASE))

        # 未知模式，简单字符串匹配
        return pattern_name.lower() in code.lower()

    def _match_function(self, code: str, func_name: str) -> bool:
        """
        检查函数是否定义。

        Args:
            code: 代码字符串
            func_name: 函数名

        Returns:
            True 如果函数已定义
        """
        # 匹配 def func_name(...):
        pattern = rf"\bdef\s+{re.escape(func_name)}\s*\("
        return bool(re.search(pattern, code))
