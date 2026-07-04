"""
evaluate.py - 主评测脚本

端到端评测流程：
1. 加载模型
2. 对测试集中的每篇论文生成代码
3. 评估可运行性和忠实度
4. 汇总统计并保存结果
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from utils.nvidia_utils import TaskMonitor

from utils.model_loader import ModelLoader
from utils.code_parser import extract_code
from evaluators.execution_eval import ExecutionEvaluator
from evaluators.fidelity_eval import FidelityEvaluator
from evaluators.paper_extractor import PaperComponentExtractor


class ModelEvaluator:
    """端到端模型评估器"""

    def __init__(
        self,
        model_path: str,
        use_api: bool = False,
        api_base: str = None,
        cache_dir: str = "outputs/eval",
        max_tokens: int = 2048,
    ):
        """
        初始化评估器。

        Args:
            model_path: 模型路径或 HuggingFace model_id
            use_api: 是否使用 API 模式
            api_base: API 地址
            cache_dir: 缓存和输出目录
        """
        print(f"[Evaluator] Initializing with model: {model_path}")

        self.max_tokens = max_tokens
        self.model = ModelLoader(model_path, use_api, api_base)
        self.execution_eval = ExecutionEvaluator()

        # PaperComponentExtractor 支持 OpenRouter（通过环境变量）
        self.paper_extractor = PaperComponentExtractor(
            cache_dir=os.path.join(cache_dir, "component_cache")
        )
        self.fidelity_eval = FidelityEvaluator(self.paper_extractor)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        output_dir: str,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        评估整个数据集。

        Args:
            dataset: 测试集列表
            output_dir: 输出目录
            resume: 是否从缓存恢复

        Returns:
            汇总统计结果
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / "results.jsonl"

        monitor_dir = output_path / "monitor"
        mon = TaskMonitor(
            name="Eval_paper2CoreCode",
            log_dir=str(monitor_dir),
            interval=30.0,
        )

        # 如果恢复模式，加载已有结果
        processed_ids = set()
        if resume and results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f:
                    result = json.loads(line)
                    processed_ids.add(result["paper_id"])
            print(f"[Evaluator] Resuming from {len(processed_ids)} processed samples.")

        mon.start()
        try:
            # 逐样本评估
            results = []
            for i, sample in enumerate(dataset):
                paper_id = sample.get("paper_id", f"sample_{i}")

                if resume and paper_id in processed_ids:
                    print(
                        f"[{i + 1}/{len(dataset)}] Skipping {paper_id} (already processed)"
                    )
                    continue

                print(f"\n[{i + 1}/{len(dataset)}] Evaluating {paper_id}")
                mon.note(f"start {paper_id}")

                result = self._evaluate_single(sample)
                results.append(result)

                # 实时保存
                with open(results_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        finally:
            mon.stop()

        # 加载所有结果（包括恢复的）
        all_results = []
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f:
                    all_results.append(json.loads(line))

        # 汇总统计
        summary = self._aggregate(all_results)

        # 保存汇总
        with open(output_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n[Evaluator] Evaluation complete. Results saved to {output_dir}")
        print(f"[Evaluator] Summary:\n{json.dumps(summary, indent=2)}")

        return summary

    def _evaluate_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个样本"""
        paper_id = sample.get("paper_id", "unknown")
        paper_text = sample.get("paper_text", "")
        ref_code = sample.get("ref_code", None)

        # 1. 模型推理
        print(f"  [1/3] Generating code...")
        try:
            model_output = self.model.generate(paper_text, max_tokens=self.max_tokens)
            pred_code = extract_code(model_output)
        except Exception as e:
            print(f"  [ERROR] Generation failed: {e}")
            pred_code = None
            model_output = ""

        if not pred_code:
            # 生成失败
            return {
                "paper_id": paper_id,
                "success": False,
                "error": "Code extraction failed",
                "model_output": model_output[:500],  # 截取前 500 字符
            }

        # 2. 执行评估
        print(f"  [2/3] Evaluating execution...")
        try:
            exec_result = self.execution_eval.evaluate(pred_code)
        except Exception as e:
            print(f"  [ERROR] Execution eval failed: {e}")
            exec_result = {}

        # 3. 忠实度评估
        print(f"  [3/3] Evaluating fidelity...")
        try:
            fid_result = self.fidelity_eval.evaluate(
                pred_code=pred_code,
                paper_text=paper_text,
                paper_id=paper_id,
                ref_code=ref_code,
            )
        except Exception as e:
            print(f"  [ERROR] Fidelity eval failed: {e}")
            fid_result = {}

        # 汇总结果
        return {
            "paper_id": paper_id,
            "success": True,
            "pred_code": pred_code,
            "execution": exec_result,
            "fidelity": fid_result,
            "paper_title": sample.get("paper_title", ""),
        }

    def _aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总统计"""
        n = len(results)
        if n == 0:
            return {}

        # 过滤成功的结果
        valid_results = [r for r in results if r.get("success", False)]
        n_valid = len(valid_results)

        if n_valid == 0:
            return {
                "total_samples": n,
                "valid_samples": 0,
                "error": "No valid results",
            }

        # 可运行性指标
        xpr = (
            sum(1 for r in valid_results if r["execution"].get("xpr", False)) / n_valid
        )
        sar = (
            sum(1 for r in valid_results if r["execution"].get("sar", False)) / n_valid
        )
        syr = (
            sum(1 for r in valid_results if r["execution"].get("syr", False)) / n_valid
        )

        # 忠实度指标
        pcr_avg = sum(r["fidelity"].get("pcr", 0.0) for r in valid_results) / n_valid
        aar_avg = sum(r["fidelity"].get("aar", 0.0) for r in valid_results) / n_valid
        cbs_avg = (
            sum(r["fidelity"].get("codebertscore", 0.0) for r in valid_results)
            / n_valid
        )

        # 综合评分（按日志中的公式）
        execution_score = 0.4 * xpr + 0.4 * sar + 0.2 * syr
        fidelity_score = 0.5 * pcr_avg + 0.3 * cbs_avg + 0.2 * aar_avg
        overall_score = 0.5 * execution_score + 0.5 * fidelity_score

        return {
            "total_samples": n,
            "valid_samples": n_valid,
            "xpr": xpr,
            "sar": sar,
            "syr": syr,
            "pcr_avg": pcr_avg,
            "aar_avg": aar_avg,
            "codebertscore_avg": cbs_avg,
            "execution_score": execution_score,
            "fidelity_score": fidelity_score,
            "overall_score": overall_score,
        }


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """加载测试集"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on paper-to-code task")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model path or HuggingFace model_id",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to test dataset JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--use_api",
        action="store_true",
        help="Use API mode (vLLM/OpenAI-compatible)",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="API base URL",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from cached results",
    )

    args = parser.parse_args()

    # 加载数据集
    print(f"[Main] Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"[Main] Loaded {len(dataset)} samples")

    # 初始化评估器
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        use_api=args.use_api,
        api_base=args.api_base,
        cache_dir=args.output_dir,
        max_tokens=args.max_tokens,
    )

    # 运行评估
    summary = evaluator.evaluate_dataset(
        dataset=dataset,
        output_dir=args.output_dir,
        resume=args.resume,
    )

    print(f"\n{'=' * 60}")
    print(f"Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"Total Samples: {summary.get('total_samples', 0)}")
    print(f"Valid Samples: {summary.get('valid_samples', 0)}")
    print(f"\nExecution Metrics:")
    print(f"  XPR: {summary.get('xpr', 0.0):.2%}")
    print(f"  SAR: {summary.get('sar', 0.0):.2%}")
    print(f"  SYR: {summary.get('syr', 0.0):.2%}")
    print(f"\nFidelity Metrics:")
    print(f"  PCR: {summary.get('pcr_avg', 0.0):.2%}")
    print(f"  AAR: {summary.get('aar_avg', 0.0):.2%}")
    print(f"  CodeBERTScore: {summary.get('codebertscore_avg', 0.0):.3f}")
    print(f"\nOverall Score: {summary.get('overall_score', 0.0):.2%}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
