#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集构建脚本 - LLM蒸馏版本（与训练集一致）

根据评测需求:
1. paper_text: 论文核心文本 (输入模型)
2. ref_code: GitHub 参考代码 (评测参考,用于 AAR 和 CodeBERTScore)

【重要变更】ref_code 现在使用与训练集相同的 LLM 蒸馏方法,确保评测公平性。
"""

import json
import re
import subprocess
import tempfile
import os
import sys
import threading
from pathlib import Path
from typing import Optional
import time

# ========== 添加 Paper2Code 路径以导入蒸馏模块 ==========
REPO_ROOT = Path(__file__).parent.parent
paper2code_path = REPO_ROOT / "Paper2Code" / "data" / "paper2code"
if str(paper2code_path) not in sys.path:
    sys.path.insert(0, str(paper2code_path))

# 导入蒸馏类（与训练集完全一致）
try:
    from distill_code_repository import GitExtractor, DistillationEngine

    HAS_DISTILLER = True
    print("✓ 已加载 LLM 蒸馏引擎（与训练集一致）")
except ImportError as e:
    HAS_DISTILLER = False
    print(f"⚠️ 无法加载蒸馏引擎: {e}")
    print("将无法使用蒸馏功能，ref_code 将为空")

# 修复 Windows 控制台编码问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        # Python < 3.7
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

try:
    import fitz  # PyMuPDF

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("警告: PyMuPDF 未安装,将使用 PyPDF2")
    try:
        import PyPDF2
    except ImportError:
        print("错误: 需要安装 PyMuPDF 或 PyPDF2")
        print("运行: pip install PyMuPDF")
        exit(1)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class TestSetBuilder:
    """测试集构建器（使用LLM蒸馏）"""

    def __init__(self, config_path: str, output_path: str):
        self.config_path = Path(config_path)
        self.output_path = Path(output_path)
        self.pdf_dirs = {
            "iclr2025": Path("iclr2025_pdfs"),
            "icml2025": Path("icml2025_pdfs"),
            "nips2025": Path("nips2025_pdfs"),
        }
        self.output_txt_dir = Path("test_cleaned_output")
        self.output_txt_dir.mkdir(parents=True, exist_ok=True)

        # ========== 初始化蒸馏引擎（与训练集一致）==========
        if HAS_DISTILLER:
            prompt_path = (
                REPO_ROOT
                / "Paper2Code"
                / "data"
                / "paper2code"
                / "distill_code_repository_system_prompt.md"
            )

            try:
                self.distiller = DistillationEngine(
                    prompt_file_path=str(prompt_path),
                    model_name="gpt-4o-mini",  # 与训练集一致
                )
                self.distill_success = 0
                self.distill_failed = 0
                print("✓ 蒸馏引擎初始化成功 (gpt-4o-mini)")
            except Exception as e:
                print(f"⚠️ 蒸馏引擎初始化失败: {e}")
                self.distiller = None
        else:
            self.distiller = None

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """从 PDF 提取核心文本"""
        print(f"  提取文本: {pdf_path.name}")

        if HAS_PYMUPDF:
            return self._extract_with_pymupdf(pdf_path)
        else:
            return self._extract_with_pypdf2(pdf_path)

    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """使用 PyMuPDF 提取 (推荐)"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""

            # 提取前 20 页
            for page_num in range(min(20, len(doc))):
                page = doc[page_num]
                full_text += page.get_text()

            doc.close()

            return self._extract_core_sections(full_text)

        except Exception as e:
            print(f"    ⚠️ PyMuPDF 提取失败: {e}")
            return ""

    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """使用 PyPDF2 提取 (备选)"""
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ""

                for i in range(min(20, len(reader.pages))):
                    page = reader.pages[i]
                    full_text += page.extract_text()

                return self._extract_core_sections(full_text)

        except Exception as e:
            print(f"    ⚠️ PyPDF2 提取失败: {e}")
            return ""

    def _extract_core_sections(self, text: str) -> str:
        """从文本中提取 Abstract 和核心章节"""
        # 提取 Abstract
        abstract_pattern = r"(?i)abstract\s*(.*?)(?=\n\s*\n|\n\s*introduction|\n\s*1\.)"
        abstract_match = re.search(abstract_pattern, text, re.DOTALL)

        abstract_text = ""
        if abstract_match:
            abstract_text = f"[Abstract]\n{abstract_match.group(1).strip()}\n\n"

        # 提取 Method 相关章节
        method_keywords = [
            "method",
            "methodology",
            "approach",
            "model",
            "architecture",
            "algorithm",
            "implementation",
        ]

        method_sections = []
        for keyword in method_keywords:
            pattern = rf"(?i)(\d+\.?\s*{keyword}.*?)(?=\n\s*\d+\.|\n\s*references|$)"
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches[:2]:  # 最多 2 个相关章节
                section_text = match.strip()
                if len(section_text) > 500:  # 只保留有实质内容的章节
                    if len(section_text) > 3000:
                        section_text = section_text[:3000] + "..."
                    method_sections.append(f"[{keyword.title()}]\n{section_text}\n\n")

        result = abstract_text + "".join(method_sections[:3])  # 最多 3 个章节

        # 如果提取失败,返回前 6000 字符
        if len(result) < 500:
            result = text[:6000]

        return result.strip()

    def extract_reference_code(self, github_url: str, repo_name: str) -> Optional[str]:
        """从 GitHub 仓库提取核心模型代码（使用 LLM 蒸馏，与训练集一致）

        流程：
          1. GitExtractor 克隆仓库并合并所有 .py 文件
          2. tiktoken 计数，超限则跳过
          3. DistillationEngine 调用 gpt-4o-mini 蒸馏核心代码
          4. 返回 <file> 标签格式的蒸馏代码；失败时返回 ""

        Args:
            github_url: GitHub 仓库 URL
            repo_name:  仓库名称（仅用于日志）

        Returns:
            蒸馏后的核心代码（<file> 标签格式），失败时返回空字符串
        """
        print(f"  提取代码: {github_url}", flush=True)

        # 黑名单：已知有问题的仓库（克隆时卡住或失败）
        problem_repos = [
            "epflneuroailab/topolm",  # 克隆时卡住
            "divelab/AIRS",  # 可能会卡住
        ]
        for problem_repo in problem_repos:
            if problem_repo in github_url:
                print(f"    ⚠️ 跳过已知有问题的仓库: {problem_repo}", flush=True)
                return ""

        # 蒸馏引擎不可用时直接返回空
        if not self.distiller:
            print(f"    ⚠️ 蒸馏引擎不可用，跳过", flush=True)
            return ""

        try:
            # ── Step 1: 克隆仓库并提取所有 Python 代码 ──────────────────
            print(f"    [1/3] 克隆仓库并提取Python文件...", flush=True)
            raw_code = GitExtractor.extract_python_code(github_url)

            if not raw_code or len(raw_code) < 100:
                print(f"    ⚠️ 未找到有效Python代码", flush=True)
                self.distill_failed += 1
                return ""

            # ── Step 2: LLM 蒸馏 ────────────────────────────────────────
            print(f"    [2/2] LLM蒸馏中 (gpt-4o-mini)...", flush=True)
            distilled_code = self.distiller.distill_code(raw_code)

            self.distill_success += 1

            print(
                f"    ✓ 蒸馏完成 (输出: {len(distilled_code):,} 字符)",
                flush=True,
            )
            return distilled_code

        except Exception as e:
            print(f"    ⚠️ 蒸馏失败: {type(e).__name__}: {str(e)[:100]}", flush=True)
            self.distill_failed += 1
            return ""

    def find_pdf_for_paper(
        self, paper_title: str, repo_name: str, conference: str
    ) -> Optional[Path]:
        """查找论文 PDF 文件

        Args:
            paper_title: 论文标题（优先使用）
            repo_name: 仓库名称（备用）
            conference: 会议名称
        """
        pdf_dir = self.pdf_dirs.get(conference)
        if not pdf_dir or not pdf_dir.exists():
            return None

        # 策略1: 使用论文标题进行模糊匹配（主要策略）
        paper_clean = re.sub(r"[^\w\s-]", "", paper_title.lower())
        paper_words = set(paper_clean.split())

        best_match = None
        best_score = 0

        for pdf_file in pdf_dir.glob("*.pdf"):
            pdf_clean = re.sub(r"[^\w\s-]", "", pdf_file.stem.lower())
            pdf_words = set(pdf_clean.split())

            # 计算匹配分数（共同单词数量）
            common_words = paper_words & pdf_words
            score = len(common_words)

            # 如果有超过3个共同单词，认为匹配度很高
            if score > best_score and score >= 3:
                best_score = score
                best_match = pdf_file

            # 完全包含匹配（高优先级）
            if paper_clean in pdf_clean or pdf_clean in paper_clean:
                return pdf_file

        if best_match:
            return best_match

        # 策略2: 使用 repo_name 作为备用（适用于简短的项目名）
        patterns = [
            f"{repo_name}.pdf",  # 精确匹配
            f"*{repo_name}*.pdf",  # 包含匹配
        ]

        for pattern in patterns:
            matches = list(pdf_dir.glob(pattern))
            if matches:
                return matches[0]

        # 策略3: repo_name 模糊匹配
        repo_clean = re.sub(r"[^\w\s-]", "", repo_name.lower())
        for pdf_file in pdf_dir.glob("*.pdf"):
            pdf_clean = re.sub(r"[^\w\s-]", "", pdf_file.stem.lower())
            if repo_clean in pdf_clean or pdf_clean in repo_clean:
                return pdf_file

        return None

    def build(self):
        """构建完整测试集"""
        print("=" * 70)
        print("开始构建测试集...")
        print("=" * 70)

        # 读取配置
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 尝试加载已有的测试集（支持断点续传）
        test_set = []
        processed_titles = set()
        if self.output_path.exists():
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    test_set = json.load(f)
                    processed_titles = {item["paper_title"] for item in test_set}
                print(
                    f"\n已加载 {len(test_set)} 个已处理的样本，继续处理...\n",
                    flush=True,
                )
            except:
                print("\n无法加载已有文件，重新开始...\n", flush=True)

        stats = {"total": 0, "pdf_found": 0, "text_extracted": 0, "code_extracted": 0}

        for conference, papers in config.items():
            print(f"\n处理 {conference.upper()} ({len(papers)} 篇论文)")
            print("-" * 70)

            for i, paper_info in enumerate(papers, 1):
                paper_title = paper_info["paper"]
                repo_name = paper_info["repo_name"]
                github_url = paper_info["repo_url"]

                # 跳过已处理的论文
                if paper_title in processed_titles:
                    print(f"\n[{i}/{len(papers)}] {paper_title} - 已处理，跳过")
                    continue

                print(f"\n[{i}/{len(papers)}] {paper_title}", flush=True)
                stats["total"] += 1

                # 生成 paper_id
                paper_id = re.sub(r"[^\w\s-]", "", repo_name.lower())
                paper_id = re.sub(r"[\s]+", "_", paper_id)

                # 查找 PDF（优先使用论文标题）
                pdf_path = self.find_pdf_for_paper(paper_title, repo_name, conference)
                if not pdf_path:
                    print(f"  ⚠️ 未找到 PDF,跳过")
                    continue

                stats["pdf_found"] += 1

                # 提取论文文本
                paper_text = self.extract_text_from_pdf(pdf_path)
                if not paper_text or len(paper_text) < 500:
                    print(f"  ⚠️ 文本提取失败或太短,跳过")
                    continue

                stats["text_extracted"] += 1

                # 保存文本到文件 (供后续检查)
                txt_path = self.output_txt_dir / f"{repo_name}.txt"
                txt_path.write_text(paper_text, encoding="utf-8")

                # 提取参考代码
                ref_code = self.extract_reference_code(github_url, repo_name)
                if ref_code:
                    stats["code_extracted"] += 1
                else:
                    ref_code = ""  # 空字符串,评测时自动跳过 AAR/CBS

                # 组装样本
                sample = {
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "paper_text": paper_text,
                    "ref_code": ref_code,
                    "github_url": github_url,
                }

                test_set.append(sample)
                print(
                    f"  ✓ 成功添加 (文本: {len(paper_text)} 字符, 代码: {len(ref_code)} 字符)"
                )

                # 每 5 个样本保存一次 (防止中断丢失)
                if len(test_set) % 5 == 0:
                    self._save(test_set)

                # 避免 GitHub rate limit
                time.sleep(1)

        # 最终保存
        self._save(test_set)

        # 打印统计
        print("\n" + "=" * 70)
        print(f"构建完成! 共 {len(test_set)} 个样本")
        print("=" * 70)
        print(f"\n统计:")
        print(f"  总论文数: {stats['total']}")
        print(
            f"  找到 PDF: {stats['pdf_found']} ({stats['pdf_found'] / stats['total'] * 100:.1f}%)"
        )
        print(
            f"  提取文本: {stats['text_extracted']} ({stats['text_extracted'] / stats['total'] * 100:.1f}%)"
        )
        print(
            f"  提取代码: {stats['code_extracted']} ({stats['code_extracted'] / stats['total'] * 100:.1f}%)"
        )

        # LLM 蒸馏统计
        if HAS_DISTILLER and self.distiller:
            print(f"\n💰 LLM蒸馏统计:")
            print(f"  成功蒸馏: {self.distill_success}")
            print(f"  失败/跳过: {self.distill_failed} (ref_code留空)")

        print(f"\n文本文件: {self.output_txt_dir}")
        print(f"测试集文件: {self.output_path}")

    def _save(self, test_set):
        """保存测试集"""
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(test_set, f, indent=2, ensure_ascii=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="构建测试集 (正确版本)")
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "data" / "dataset_2025.json"),
        help="Input config file (paper titles + GitHub URLs)",
    )
    parser.add_argument("--output", default="test_dataset.json", help="输出测试集文件")

    args = parser.parse_args()

    builder = TestSetBuilder(args.config, args.output)
    builder.build()


if __name__ == "__main__":
    main()
