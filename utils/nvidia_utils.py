"""
nvidia_utils.py - 显存和耗时监控工具

基础组件（向后兼容）:
    from utils.nvidia_utils import Timer, MemoryMonitor, print_memory_status
    from utils.nvidia_utils import get_memory_snapshot

    # 计时
    with Timer("训练"):
        train()

    # 后台显存采样
    with MemoryMonitor(interval=5):
        task()

    # 一次性快照
    print_memory_status("Before eval")

统一监控器（推荐）:
    from utils.nvidia_utils import TaskMonitor, monitor, print_memory_status

    # 方式 1：上下文管理器
    with TaskMonitor("SFT Training", log_dir="logs/monitoring") as mon:
        train()
        mon.note("epoch 1 done")

    # 方式 2：方法式
    mon = TaskMonitor("Evaluation", log_dir="logs/monitoring")
    mon.start()
    evaluate()
    mon.stop()

    # 方式 3：装饰器
    @monitor("Distillation", log_dir="logs/monitoring")
    def distill_code(code: str) -> str:
        ...

输出结构:
    logs/monitoring/
    └── SFT_Training_20260630_143000/
        ├── records.jsonl    # 每行: {elapsed_s, gpu_allocated_mb, gpu_reserved_mb, ...}
        └── summary.json     # {task, elapsed_s, memory: {peak, avg, ...}}
"""

from __future__ import annotations

import json
import time
import subprocess
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ═══════════════════════════════════════════════
#  底层：GPU 显存快照
# ═══════════════════════════════════════════════

def get_memory_snapshot() -> Optional[Dict[str, Any]]:
    """获取 GPU 0 的显存快照。优先 nvidia-smi，回退 PyTorch API。

    Returns:
        dict: {allocated_mb, reserved_mb, total_mb, free_mb, timestamp, source}
    """
    # 方法1: nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4 and parts[0] == "0":
                    return {
                        "allocated_mb": None,
                        "reserved_mb": float(parts[1]),
                        "total_mb": float(parts[2]),
                        "free_mb": float(parts[3]),
                        "timestamp": time.time(),
                        "source": "nvidia-smi",
                    }
    except Exception:
        pass

    # 方法2: PyTorch API
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(0) / 1024 / 1024
            reserved = torch.cuda.memory_reserved(0) / 1024 / 1024
            total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            return {
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "total_mb": total,
                "free_mb": total - reserved,
                "timestamp": time.time(),
                "source": "torch",
            }
    except Exception:
        pass

    return None


# ═══════════════════════════════════════════════
#  基础组件：Timer / MemoryMonitor（保留，向后兼容）
# ═══════════════════════════════════════════════

class Timer:
    """简单计时器上下文管理器"""

    def __init__(self, name: str = "Task", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            print(f"[Timer] ⏱️  {self.name} started at {datetime.now().strftime('%H:%M:%S')}")
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"[Timer] ✅ {self.name} completed in {self.elapsed:.2f}s "
                  f"({self.elapsed / 60:.2f}min)")

    def get_elapsed(self) -> float:
        if self.elapsed is not None:
            return self.elapsed
        if self.start_time is not None:
            return time.time() - self.start_time
        return 0.0


class MemoryMonitor:
    """显存后台采样器（上下文管理器）"""

    def __init__(self, log_file: Optional[str] = None, interval: float = 5.0,
                 verbose: bool = True):
        self.log_file = log_file
        self.interval = interval
        self.verbose = verbose
        self.samples: List[Dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        if self.verbose:
            print(f"[MemoryMonitor] 🔍 Started (interval={self.interval}s)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.samples and self.verbose:
            stats = self.get_stats()
            self._print_stats(stats)
            if self.log_file:
                self._save_log(stats)

    def get_stats(self) -> Dict[str, Any]:
        if not self.samples:
            return {}
        reserved = [s["reserved_mb"] for s in self.samples]
        allocated = [s["allocated_mb"] for s in self.samples
                     if s["allocated_mb"] is not None]
        return {
            "max_reserved_mb": max(reserved),
            "avg_reserved_mb": sum(reserved) / len(reserved),
            "max_allocated_mb": max(allocated) if allocated else None,
            "avg_allocated_mb": sum(allocated) / len(allocated) if allocated else None,
            "sample_count": len(self.samples),
        }

    def _monitor_loop(self):
        while self._running:
            snapshot = get_memory_snapshot()
            if snapshot:
                self.samples.append(snapshot)
            time.sleep(self.interval)

    def _print_stats(self, stats: Dict[str, Any]):
        print("[MemoryMonitor] 📊 Memory Statistics:")
        print(f"  Peak Reserved:  {stats['max_reserved_mb']:.1f} MB "
              f"({stats['max_reserved_mb'] / 1024:.2f} GB)")
        if stats["max_allocated_mb"] is not None:
            print(f"  Peak Allocated: {stats['max_allocated_mb']:.1f} MB "
                  f"({stats['max_allocated_mb'] / 1024:.2f} GB)")
        print(f"  Avg Reserved:   {stats['avg_reserved_mb']:.1f} MB")
        print(f"  Samples:        {stats['sample_count']}")

    def _save_log(self, stats: Dict[str, Any]):
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "statistics": stats,
                }, f, indent=2, ensure_ascii=False)
            print(f"[MemoryMonitor] 📝 Log saved to {self.log_file}")
        except Exception as exc:
            print(f"[MemoryMonitor] ⚠️ Failed to save log: {exc}")


def print_memory_status(prefix: str = ""):
    """打印当前 GPU 0 显存状态（一次性）"""
    snapshot = get_memory_snapshot()
    if snapshot:
        label = f"[{prefix}] " if prefix else ""
        used_mb = snapshot["reserved_mb"]
        total_mb = snapshot["total_mb"]
        print(f"{label}GPU Memory: {used_mb / 1024:.2f}GB / {total_mb / 1024:.2f}GB "
              f"({used_mb / total_mb * 100:.1f}% used)")
    else:
        print("[Memory] ⚠️ Unable to get GPU memory info")


# ═══════════════════════════════════════════════
#  TaskMonitor — 统一监控器
# ═══════════════════════════════════════════════

class TaskMonitor:
    """训练/评测任务监控器，同时记录耗时和显存。

    输出目录: {log_dir}/{name}_{timestamp}/
      records.jsonl  — 每条采样记录
      summary.json   — 汇总统计
    """

    def __init__(
        self,
        name: str,
        log_dir: str = "logs/monitoring",
        interval: float = 5.0,
        verbose: bool = True,
    ):
        self.name = name
        self.verbose = verbose
        self.interval = interval
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(log_dir) / f"{name}_{self.timestamp}"
        self.records_path = self.run_dir / "records.jsonl"
        self.summary_path = self.run_dir / "summary.json"

        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._records: List[Dict[str, Any]] = []

    # ── 上下文管理器 ──

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ── 方法式 API ──

    def start(self):
        """开始监控"""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()
        self._running = True
        self._records = []

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

        if self.verbose:
            print(f"[TaskMonitor] 🔍 '{self.name}' started "
                  f"({datetime.now().strftime('%H:%M:%S')})")
            print(f"[TaskMonitor] 📁 {self.run_dir}")

    def stop(self):
        """停止监控并写入文件"""
        self._end_time = time.time()
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

        elapsed = self._end_time - (self._start_time or self._end_time)

        self._write_records()
        summary = self._build_summary(elapsed)
        self._write_summary(summary)

        if self.verbose:
            self._print_summary(summary)

    def note(self, text: str):
        """在采样流中插入文本标记（用于标记阶段边界）。"""
        self._records.append({
            "elapsed_s": time.time() - (self._start_time or time.time()),
            "note": text,
        })

    # ── 内部 ──

    def _sample_loop(self):
        while self._running:
            snapshot = get_memory_snapshot()
            if snapshot:
                rec = {
                    "elapsed_s": time.time() - (self._start_time or time.time()),
                    "gpu_total_mb": snapshot["total_mb"],
                    "gpu_reserved_mb": snapshot["reserved_mb"],
                    "gpu_allocated_mb": snapshot["allocated_mb"],
                    "gpu_free_mb": snapshot["free_mb"],
                    "source": snapshot["source"],
                }
                self._records.append(rec)
            time.sleep(self.interval)

    def _build_summary(self, elapsed_s: float) -> Dict[str, Any]:
        mem_samples = [r for r in self._records if "gpu_reserved_mb" in r]
        reserved = [r["gpu_reserved_mb"] for r in mem_samples]
        allocated = [r["gpu_allocated_mb"] for r in mem_samples
                     if r["gpu_allocated_mb"] is not None]

        return {
            "task": self.name,
            "started_at": datetime.fromtimestamp(self._start_time).isoformat()
                          if self._start_time else None,
            "elapsed_s": round(elapsed_s, 1),
            "elapsed_min": round(elapsed_s / 60, 2),
            "sample_count": len(mem_samples),
            "interval_s": self.interval,
            "memory": {
                "peak_reserved_mb": round(max(reserved), 1) if reserved else None,
                "avg_reserved_mb": round(sum(reserved) / len(reserved), 1) if reserved else None,
                "peak_allocated_mb": round(max(allocated), 1) if allocated else None,
                "avg_allocated_mb": round(sum(allocated) / len(allocated), 1) if allocated else None,
            },
        }

    def _write_records(self):
        with open(self.records_path, "w", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _write_summary(self, summary: Dict[str, Any]):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _print_summary(self, summary: Dict[str, Any]):
        mem = summary["memory"]
        print(f"[TaskMonitor] ✅ '{self.name}' done "
              f"({summary['elapsed_min']} min, {summary['sample_count']} samples)")
        if mem["peak_reserved_mb"]:
            print(f"  Peak Reserved:  {mem['peak_reserved_mb']:.0f} MB "
                  f"({mem['peak_reserved_mb'] / 1024:.2f} GB)")
        if mem["peak_allocated_mb"]:
            print(f"  Peak Allocated: {mem['peak_allocated_mb']:.0f} MB "
                  f"({mem['peak_allocated_mb'] / 1024:.2f} GB)")
        print(f"  Avg Reserved:   {mem['avg_reserved_mb']:.0f} MB")
        print(f"  Summary:        {self.summary_path}")


# ═══════════════════════════════════════════════
#  装饰器
# ═══════════════════════════════════════════════

def monitor(
    name: Optional[str] = None,
    log_dir: str = "logs/monitoring",
    interval: float = 5.0,
    verbose: bool = True,
) -> Callable:
    """装饰器：自动对函数进行耗时+显存监控。

    使用:
        @monitor("distill")
        def distill_code(code: str) -> str:
            ...

        @monitor(log_dir="logs/ablation")  # name 默认为函数名
        def train_model(config):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            task_name = name or func.__name__
            with TaskMonitor(task_name, log_dir=log_dir, interval=interval,
                             verbose=verbose):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ═══════════════════════════════════════════════
#  CLI: 包裹外部命令
# ═══════════════════════════════════════════════

def run_monitored_command(
    name: str,
    cmd: List[str],
    log_dir: str = "logs/monitoring",
    interval: float = 30.0,
    log_file: Optional[str] = None,
) -> int:
    """在 TaskMonitor 中运行一个外部命令。

    Args:
        name: 任务名
        cmd: 要运行的命令和参数列表
        log_dir: 监控日志目录
        interval: 显存采样间隔（秒）
        log_file: 命令输出日志文件（可选）

    Returns:
        命令的退出码
    """
    with TaskMonitor(name, log_dir=log_dir, interval=interval) as mon:
        stdout_dest = None
        if log_file:
            stdout_dest = open(log_file, "a")

        proc = subprocess.Popen(
            cmd,
            stdout=stdout_dest or subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if not stdout_dest and proc.stdout:
            for line in proc.stdout:
                print(line, end="")
        proc.wait()
        if stdout_dest:
            stdout_dest.close()

        mon.note(f"exited with code {proc.returncode}")
        return proc.returncode


# ═══════════════════════════════════════════════
#  Demo / CLI
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--":
        # 包裹模式: python -m utils.nvidia_utils -- name --cmd arg1 arg2 ...
        name = sys.argv[2] if len(sys.argv) > 2 else "Task"
        cmd = sys.argv[3:]
        if not cmd:
            print("Usage: python -m utils.nvidia_utils -- <task_name> <command...>")
            sys.exit(1)
        exit_code = run_monitored_command(name, cmd)
        sys.exit(exit_code)

    else:
        # Demo 模式
        print("=" * 60)
        print("nvidia_utils.py — Monitoring Demo")
        print("=" * 60)

        print("\n--- get_memory_snapshot() ---")
        snap = get_memory_snapshot()
        if snap:
            print(f"  GPU: {snap['reserved_mb'] / 1024:.2f} GB / "
                  f"{snap['total_mb'] / 1024:.2f} GB  (source: {snap['source']})")
        else:
            print("  No GPU available")

        print("\n--- Timer ---")
        with Timer("Demo Task"):
            time.sleep(0.5)

        print("\n--- TaskMonitor (context manager) ---")
        with TaskMonitor("Demo_Monitor", interval=1) as mon:
            time.sleep(1)
            mon.note("halfway")
            time.sleep(1)
        print(f"  records: {mon.records_path}")
        print(f"  summary: {mon.summary_path}")

        print("\n--- @monitor decorator ---")
        @monitor("demo_decorated", interval=1)
        def demo_fn():
            time.sleep(0.5)
        demo_fn()

        print("\n✅ Demo complete")
