"""Manage SVHunter pipeline subprocess execution with stdout capture."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class PipelineTask:
    task_id: str
    mode: str  # "generate" or "call"
    params: dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    logs: list[str] = field(default_factory=list)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)


# In-memory task store (single-process; sufficient for local use)
_tasks: dict[str, PipelineTask] = {}

# Path to the SVHunter repository root (one level up from version4/)
SVHUNTER_ROOT = str(
    Path(__file__).resolve().parent.parent.parent.parent / "SVHunter-1"
)


def get_task(task_id: str) -> PipelineTask | None:
    return _tasks.get(task_id)


def list_tasks() -> list[dict[str, Any]]:
    return [
        {
            "taskId": t.task_id,
            "mode": t.mode,
            "status": t.status.value,
            "startedAt": t.started_at,
            "finishedAt": t.finished_at,
            "logCount": len(t.logs),
        }
        for t in _tasks.values()
    ]


def build_generate_cmd(params: dict[str, Any]) -> list[str]:
    """Build the shell command for ``SVHunter.py generate``."""
    cmd = [
        "python",
        os.path.join(SVHUNTER_ROOT, "SVHunter.py"),
        "generate",
        params["bamPath"],
        params["outputDir"],
    ]
    if params.get("threads"):
        cmd.append(str(params["threads"]))
    if params.get("chroms"):
        cmd.append(str(params["chroms"]))
    return cmd


def build_call_cmd(params: dict[str, Any]) -> list[str]:
    """Build the shell command for ``SVHunter.py call``."""
    cmd = [
        "python",
        os.path.join(SVHUNTER_ROOT, "SVHunter.py"),
        "call",
        params["modelPath"],
        params["dataPath"],
        params["bamPath"],
        params["predictPath"],
        params["vcfOutputPath"],
    ]
    if params.get("threads"):
        cmd.append(str(params["threads"]))
    if params.get("chroms"):
        cmd.append(str(params["chroms"]))
    if params.get("gpus"):
        cmd.append(str(params["gpus"]))
    return cmd


async def start_task(mode: str, params: dict[str, Any]) -> str:
    """Create a task, launch the subprocess in the background, return task_id."""
    task_id = uuid.uuid4().hex[:12]
    task = PipelineTask(task_id=task_id, mode=mode, params=params)
    _tasks[task_id] = task

    if mode == "generate":
        cmd = build_generate_cmd(params)
    elif mode == "call":
        cmd = build_call_cmd(params)
    else:
        task.status = TaskStatus.FAILED
        task.error = f"Unknown mode: {mode}"
        return task_id

    task.logs.append(f"[CMD] {' '.join(cmd)}")
    task.status = TaskStatus.RUNNING
    task.started_at = time.time()

    # Launch in a background thread so we don't block the event loop
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_process, task, cmd)
    return task_id


def _run_process(task: PipelineTask, cmd: list[str]) -> None:
    """Execute the subprocess and capture stdout/stderr line by line."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=SVHUNTER_ROOT,
        )
        task.process = proc
        if proc.stdout:
            for line in proc.stdout:
                task.logs.append(line.rstrip("\n"))
        proc.wait()
        if proc.returncode == 0:
            task.status = TaskStatus.SUCCESS
        else:
            task.status = TaskStatus.FAILED
            task.error = f"Process exited with code {proc.returncode}"
    except Exception as exc:
        task.status = TaskStatus.FAILED
        task.error = str(exc)
    finally:
        task.finished_at = time.time()
