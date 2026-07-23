#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fcntl
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


DATA_DIR = Path(
    os.environ.get(
        "RETROFIT_PROJECT_DATA_DIR",
        "/data/energy-ai/retrofit-projects"
    )
)

PROJECT_DIR = DATA_DIR / "projects"
LEDGER_FILE = DATA_DIR / "project_ledger.jsonl"
LOCK_FILE = DATA_DIR / ".project_manager.lock"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_id_time() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)


def load_payload() -> Dict[str, Any]:
    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()

    if not raw.strip():
        raise ValueError("未收到输入JSON")

    payload = json.loads(raw)

    if not isinstance(payload, dict):
        raise ValueError("输入必须是JSON对象")

    return payload


def number(
    payload: Dict[str, Any],
    key: str,
    default: float = 0.0
) -> float:
    value = payload.get(key, default)

    if value in (None, ""):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"字段 {key} 必须是数字")


def project_path(project_id: str) -> Path:
    return PROJECT_DIR / f"{project_id}.json"


def load_project(project_id: str) -> Dict[str, Any]:
    path = project_path(project_id)

    if not path.exists():
        raise ValueError(f"未找到项目：{project_id}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("项目文件格式错误")

    return data


def save_project(project: Dict[str, Any]) -> Path:
    project_id = str(project["project_id"])
    path = project_path(project_id)
    temp_path = path.with_suffix(".json.tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(
            project,
            file,
            ensure_ascii=False,
            indent=2
        )
        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_path, path)
    return path


def append_ledger(record: Dict[str, Any]) -> None:
    with LEDGER_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )
        file.flush()
        os.fsync(file.fileno())


def parse_date(value: Any) -> datetime | None:
    text = str(value or "").strip()

    if not text:
        return None

    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None


def normalize_milestones(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    milestones: List[Dict[str, Any]] = []

    for index, item in enumerate(value, start=1):
        if isinstance(item, str):
            milestones.append(
                {
                    "milestone_id": f"M{index:02d}",
                    "name": item,
                    "status": "未开始",
                    "planned_date": ""
                }
            )
        elif isinstance(item, dict):
            milestones.append(
                {
                    "milestone_id": str(
                        item.get(
                            "milestone_id",
                            f"M{index:02d}"
                        )
                    ),
                    "name": str(
                        item.get("name", "")
                    ).strip(),
                    "status": str(
                        item.get("status", "未开始")
                    ),
                    "planned_date": str(
                        item.get("planned_date", "")
                    )
                }
            )

    return [
        item for item in milestones
        if item.get("name")
    ]


def calculate_project_state(
    project: Dict[str, Any]
) -> None:
    tasks = project.get("tasks", [])

    if not isinstance(tasks, list):
        tasks = []

    valid_tasks = [
        task for task in tasks
        if isinstance(task, dict)
    ]

    if valid_tasks:
        progress_values = [
            max(
                0.0,
                min(
                    100.0,
                    float(task.get("progress_percent", 0))
                )
            )
            for task in valid_tasks
        ]

        project["progress_percent"] = round(
            sum(progress_values) / len(progress_values),
            2
        )
    else:
        project["progress_percent"] = 0.0

    project["budget_used_cny"] = round(
        sum(
            float(task.get("actual_cost_cny", 0) or 0)
            for task in valid_tasks
        ),
        2
    )

    total_budget = float(
        project.get("budget_total_cny", 0) or 0
    )
    used_budget = float(
        project.get("budget_used_cny", 0) or 0
    )

    overdue_tasks = 0
    high_priority_overdue = 0
    today = datetime.now()

    for task in valid_tasks:
        due_date = parse_date(task.get("due_date"))
        status = str(task.get("status", ""))

        if (
            due_date is not None
            and due_date.date() < today.date()
            and status not in ("已完成", "已取消")
        ):
            overdue_tasks += 1
            task["overdue"] = True

            if task.get("priority") == "高":
                high_priority_overdue += 1
        else:
            task["overdue"] = False

    project["overdue_task_count"] = overdue_tasks

    if total_budget > 0 and used_budget > total_budget:
        project["risk_status"] = "高"
        project["risk_reason"] = "项目已超预算"
    elif high_priority_overdue > 0:
        project["risk_status"] = "高"
        project["risk_reason"] = "存在高优先级逾期任务"
    elif overdue_tasks > 0:
        project["risk_status"] = "中"
        project["risk_reason"] = "存在逾期任务"
    elif (
        total_budget > 0
        and used_budget >= total_budget * 0.9
    ):
        project["risk_status"] = "中"
        project["risk_reason"] = "预算使用率已达到90%"
    else:
        project["risk_status"] = "低"
        project["risk_reason"] = "暂无重大风险"

    if valid_tasks and all(
        task.get("status") == "已完成"
        for task in valid_tasks
    ):
        project["status"] = "已完成"
        project["progress_percent"] = 100.0
    elif any(
        task.get("status") in ("进行中", "已完成")
        or float(task.get("progress_percent", 0) or 0) > 0
        for task in valid_tasks
    ):
        project["status"] = "进行中"
    elif project.get("status") not in ("暂停", "已取消"):
        project["status"] = "已立项"

    project["updated_at"] = now_text()


def find_existing_project(
    project_name: str,
    start_date: str
) -> Dict[str, Any] | None:
    for path in PROJECT_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                project = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        if (
            str(project.get("project_name", "")).strip()
            == project_name
            and str(project.get("start_date", "")).strip()
            == start_date
            and project.get("status") != "已取消"
        ):
            return project

    return None


def create_project(payload: Dict[str, Any]) -> Dict[str, Any]:
    project_name = str(
        payload.get("project_name", "")
    ).strip()

    if not project_name:
        raise ValueError("缺少字段 project_name")

    start_date = str(
        payload.get("start_date", "")
    ).strip()

    existing = find_existing_project(
        project_name,
        start_date
    )

    if existing:
        return {
            "result": "ok",
            "employee": "节能改造项目管理 Subagent",
            "message": "检测到同名同起始日期项目，已返回已有项目。",
            "duplicate_detected": True,
            "visible_result": {
                "项目编号": existing["project_id"],
                "项目名称": existing["project_name"],
                "项目状态": existing["status"],
                "项目进度": existing["progress_percent"],
                "风险等级": existing["risk_status"]
            },
            "business_state_changes": [
                {
                    "business_object": "改造项目",
                    "before": f"已存在：{existing['project_id']}",
                    "after": "无变化"
                }
            ],
            "project": existing
        }

    project_id = (
        f"RP-{now_id_time()}-"
        f"{uuid.uuid4().hex[:6].upper()}"
    )

    project = {
        "project_id": project_id,
        "project_name": project_name,
        "project_type": str(
            payload.get(
                "project_type",
                "节能降碳改造"
            )
        ),
        "location": str(
            payload.get("location", "")
        ),
        "project_manager": str(
            payload.get("project_manager", "")
        ),
        "budget_total_cny": round(
            number(payload, "budget_total_cny"),
            2
        ),
        "budget_used_cny": 0.0,
        "start_date": start_date,
        "planned_end_date": str(
            payload.get("planned_end_date", "")
        ),
        "status": "已立项",
        "progress_percent": 0.0,
        "risk_status": "低",
        "risk_reason": "暂无重大风险",
        "overdue_task_count": 0,
        "milestones": normalize_milestones(
            payload.get("milestones", [])
        ),
        "tasks": [],
        "created_at": now_text(),
        "updated_at": now_text()
    }

    save_project(project)

    event_id = (
        f"PE-{now_id_time()}-"
        f"{uuid.uuid4().hex[:6].upper()}"
    )

    append_ledger(
        {
            "event_id": event_id,
            "project_id": project_id,
            "action": "create_project",
            "description": "创建改造项目",
            "created_at": now_text()
        }
    )

    return {
        "result": "ok",
        "employee": "节能改造项目管理 Subagent",
        "message": "改造项目已成功立项",
        "duplicate_detected": False,
        "visible_result": {
            "项目编号": project_id,
            "项目名称": project_name,
            "负责人": project["project_manager"],
            "项目状态": project["status"],
            "总预算": project["budget_total_cny"],
            "计划完成日期": project["planned_end_date"]
        },
        "business_state_changes": [
            {
                "business_object": "改造项目",
                "before": "不存在",
                "after": f"已创建：{project_id}"
            },
            {
                "business_object": "项目台账",
                "before": "无本次记录",
                "after": f"已登记：{event_id}"
            }
        ],
        "project": project
    }


def add_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    project_id = str(
        payload.get("project_id", "")
    ).strip()

    task_name = str(
        payload.get("task_name", "")
    ).strip()

    if not project_id:
        raise ValueError("缺少字段 project_id")

    if not task_name:
        raise ValueError("缺少字段 task_name")

    project = load_project(project_id)

    task_id = (
        f"TASK-{now_id_time()}-"
        f"{uuid.uuid4().hex[:4].upper()}"
    )

    task = {
        "task_id": task_id,
        "task_name": task_name,
        "owner": str(payload.get("owner", "")),
        "status": str(
            payload.get("status", "待开始")
        ),
        "progress_percent": round(
            number(payload, "progress_percent"),
            2
        ),
        "priority": str(
            payload.get("priority", "中")
        ),
        "due_date": str(
            payload.get("due_date", "")
        ),
        "milestone": str(
            payload.get("milestone", "")
        ),
        "task_budget_cny": round(
            number(payload, "task_budget_cny"),
            2
        ),
        "actual_cost_cny": round(
            number(payload, "actual_cost_cny"),
            2
        ),
        "overdue": False,
        "created_at": now_text(),
        "updated_at": now_text()
    }

    project.setdefault("tasks", []).append(task)
    calculate_project_state(project)
    save_project(project)

    event_id = (
        f"PE-{now_id_time()}-"
        f"{uuid.uuid4().hex[:6].upper()}"
    )

    append_ledger(
        {
            "event_id": event_id,
            "project_id": project_id,
            "task_id": task_id,
            "action": "add_task",
            "description": f"新增任务：{task_name}",
            "created_at": now_text()
        }
    )

    return {
        "result": "ok",
        "employee": "节能改造项目管理 Subagent",
        "message": "项目任务已创建",
        "visible_result": {
            "项目编号": project_id,
            "任务编号": task_id,
            "任务名称": task_name,
            "负责人": task["owner"],
            "任务状态": task["status"],
            "截止日期": task["due_date"]
        },
        "business_state_changes": [
            {
                "business_object": "项目任务",
                "before": "不存在",
                "after": f"已创建：{task_id}"
            },
            {
                "business_object": "项目进度",
                "before": "重新计算前",
                "after": f"{project['progress_percent']}%"
            }
        ],
        "task": task,
        "project": project
    }


def update_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    project_id = str(
        payload.get("project_id", "")
    ).strip()

    task_id = str(
        payload.get("task_id", "")
    ).strip()

    if not project_id:
        raise ValueError("缺少字段 project_id")

    if not task_id:
        raise ValueError("缺少字段 task_id")

    project = load_project(project_id)

    target = None

    for task in project.get("tasks", []):
        if task.get("task_id") == task_id:
            target = task
            break

    if target is None:
        raise ValueError(f"未找到任务：{task_id}")

    before = dict(target)

    text_fields = (
        "task_name",
        "owner",
        "status",
        "priority",
        "due_date",
        "milestone"
    )

    for field_name in text_fields:
        if field_name in payload:
            target[field_name] = str(
                payload.get(field_name, "")
            )

    if "progress_percent" in payload:
        target["progress_percent"] = max(
            0.0,
            min(
                100.0,
                number(payload, "progress_percent")
            )
        )

    if "task_budget_cny" in payload:
        target["task_budget_cny"] = round(
            number(payload, "task_budget_cny"),
            2
        )

    if "actual_cost_cny" in payload:
        target["actual_cost_cny"] = round(
            number(payload, "actual_cost_cny"),
            2
        )

    if target.get("status") == "已完成":
        target["progress_percent"] = 100.0

    if float(target.get("progress_percent", 0) or 0) >= 100:
        target["status"] = "已完成"

    target["updated_at"] = now_text()

    calculate_project_state(project)
    save_project(project)

    event_id = (
        f"PE-{now_id_time()}-"
        f"{uuid.uuid4().hex[:6].upper()}"
    )

    append_ledger(
        {
            "event_id": event_id,
            "project_id": project_id,
            "task_id": task_id,
            "action": "update_task",
            "description": f"更新任务：{target['task_name']}",
            "before": before,
            "after": target,
            "created_at": now_text()
        }
    )

    return {
        "result": "ok",
        "employee": "节能改造项目管理 Subagent",
        "message": "项目任务已更新",
        "visible_result": {
            "项目编号": project_id,
            "任务编号": task_id,
            "任务状态": target["status"],
            "任务进度": target["progress_percent"],
            "项目总进度": project["progress_percent"],
            "已用预算": project["budget_used_cny"],
            "风险等级": project["risk_status"]
        },
        "business_state_changes": [
            {
                "business_object": "项目任务",
                "before": (
                    f"{before.get('status')} / "
                    f"{before.get('progress_percent')}%"
                ),
                "after": (
                    f"{target.get('status')} / "
                    f"{target.get('progress_percent')}%"
                )
            },
            {
                "business_object": "项目状态",
                "before": "更新前",
                "after": (
                    f"{project['status']}，"
                    f"进度{project['progress_percent']}%"
                )
            },
            {
                "business_object": "项目预算",
                "before": "重新计算前",
                "after": (
                    f"已用{project['budget_used_cny']}元"
                )
            }
        ],
        "task": target,
        "project": project
    }


def get_project(payload: Dict[str, Any]) -> Dict[str, Any]:
    project_id = str(
        payload.get("project_id", "")
    ).strip()

    project_name = str(
        payload.get("project_name", "")
    ).strip()

    project = None

    if project_id:
        project = load_project(project_id)
    elif project_name:
        for path in PROJECT_DIR.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as file:
                    candidate = json.load(file)
            except (OSError, json.JSONDecodeError):
                continue

            if project_name in str(
                candidate.get("project_name", "")
            ):
                project = candidate
                break
    else:
        raise ValueError(
            "缺少 project_id 或 project_name"
        )

    if project is None:
        return {
            "result": "not_found",
            "employee": "节能改造项目管理 Subagent",
            "message": "未找到对应项目"
        }

    calculate_project_state(project)
    save_project(project)

    return {
        "result": "ok",
        "employee": "节能改造项目管理 Subagent",
        "message": "项目查询完成",
        "visible_result": {
            "项目编号": project["project_id"],
            "项目名称": project["project_name"],
            "项目状态": project["status"],
            "项目进度": project["progress_percent"],
            "总预算": project["budget_total_cny"],
            "已用预算": project["budget_used_cny"],
            "风险等级": project["risk_status"],
            "逾期任务": project["overdue_task_count"]
        },
        "project": project
    }


def list_projects(payload: Dict[str, Any]) -> Dict[str, Any]:
    limit = int(payload.get("limit", 20))
    projects: List[Dict[str, Any]] = []

    for path in PROJECT_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                project = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        projects.append(
            {
                "project_id": project.get("project_id"),
                "project_name": project.get("project_name"),
                "status": project.get("status"),
                "progress_percent": project.get(
                    "progress_percent"
                ),
                "budget_total_cny": project.get(
                    "budget_total_cny"
                ),
                "budget_used_cny": project.get(
                    "budget_used_cny"
                ),
                "risk_status": project.get("risk_status"),
                "updated_at": project.get("updated_at")
            }
        )

    projects.sort(
        key=lambda item: str(item.get("updated_at", "")),
        reverse=True
    )

    projects = projects[:limit]

    return {
        "result": "ok",
        "employee": "节能改造项目管理 Subagent",
        "record_count": len(projects),
        "projects": projects
    }


def main() -> None:
    ensure_directories()

    try:
        payload = load_payload()
        action = str(
            payload.get("action", "get_project")
        )

        with LOCK_FILE.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX
            )

            try:
                if action == "create_project":
                    result = create_project(payload)
                elif action == "add_task":
                    result = add_task(payload)
                elif action == "update_task":
                    result = update_task(payload)
                elif action == "get_project":
                    result = get_project(payload)
                elif action == "list_projects":
                    result = list_projects(payload)
                else:
                    raise ValueError(
                        f"不支持的action：{action}"
                    )
            finally:
                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_UN
                )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
        )

    except Exception as exc:
        print(
            json.dumps(
                {
                    "result": "error",
                    "employee": "节能改造项目管理 Subagent",
                    "message": str(exc)
                },
                ensure_ascii=False,
                indent=2
            )
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
