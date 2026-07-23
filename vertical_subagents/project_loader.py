#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECTS_ROOT = Path("/data/energy-carbon-projects")


def output_json(payload: dict[str, Any], exit_code: int = 0) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    sys.exit(exit_code)


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[\s\-_]+", "", value)
    return value


def load_project_json(project_dir: Path) -> dict[str, Any] | None:
    project_file = project_dir / "input" / "project.json"

    if not project_file.is_file():
        return None

    try:
        with project_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    return data


def calculate_match_score(
    query: str,
    project_dir: Path,
    project_data: dict[str, Any],
) -> int:
    normalized_query = normalize_text(query)

    candidates = [
        str(project_data.get("project_name", "")),
        str(project_data.get("project_id", "")),
        project_dir.name,
    ]

    aliases = project_data.get("project_aliases", [])

    if isinstance(aliases, list):
        candidates.extend(str(item) for item in aliases)

    normalized_candidates = [
        normalize_text(candidate)
        for candidate in candidates
        if candidate
    ]

    score = 0

    for candidate in normalized_candidates:
        if normalized_query == candidate:
            score = max(score, 100)
        elif normalized_query in candidate or candidate in normalized_query:
            score = max(score, 70)

    return score


def find_project(query: str) -> dict[str, Any]:
    if not PROJECTS_ROOT.is_dir():
        output_json(
            {
                "result": "error",
                "error": f"项目根目录不存在：{PROJECTS_ROOT}",
            },
            1,
        )

    matches: list[dict[str, Any]] = []

    for project_dir in PROJECTS_ROOT.iterdir():
        if not project_dir.is_dir():
            continue

        project_data = load_project_json(project_dir)

        if project_data is None:
            continue

        score = calculate_match_score(
            query=query,
            project_dir=project_dir,
            project_data=project_data,
        )

        if score > 0:
            matches.append(
                {
                    "score": score,
                    "project_dir": str(project_dir),
                    "project_file": str(
                        project_dir / "input" / "project.json"
                    ),
                    "project": project_data,
                }
            )

    matches.sort(key=lambda item: item["score"], reverse=True)

    if not matches:
        output_json(
            {
                "result": "not_found",
                "query": query,
                "message": "没有找到匹配的能碳项目",
            }
        )

    best_match = matches[0]

    same_score_matches = [
        item for item in matches
        if item["score"] == best_match["score"]
    ]

    if len(same_score_matches) > 1:
        output_json(
            {
                "result": "ambiguous",
                "query": query,
                "message": "找到多个相似项目，请提供更准确的项目名称或编号",
                "matches": [
                    {
                        "project_name": item["project"].get("project_name"),
                        "project_id": item["project"].get("project_id"),
                        "project_dir": item["project_dir"],
                    }
                    for item in same_score_matches
                ],
            }
        )

    return {
        "result": "ok",
        "query": query,
        "project_dir": best_match["project_dir"],
        "project_file": best_match["project_file"],
        "project": best_match["project"],
    }


def main() -> None:
    try:
        request = json.load(sys.stdin)
    except json.JSONDecodeError:
        output_json(
            {
                "result": "error",
                "error": "请输入有效JSON",
            },
            1,
        )

    project_query = str(request.get("project_query", "")).strip()

    if not project_query:
        output_json(
            {
                "result": "error",
                "error": "缺少参数 project_query",
            },
            1,
        )

    result = find_project(project_query)
    output_json(result)


if __name__ == "__main__":
    main()

