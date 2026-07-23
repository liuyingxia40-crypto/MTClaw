#!/usr/bin/env python3

import json
import sys
from typing import Any


REQUIRED_FIELDS = {
    "project_name": "项目名称",
    "project_type": "项目类型",
    "location": "项目地点",
    "report_year": "报告年度",
    "building_area_m2": "建筑面积",
    "electricity_kwh": "年度用电量",
}

RECOMMENDED_FIELDS = {
    "natural_gas_m3": "年度天然气用量",
    "energy_cost_cny": "年度能源费用",
    "occupancy_rate_percent": "平均入住率或运营负荷率",
    "room_count": "客房数量",
}


def output_json(payload: dict[str, Any], exit_code: int = 0) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    sys.exit(exit_code)


def is_empty(value: Any) -> bool:
    return value is None or value == ""


def validate_positive_number(
    project: dict[str, Any],
    field: str,
    label: str,
    errors: list[str],
) -> None:
    value = project.get(field)

    if is_empty(value):
        return

    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label}必须是数字")
        return

    if number <= 0:
        errors.append(f"{label}必须大于0")


def validate_non_negative_number(
    project: dict[str, Any],
    field: str,
    label: str,
    errors: list[str],
) -> None:
    value = project.get(field)

    if is_empty(value):
        return

    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label}必须是数字")
        return

    if number < 0:
        errors.append(f"{label}不能小于0")


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

    project = request.get("project", request)

    if not isinstance(project, dict):
        output_json(
            {
                "result": "error",
                "error": "project必须是JSON对象",
            },
            1,
        )

    missing_required = []
    missing_recommended = []
    errors = []
    warnings = []

    for field, label in REQUIRED_FIELDS.items():
        if is_empty(project.get(field)):
            missing_required.append(
                {
                    "field": field,
                    "label": label,
                }
            )

    for field, label in RECOMMENDED_FIELDS.items():
        if is_empty(project.get(field)):
            missing_recommended.append(
                {
                    "field": field,
                    "label": label,
                }
            )

    validate_positive_number(
        project,
        "building_area_m2",
        "建筑面积",
        errors,
    )

    validate_non_negative_number(
        project,
        "electricity_kwh",
        "年度用电量",
        errors,
    )

    validate_non_negative_number(
        project,
        "natural_gas_m3",
        "年度天然气用量",
        errors,
    )

    validate_non_negative_number(
        project,
        "energy_cost_cny",
        "年度能源费用",
        errors,
    )

    validate_non_negative_number(
        project,
        "room_count",
        "客房数量",
        errors,
    )

    occupancy_rate = project.get("occupancy_rate_percent")

    if not is_empty(occupancy_rate):
        try:
            occupancy_rate_number = float(occupancy_rate)

            if occupancy_rate_number < 0 or occupancy_rate_number > 100:
                errors.append("平均入住率必须在0到100之间")
        except (TypeError, ValueError):
            errors.append("平均入住率必须是数字")

    report_year = project.get("report_year")

    if not is_empty(report_year):
        try:
            report_year_number = int(report_year)

            if report_year_number < 2000 or report_year_number > 2100:
                errors.append("报告年度不在合理范围内")
        except (TypeError, ValueError):
            errors.append("报告年度必须是整数")

    if float(project.get("electricity_kwh", 0) or 0) == 0:
        warnings.append("年度用电量为0，请确认项目是否确实未使用外购电力")

    if missing_recommended:
        warnings.append(
            "部分推荐字段缺失，报告仍可生成，但诊断深度会受到影响"
        )

    if missing_required or errors:
        output_json(
            {
                "result": "invalid",
                "can_generate_report": False,
                "project_name": project.get("project_name"),
                "missing_required": missing_required,
                "missing_recommended": missing_recommended,
                "errors": errors,
                "warnings": warnings,
                "message": "项目数据校验未通过，暂不能生成正式报告",
            }
        )

    completeness_score = round(
        (
            (
                len(REQUIRED_FIELDS)
                + len(RECOMMENDED_FIELDS)
                - len(missing_recommended)
            )
            / (len(REQUIRED_FIELDS) + len(RECOMMENDED_FIELDS))
        )
        * 100,
        1,
    )

    output_json(
        {
            "result": "ok",
            "can_generate_report": True,
            "project_name": project.get("project_name"),
            "completeness_score": completeness_score,
            "missing_required": [],
            "missing_recommended": missing_recommended,
            "errors": [],
            "warnings": warnings,
            "message": "项目数据校验通过",
        }
    )


if __name__ == "__main__":
    main()
