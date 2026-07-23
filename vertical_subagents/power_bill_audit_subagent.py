#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fcntl
import hashlib
import json
import math
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


DATA_DIR = Path(
    os.environ.get(
        "POWER_AUDIT_DATA_DIR",
        "/data/energy-ai/power-audit"
    )
)

LEDGER_FILE = DATA_DIR / "power_audit_ledger.jsonl"
WORK_ORDER_DIR = DATA_DIR / "work_orders"
LOCK_FILE = DATA_DIR / ".power_audit.lock"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_id_time() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WORK_ORDER_DIR.mkdir(parents=True, exist_ok=True)


def load_payload() -> Dict[str, Any]:
    """
    同时支持：
    1. python3 script.py '{"action":"audit", ...}'
    2. echo '{"action":"audit", ...}' | python3 script.py
    """
    raw = ""

    if len(sys.argv) > 1:
        raw = sys.argv[1]
    else:
        raw = sys.stdin.read()

    if not raw.strip():
        raise ValueError("未收到输入 JSON")

    payload = json.loads(raw)

    if not isinstance(payload, dict):
        raise ValueError("输入必须是 JSON 对象")

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


def append_ledger(record: Dict[str, Any]) -> None:
    with LEDGER_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def save_work_order(
    work_order_id: str,
    work_order: Dict[str, Any]
) -> Path:
    path = WORK_ORDER_DIR / f"{work_order_id}.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            work_order,
            file,
            ensure_ascii=False,
            indent=2
        )

    return path


def build_task_list(anomalies: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    tasks: List[Dict[str, str]] = []

    task_map = {
        "BILL_TOTAL_MISMATCH": "复核电费单各费用分项与总额",
        "CAPACITY_OVERSIZED": "核实合同容量并评估容量调整办理条件",
        "CAPACITY_HIGH_LOAD": "核查最大需量及高负荷运行风险",
        "POWER_FACTOR_PENALTY": "检查无功补偿设备及功率因数考核",
        "UNIT_COST_HIGH": "复核峰平谷用电结构及综合电价",
        "MISSING_ELECTRICITY_DATA": "补充本月用电量数据"
    }

    for anomaly in anomalies:
        code = anomaly["code"]

        tasks.append(
            {
                "task_name": task_map.get(
                    code,
                    "复核电费异常事项"
                ),
                "status": "待处理",
                "source_anomaly": code
            }
        )

    return tasks



FINGERPRINT_NUMBER_FIELDS = (
    "electricity_kwh",
    "total_charge_cny",
    "energy_charge_cny",
    "basic_charge_cny",
    "power_factor_adjustment_cny",
    "other_charge_cny",
    "contract_capacity_kva",
    "max_demand_kw",
    "basic_capacity_rate_cny_per_kva_month",
    "baseline_unit_cost_cny_per_kwh",
)


def build_bill_fingerprint(
    payload: Dict[str, Any],
    project_name: str,
    billing_month: str
) -> str:
    """
    根据项目、月份和账单核心数据生成稳定指纹。
    完全相同的账单会得到完全相同的指纹。
    """

    fingerprint_data: Dict[str, Any] = {
        "project_name": project_name.strip().lower(),
        "billing_month": billing_month.strip()
    }

    for field_name in FINGERPRINT_NUMBER_FIELDS:
        fingerprint_data[field_name] = round(
            number(payload, field_name),
            6
        )

    canonical = json.dumps(
        fingerprint_data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":")
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def load_all_ledger_records() -> List[Dict[str, Any]]:
    if not LEDGER_FILE.exists():
        return []

    records: List[Dict[str, Any]] = []

    with LEDGER_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(record, dict):
                records.append(record)

    return records


def rewrite_ledger_records(
    records: List[Dict[str, Any]]
) -> None:
    """
    原子方式重写台账，防止修订元数据写入一半。
    """

    temp_file = LEDGER_FILE.with_suffix(
        ".jsonl.tmp"
    )

    with temp_file.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_file, LEDGER_FILE)


def load_saved_work_order(
    work_order_id: Any
) -> Dict[str, Any] | None:
    if not work_order_id:
        return None

    path = WORK_ORDER_DIR / f"{work_order_id}.json"

    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    return data if isinstance(data, dict) else None


def build_duplicate_result(
    record: Dict[str, Any]
) -> Dict[str, Any]:
    """
    返回已有真实记录，不创建新工单、不写入新台账。
    """

    work_order_id = record.get("work_order_id")
    work_order = load_saved_work_order(
        work_order_id
    )

    audit_id = record.get("audit_id")
    project_name = record.get("project_name")
    billing_month = record.get("billing_month")

    return {
        "result": "ok",
        "employee": "电费账单稽核 Subagent",
        "message": (
            "检测到完全相同的账单请求，"
            "已返回已有稽核结果，未重复创建工单或台账。"
        ),
        "duplicate_detected": True,
        "visible_result": {
            "项目": project_name,
            "账单月份": billing_month,
            "处理模式": "重复请求_返回已有记录",
            "核验状态": record.get(
                "verification_status"
            ),
            "异常数量": record.get(
                "anomaly_count",
                0
            ),
            "稽核记录": audit_id,
            "稽核工单": (
                work_order_id or "未创建"
            ),
            "修订版本": record.get(
                "revision_no",
                1
            ),
            "预计年度节省金额": record.get(
                "estimated_annual_saving_cny",
                0
            )
        },
        "business_state_changes": [
            {
                "business_object": "电费稽核工单",
                "before": (
                    f"已存在：{work_order_id}"
                    if work_order_id
                    else "未创建"
                ),
                "after": "无变化"
            },
            {
                "business_object": "电费稽核台账",
                "before": f"已登记：{audit_id}",
                "after": "无变化"
            }
        ],
        "audit": {
            **record,
            "duplicate_detected": True
        },
        "anomalies": [],
        "recommendations": [
            "本次数据与已有账单完全一致，无需重复执行稽核。"
        ],
        "work_order": work_order,
        "storage": {
            "ledger_file": str(LEDGER_FILE),
            "work_order_file": (
                str(
                    WORK_ORDER_DIR
                    / f"{work_order_id}.json"
                )
                if work_order_id
                else None
            )
        },
        "notice": (
            "本次请求未新增台账记录，"
            "也未重复创建稽核工单。"
        )
    }


def annotate_created_result(
    result: Dict[str, Any],
    bill_fingerprint: str,
    revision_no: int,
    revision_of: str | None
) -> Dict[str, Any]:
    """
    给刚创建的台账和工单补充指纹、版本及修订关系。
    """

    audit = result.get("audit")

    if not isinstance(audit, dict):
        return result

    audit_id = audit.get("audit_id")

    if not audit_id:
        return result

    metadata = {
        "bill_fingerprint": bill_fingerprint,
        "revision_no": revision_no,
        "revision_of": revision_of
    }

    records = load_all_ledger_records()

    for record in reversed(records):
        if record.get("audit_id") == audit_id:
            record.update(metadata)
            break

    rewrite_ledger_records(records)

    audit.update(metadata)

    work_order = result.get("work_order")

    if isinstance(work_order, dict):
        work_order.update(metadata)

        work_order_id = work_order.get(
            "work_order_id"
        )

        if work_order_id:
            save_work_order(
                str(work_order_id),
                work_order
            )

    visible_result = result.get(
        "visible_result"
    )

    if isinstance(visible_result, dict):
        visible_result["处理模式"] = (
            "修订稽核"
            if revision_of
            else "首次稽核"
        )
        visible_result["修订版本"] = revision_no
        visible_result["稽核记录"] = audit_id

    if revision_of:
        result["message"] = (
            "检测到账单数据发生变化，"
            "已创建新的修订稽核记录。"
        )

        changes = result.get(
            "business_state_changes"
        )

        if isinstance(changes, list):
            changes.append(
                {
                    "business_object":
                        "电费稽核版本",
                    "before":
                        f"原记录：{revision_of}",
                    "after":
                        (
                            f"修订记录：{audit_id}"
                            f"（第{revision_no}版）"
                        )
                }
            )

    result["duplicate_detected"] = False

    return result


def audit_bill(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    带并发锁、账单去重和修订控制的正式稽核入口。
    """

    ensure_directories()

    project_name = str(
        payload.get("project_name", "")
    ).strip()

    billing_month = str(
        payload.get("billing_month", "")
    ).strip()

    if not project_name:
        raise ValueError(
            "缺少必填字段 project_name"
        )

    if not billing_month:
        raise ValueError(
            "缺少必填字段 billing_month"
        )

    bill_fingerprint = build_bill_fingerprint(
        payload,
        project_name,
        billing_month
    )

    with LOCK_FILE.open(
        "a+",
        encoding="utf-8"
    ) as lock_file:
        fcntl.flock(
            lock_file.fileno(),
            fcntl.LOCK_EX
        )

        try:
            records = load_all_ledger_records()

            same_month_records = [
                record
                for record in records
                if (
                    str(
                        record.get(
                            "project_name",
                            ""
                        )
                    ).strip() == project_name
                    and str(
                        record.get(
                            "billing_month",
                            ""
                        )
                    ).strip() == billing_month
                )
            ]

            # 完全相同的数据：返回已有结果
            for record in reversed(
                same_month_records
            ):
                if (
                    record.get(
                        "bill_fingerprint"
                    )
                    == bill_fingerprint
                ):
                    return build_duplicate_result(
                        record
                    )

            revision_of = None
            revision_no = 1

            if same_month_records:
                latest_record = (
                    same_month_records[-1]
                )

                revision_of = latest_record.get(
                    "audit_id"
                )

                version_numbers = []

                for record in same_month_records:
                    try:
                        version_numbers.append(
                            int(
                                record.get(
                                    "revision_no",
                                    1
                                )
                            )
                        )
                    except (TypeError, ValueError):
                        version_numbers.append(1)

                revision_no = (
                    max(version_numbers) + 1
                )

            result = _audit_bill_unlocked(
                payload
            )

            return annotate_created_result(
                result,
                bill_fingerprint,
                revision_no,
                revision_of
            )

        finally:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_UN
            )


def _audit_bill_unlocked(payload: Dict[str, Any]) -> Dict[str, Any]:
    project_name = str(
        payload.get("project_name", "")
    ).strip()

    billing_month = str(
        payload.get("billing_month", "")
    ).strip()

    if not project_name:
        raise ValueError("缺少必填字段 project_name")

    if not billing_month:
        raise ValueError("缺少必填字段 billing_month")

    audit_id = f"PA-{now_id_time()}-{uuid.uuid4().hex[:6].upper()}"

    electricity_kwh = number(payload, "electricity_kwh")
    total_charge_cny = number(payload, "total_charge_cny")

    energy_charge_cny = number(payload, "energy_charge_cny")
    basic_charge_cny = number(payload, "basic_charge_cny")
    power_factor_adjustment_cny = number(
        payload,
        "power_factor_adjustment_cny"
    )
    other_charge_cny = number(payload, "other_charge_cny")

    contract_capacity_kva = number(
        payload,
        "contract_capacity_kva"
    )
    max_demand_kw = number(payload, "max_demand_kw")

    capacity_rate = number(
        payload,
        "basic_capacity_rate_cny_per_kva_month"
    )

    baseline_unit_cost = number(
        payload,
        "baseline_unit_cost_cny_per_kwh"
    )

    component_total = (
        energy_charge_cny
        + basic_charge_cny
        + power_factor_adjustment_cny
        + other_charge_cny
    )

    if total_charge_cny <= 0 and component_total > 0:
        total_charge_cny = component_total

    bill_difference = round(
        total_charge_cny - component_total,
        2
    )

    anomalies: List[Dict[str, Any]] = []
    recommendations: List[str] = []

    # 1. 账单金额核验
    tolerance = max(total_charge_cny * 0.005, 1.0)

    if component_total > 0 and abs(bill_difference) > tolerance:
        anomalies.append(
            {
                "code": "BILL_TOTAL_MISMATCH",
                "severity": "高",
                "message": "账单总额与各费用分项合计不一致",
                "evidence": {
                    "账单总额": round(total_charge_cny, 2),
                    "分项合计": round(component_total, 2),
                    "差异金额": bill_difference
                }
            }
        )
        recommendations.append(
            "复核电费单中的税费、附加费及其他费用项目。"
        )

    # 2. 电量数据检查
    if electricity_kwh <= 0:
        anomalies.append(
            {
                "code": "MISSING_ELECTRICITY_DATA",
                "severity": "中",
                "message": "缺少有效的月度用电量",
                "evidence": {
                    "electricity_kwh": electricity_kwh
                }
            }
        )
        recommendations.append(
            "补充月度用电量后重新计算综合电价。"
        )

    # 3. 容量利用率分析
    capacity_utilization = None
    recommended_capacity_kva = None
    monthly_capacity_saving = 0.0
    annual_capacity_saving = 0.0

    if contract_capacity_kva > 0 and max_demand_kw > 0:
        capacity_utilization = (
            max_demand_kw / contract_capacity_kva
        )

        recommended_capacity_kva = int(
            math.ceil(max_demand_kw * 1.15 / 10.0) * 10
        )

        recommended_capacity_kva = min(
            recommended_capacity_kva,
            int(contract_capacity_kva)
        )

        if capacity_utilization < 0.45:
            anomalies.append(
                {
                    "code": "CAPACITY_OVERSIZED",
                    "severity": "中",
                    "message": "合同容量利用率偏低，存在容量优化空间",
                    "evidence": {
                        "合同容量_kVA": contract_capacity_kva,
                        "最大需量_kW": max_demand_kw,
                        "容量利用率": round(
                            capacity_utilization * 100,
                            2
                        ),
                        "测算容量情景值_kVA":
                            recommended_capacity_kva
                    }
                }
            )

            recommendations.append(
                "结合近12个月最大需量和供电规则，评估合同容量调整。"
            )

            if capacity_rate > 0:
                capacity_difference = max(
                    contract_capacity_kva
                    - recommended_capacity_kva,
                    0
                )

                monthly_capacity_saving = round(
                    capacity_difference * capacity_rate,
                    2
                )

                annual_capacity_saving = round(
                    monthly_capacity_saving * 12,
                    2
                )

        elif capacity_utilization > 0.90:
            anomalies.append(
                {
                    "code": "CAPACITY_HIGH_LOAD",
                    "severity": "高",
                    "message": "最大需量接近合同容量，存在高负荷风险",
                    "evidence": {
                        "合同容量_kVA": contract_capacity_kva,
                        "最大需量_kW": max_demand_kw,
                        "容量利用率": round(
                            capacity_utilization * 100,
                            2
                        )
                    }
                }
            )

            recommendations.append(
                "核查高负荷时段，并评估需量控制或容量配置。"
            )

    # 4. 功率因数考核
    if power_factor_adjustment_cny > 0:
        anomalies.append(
            {
                "code": "POWER_FACTOR_PENALTY",
                "severity": "中",
                "message": "本月存在功率因数调整电费支出",
                "evidence": {
                    "功率因数调整电费":
                        power_factor_adjustment_cny
                }
            }
        )

        recommendations.append(
            "检查无功补偿设备运行状态及功率因数考核记录。"
        )

    # 5. 综合电价比较
    unit_cost = None

    if electricity_kwh > 0 and total_charge_cny > 0:
        unit_cost = total_charge_cny / electricity_kwh

        if (
            baseline_unit_cost > 0
            and unit_cost > baseline_unit_cost * 1.10
        ):
            anomalies.append(
                {
                    "code": "UNIT_COST_HIGH",
                    "severity": "中",
                    "message": "本月综合用电单价明显高于基准值",
                    "evidence": {
                        "本月综合电价":
                            round(unit_cost, 4),
                        "基准综合电价":
                            round(baseline_unit_cost, 4),
                        "偏高比例": round(
                            (
                                unit_cost / baseline_unit_cost - 1
                            ) * 100,
                            2
                        )
                    }
                }
            )

            recommendations.append(
                "复核峰平谷用电结构、基本电费和附加费用。"
            )

    verification_status = (
        "已核验_发现异常"
        if anomalies
        else "已核验_无明显异常"
    )

    work_order_id = None
    work_order_path = None
    work_order = None

    # 发现异常后，真正创建工单
    if anomalies:
        work_order_id = (
            f"WO-{now_id_time()}-"
            f"{uuid.uuid4().hex[:6].upper()}"
        )

        work_order = {
            "work_order_id": work_order_id,
            "work_order_type": "电费稽核",
            "project_name": project_name,
            "billing_month": billing_month,
            "status": "待处理",
            "priority": (
                "高"
                if any(
                    item["severity"] == "高"
                    for item in anomalies
                )
                else "中"
            ),
            "created_at": now_text(),
            "tasks": build_task_list(anomalies),
            "anomaly_count": len(anomalies),
            "estimated_annual_saving_cny":
                annual_capacity_saving
        }

        work_order_path = save_work_order(
            work_order_id,
            work_order
        )

    ledger_record = {
        "audit_id": audit_id,
        "project_name": project_name,
        "billing_month": billing_month,
        "verification_status": verification_status,
        "total_charge_cny": round(total_charge_cny, 2),
        "electricity_kwh": round(electricity_kwh, 2),
        "unit_cost_cny_per_kwh": (
            round(unit_cost, 4)
            if unit_cost is not None
            else None
        ),
        "anomaly_count": len(anomalies),
        "work_order_id": work_order_id,
        "estimated_annual_saving_cny":
            annual_capacity_saving,
        "created_at": now_text()
    }

    # 真正写入业务台账
    append_ledger(ledger_record)

    return {
        "result": "ok",
        "employee": "电费账单稽核 Subagent",
        "message": "电费账单核验完成",
        "visible_result": {
            "项目": project_name,
            "账单月份": billing_month,
            "核验状态": verification_status,
            "异常数量": len(anomalies),
            "稽核工单": work_order_id or "未创建",
            "预计年度节省金额":
                annual_capacity_saving
        },
        "business_state_changes": [
            {
                "business_object": "电费账单",
                "before": "待核验",
                "after": verification_status
            },
            {
                "business_object": "电费稽核工单",
                "before": "不存在",
                "after": (
                    f"已创建：{work_order_id}"
                    if work_order_id
                    else "无异常，未创建"
                )
            },
            {
                "business_object": "电费稽核台账",
                "before": "无本次记录",
                "after": f"已登记：{audit_id}"
            }
        ],
        "audit": {
            "audit_id": audit_id,
            "bill_total_cny": round(total_charge_cny, 2),
            "component_total_cny": round(component_total, 2),
            "bill_difference_cny": bill_difference,
            "unit_cost_cny_per_kwh": (
                round(unit_cost, 4)
                if unit_cost is not None
                else None
            ),
            "capacity_utilization_percent": (
                round(capacity_utilization * 100, 2)
                if capacity_utilization is not None
                else None
            ),
            "recommended_capacity_scenario_kva":
                recommended_capacity_kva,
            "estimated_monthly_saving_cny":
                monthly_capacity_saving,
            "estimated_annual_saving_cny":
                annual_capacity_saving
        },
        "anomalies": anomalies,
        "recommendations": recommendations,
        "work_order": work_order,
        "storage": {
            "ledger_file": str(LEDGER_FILE),
            "work_order_file": (
                str(work_order_path)
                if work_order_path
                else None
            )
        },
        "notice": (
            "容量优化金额为测算情景值，正式办理前需结合当地供电规则、"
            "近12个月负荷及合同条款复核。"
        )
    }


def list_records(payload: Dict[str, Any]) -> Dict[str, Any]:
    limit = int(payload.get("limit", 20))

    if not LEDGER_FILE.exists():
        return {
            "result": "ok",
            "employee": "电费账单稽核 Subagent",
            "records": []
        }

    records: List[Dict[str, Any]] = []

    with LEDGER_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    records = records[-limit:]
    records.reverse()

    return {
        "result": "ok",
        "employee": "电费账单稽核 Subagent",
        "record_count": len(records),
        "records": records
    }


def get_work_order(payload: Dict[str, Any]) -> Dict[str, Any]:
    work_order_id = str(
        payload.get("work_order_id", "")
    ).strip()

    if not work_order_id:
        raise ValueError("缺少字段 work_order_id")

    path = WORK_ORDER_DIR / f"{work_order_id}.json"

    if not path.exists():
        return {
            "result": "not_found",
            "message": "未找到对应工单"
        }

    with path.open("r", encoding="utf-8") as file:
        work_order = json.load(file)

    return {
        "result": "ok",
        "employee": "电费账单稽核 Subagent",
        "work_order": work_order
    }


def main() -> None:
    ensure_directories()

    try:
        payload = load_payload()
        action = str(
            payload.get("action", "audit")
        ).strip().lower()

        if action == "audit":
            result = audit_bill(payload)
        elif action == "list":
            result = list_records(payload)
        elif action == "get_work_order":
            result = get_work_order(payload)
        else:
            raise ValueError(
                "action仅支持 audit、list、get_work_order"
            )

    except Exception as error:
        result = {
            "result": "error",
            "employee": "电费账单稽核 Subagent",
            "message": str(error)
        }

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
