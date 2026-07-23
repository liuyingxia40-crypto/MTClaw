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
        "CARBON_ASSET_DATA_DIR",
        "/data/energy-ai/carbon-assets",
    )
)

ASSET_DIR = DATA_DIR / "assets"
TRANSACTION_DIR = DATA_DIR / "transactions"
LEDGER_FILE = DATA_DIR / "carbon_asset_ledger.jsonl"
LOCK_FILE = DATA_DIR / ".carbon_asset_operations.lock"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_id_time() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def make_id(prefix: str, length: int = 6) -> str:
    return (
        f"{prefix}-{now_id_time()}-"
        f"{uuid.uuid4().hex[:length].upper()}"
    )


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    TRANSACTION_DIR.mkdir(parents=True, exist_ok=True)


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
    default: float = 0.0,
) -> float:
    value = payload.get(key, default)

    if value in (None, ""):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"字段 {key} 必须是数字")


def asset_path(asset_id: str) -> Path:
    return ASSET_DIR / f"{asset_id}.json"


def transaction_path(transaction_id: str) -> Path:
    return TRANSACTION_DIR / f"{transaction_id}.json"


def atomic_write(path: Path, data: Dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_path, path)


def append_ledger(record: Dict[str, Any]) -> None:
    with LEDGER_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )
        file.flush()
        os.fsync(file.fileno())


def load_asset(asset_id: str) -> Dict[str, Any]:
    path = asset_path(asset_id)

    if not path.exists():
        raise ValueError(f"未找到碳资产：{asset_id}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("碳资产文件格式错误")

    return data


def save_asset(asset: Dict[str, Any]) -> None:
    atomic_write(
        asset_path(str(asset["asset_id"])),
        asset,
    )


def normalize_asset_state(asset: Dict[str, Any]) -> None:
    available = round(
        max(
            0.0,
            float(
                asset.get(
                    "quantity_available_tco2e",
                    0,
                )
                or 0
            ),
        ),
        4,
    )

    asset["quantity_available_tco2e"] = available

    market_price = float(
        asset.get(
            "estimated_market_unit_price_cny",
            0,
        )
        or 0
    )

    if market_price <= 0:
        market_price = float(
            asset.get(
                "acquisition_unit_price_cny",
                0,
            )
            or 0
        )

    asset["estimated_market_value_cny"] = round(
        available * market_price,
        2,
    )

    if available <= 0:
        if float(
            asset.get(
                "quantity_retired_tco2e",
                0,
            )
            or 0
        ) > 0:
            asset["status"] = "已全部注销或处置"
        else:
            asset["status"] = "无可用余额"
    else:
        asset["status"] = "持有中"

    asset["updated_at"] = now_text()


def find_duplicate_asset(
    asset_name: str,
    owner: str,
    vintage_year: str,
) -> Dict[str, Any] | None:
    for path in ASSET_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                asset = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        if (
            str(asset.get("asset_name", "")).strip()
            == asset_name
            and str(asset.get("owner", "")).strip()
            == owner
            and str(asset.get("vintage_year", "")).strip()
            == vintage_year
        ):
            return asset

    return None


def register_asset(payload: Dict[str, Any]) -> Dict[str, Any]:
    asset_name = str(
        payload.get("asset_name", "")
    ).strip()

    if not asset_name:
        raise ValueError("缺少字段 asset_name")

    owner = str(payload.get("owner", "")).strip()
    vintage_year = str(
        payload.get("vintage_year", "")
    ).strip()

    quantity = number(
        payload,
        "quantity_tco2e",
    )

    if quantity <= 0:
        raise ValueError(
            "quantity_tco2e必须大于0"
        )

    existing = find_duplicate_asset(
        asset_name,
        owner,
        vintage_year,
    )

    if existing:
        return {
            "result": "ok",
            "employee": "碳资产运营 Subagent",
            "message": (
                "检测到相同名称、所有者和年份的碳资产，"
                "已返回已有记录。"
            ),
            "duplicate_detected": True,
            "visible_result": {
                "资产编号": existing["asset_id"],
                "资产名称": existing["asset_name"],
                "资产类型": existing["asset_type"],
                "可用数量_tCO2e": existing[
                    "quantity_available_tco2e"
                ],
                "资产状态": existing["status"],
            },
            "asset": existing,
        }

    asset_id = make_id("CA")

    acquisition_price = number(
        payload,
        "acquisition_unit_price_cny",
    )

    market_price = number(
        payload,
        "estimated_market_unit_price_cny",
        acquisition_price,
    )

    acquisition_cost = number(
        payload,
        "acquisition_cost_cny",
        quantity * acquisition_price,
    )

    asset = {
        "asset_id": asset_id,
        "asset_name": asset_name,
        "asset_type": str(
            payload.get("asset_type", "CCER")
        ),
        "project_name": str(
            payload.get("project_name", "")
        ),
        "owner": owner,
        "vintage_year": vintage_year,
        "registry_reference": str(
            payload.get("registry_reference", "")
        ),
        "quantity_initial_tco2e": round(
            quantity,
            4,
        ),
        "quantity_available_tco2e": round(
            quantity,
            4,
        ),
        "quantity_retired_tco2e": 0.0,
        "quantity_sold_tco2e": 0.0,
        "quantity_transferred_out_tco2e": 0.0,
        "quantity_transferred_in_tco2e": 0.0,
        "quantity_bought_tco2e": 0.0,
        "acquisition_unit_price_cny": round(
            acquisition_price,
            4,
        ),
        "acquisition_cost_cny": round(
            acquisition_cost,
            2,
        ),
        "estimated_market_unit_price_cny": round(
            market_price,
            4,
        ),
        "estimated_market_value_cny": round(
            quantity * market_price,
            2,
        ),
        "status": "持有中",
        "created_at": now_text(),
        "updated_at": now_text(),
    }

    save_asset(asset)

    event_id = make_id("CE")

    append_ledger(
        {
            "event_id": event_id,
            "asset_id": asset_id,
            "action": "register_asset",
            "description": "登记碳资产",
            "quantity_tco2e": quantity,
            "created_at": now_text(),
        }
    )

    return {
        "result": "ok",
        "employee": "碳资产运营 Subagent",
        "message": "碳资产已完成登记",
        "duplicate_detected": False,
        "visible_result": {
            "资产编号": asset_id,
            "资产名称": asset_name,
            "资产类型": asset["asset_type"],
            "登记数量_tCO2e": quantity,
            "可用数量_tCO2e": quantity,
            "估值金额": asset[
                "estimated_market_value_cny"
            ],
            "资产状态": asset["status"],
        },
        "business_state_changes": [
            {
                "business_object": "碳资产",
                "before": "不存在",
                "after": f"已登记：{asset_id}",
            },
            {
                "business_object": "碳资产台账",
                "before": "无本次记录",
                "after": f"已登记：{event_id}",
            },
        ],
        "asset": asset,
    }


def record_transaction(
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    asset_id = str(
        payload.get("asset_id", "")
    ).strip()

    if not asset_id:
        raise ValueError("缺少字段 asset_id")

    transaction_type = str(
        payload.get("transaction_type", "")
    ).strip().lower()

    aliases = {
        "买入": "buy",
        "卖出": "sell",
        "注销": "retire",
        "履约注销": "retire",
        "转入": "transfer_in",
        "转出": "transfer_out",
    }

    transaction_type = aliases.get(
        transaction_type,
        transaction_type,
    )

    supported = {
        "buy",
        "sell",
        "retire",
        "transfer_in",
        "transfer_out",
    }

    if transaction_type not in supported:
        raise ValueError(
            "transaction_type仅支持buy、sell、retire、"
            "transfer_in、transfer_out"
        )

    quantity = number(payload, "quantity_tco2e")

    if quantity <= 0:
        raise ValueError(
            "quantity_tco2e必须大于0"
        )

    unit_price = number(
        payload,
        "unit_price_cny",
    )

    asset = load_asset(asset_id)

    before_available = float(
        asset.get(
            "quantity_available_tco2e",
            0,
        )
        or 0
    )

    outgoing = transaction_type in {
        "sell",
        "retire",
        "transfer_out",
    }

    if outgoing and quantity > before_available:
        raise ValueError(
            f"可用碳资产不足，当前可用"
            f"{before_available} tCO2e"
        )

    if transaction_type in {
        "buy",
        "transfer_in",
    }:
        asset["quantity_available_tco2e"] = (
            before_available + quantity
        )
    else:
        asset["quantity_available_tco2e"] = (
            before_available - quantity
        )

    field_map = {
        "buy": "quantity_bought_tco2e",
        "sell": "quantity_sold_tco2e",
        "retire": "quantity_retired_tco2e",
        "transfer_in": "quantity_transferred_in_tco2e",
        "transfer_out": "quantity_transferred_out_tco2e",
    }

    field_name = field_map[transaction_type]

    asset[field_name] = round(
        float(asset.get(field_name, 0) or 0)
        + quantity,
        4,
    )

    if unit_price > 0:
        asset[
            "estimated_market_unit_price_cny"
        ] = round(unit_price, 4)

    normalize_asset_state(asset)
    save_asset(asset)

    transaction_id = make_id("CT")

    transaction = {
        "transaction_id": transaction_id,
        "asset_id": asset_id,
        "asset_name": asset["asset_name"],
        "transaction_type": transaction_type,
        "quantity_tco2e": round(quantity, 4),
        "unit_price_cny": round(unit_price, 4),
        "transaction_amount_cny": round(
            quantity * unit_price,
            2,
        ),
        "counterparty": str(
            payload.get("counterparty", "")
        ),
        "transaction_date": str(
            payload.get(
                "transaction_date",
                datetime.now().strftime("%Y-%m-%d"),
            )
        ),
        "purpose": str(
            payload.get("purpose", "")
        ),
        "created_at": now_text(),
    }

    atomic_write(
        transaction_path(transaction_id),
        transaction,
    )

    append_ledger(
        {
            "event_id": make_id("CE"),
            "asset_id": asset_id,
            "transaction_id": transaction_id,
            "action": "record_transaction",
            "transaction_type": transaction_type,
            "quantity_tco2e": quantity,
            "created_at": now_text(),
        }
    )

    type_names = {
        "buy": "买入",
        "sell": "卖出",
        "retire": "履约注销",
        "transfer_in": "转入",
        "transfer_out": "转出",
    }

    return {
        "result": "ok",
        "employee": "碳资产运营 Subagent",
        "message": "碳资产业务操作已完成",
        "visible_result": {
            "交易编号": transaction_id,
            "资产编号": asset_id,
            "操作类型": type_names[
                transaction_type
            ],
            "操作数量_tCO2e": quantity,
            "操作金额": transaction[
                "transaction_amount_cny"
            ],
            "操作前可用数量": before_available,
            "操作后可用数量": asset[
                "quantity_available_tco2e"
            ],
            "资产状态": asset["status"],
        },
        "business_state_changes": [
            {
                "business_object": "碳资产余额",
                "before": (
                    f"{before_available} tCO2e"
                ),
                "after": (
                    f"{asset['quantity_available_tco2e']} "
                    "tCO2e"
                ),
            },
            {
                "business_object": "碳资产交易台账",
                "before": "无本次记录",
                "after": (
                    f"已登记：{transaction_id}"
                ),
            },
        ],
        "transaction": transaction,
        "asset": asset,
    }


def retire_asset(payload: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(payload)
    copied["transaction_type"] = "retire"

    if not copied.get("purpose"):
        copied["purpose"] = "碳排放履约注销"

    return record_transaction(copied)


def get_asset(payload: Dict[str, Any]) -> Dict[str, Any]:
    asset_id = str(
        payload.get("asset_id", "")
    ).strip()

    if not asset_id:
        raise ValueError("缺少字段 asset_id")

    asset = load_asset(asset_id)
    normalize_asset_state(asset)
    save_asset(asset)

    return {
        "result": "ok",
        "employee": "碳资产运营 Subagent",
        "message": "碳资产查询完成",
        "visible_result": {
            "资产编号": asset["asset_id"],
            "资产名称": asset["asset_name"],
            "资产类型": asset["asset_type"],
            "所有者": asset["owner"],
            "初始数量_tCO2e": asset[
                "quantity_initial_tco2e"
            ],
            "可用数量_tCO2e": asset[
                "quantity_available_tco2e"
            ],
            "已注销数量_tCO2e": asset[
                "quantity_retired_tco2e"
            ],
            "估值金额": asset[
                "estimated_market_value_cny"
            ],
            "资产状态": asset["status"],
        },
        "asset": asset,
    }


def get_portfolio(
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    owner_filter = str(
        payload.get("owner", "")
    ).strip()

    assets: List[Dict[str, Any]] = []

    for path in ASSET_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                asset = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        if (
            owner_filter
            and owner_filter
            not in str(asset.get("owner", ""))
        ):
            continue

        normalize_asset_state(asset)
        save_asset(asset)
        assets.append(asset)

    total_initial = round(
        sum(
            float(
                item.get(
                    "quantity_initial_tco2e",
                    0,
                )
                or 0
            )
            for item in assets
        ),
        4,
    )

    total_available = round(
        sum(
            float(
                item.get(
                    "quantity_available_tco2e",
                    0,
                )
                or 0
            )
            for item in assets
        ),
        4,
    )

    total_retired = round(
        sum(
            float(
                item.get(
                    "quantity_retired_tco2e",
                    0,
                )
                or 0
            )
            for item in assets
        ),
        4,
    )

    total_value = round(
        sum(
            float(
                item.get(
                    "estimated_market_value_cny",
                    0,
                )
                or 0
            )
            for item in assets
        ),
        2,
    )

    by_type: Dict[str, Dict[str, float]] = {}

    for asset in assets:
        asset_type = str(
            asset.get("asset_type", "其他")
        )

        summary = by_type.setdefault(
            asset_type,
            {
                "asset_count": 0,
                "available_tco2e": 0.0,
                "estimated_value_cny": 0.0,
            },
        )

        summary["asset_count"] += 1
        summary["available_tco2e"] = round(
            summary["available_tco2e"]
            + float(
                asset.get(
                    "quantity_available_tco2e",
                    0,
                )
                or 0
            ),
            4,
        )

        summary["estimated_value_cny"] = round(
            summary["estimated_value_cny"]
            + float(
                asset.get(
                    "estimated_market_value_cny",
                    0,
                )
                or 0
            ),
            2,
        )

    return {
        "result": "ok",
        "employee": "碳资产运营 Subagent",
        "message": "碳资产组合汇总完成",
        "visible_result": {
            "资产数量": len(assets),
            "初始登记总量_tCO2e": total_initial,
            "当前可用总量_tCO2e": total_available,
            "累计注销总量_tCO2e": total_retired,
            "资产组合估值": total_value,
            "所有者筛选": owner_filter or "全部",
        },
        "portfolio_by_type": by_type,
        "assets": assets,
    }


def list_transactions(
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    limit = int(payload.get("limit", 20))
    asset_id_filter = str(
        payload.get("asset_id", "")
    ).strip()

    transactions: List[Dict[str, Any]] = []

    for path in TRANSACTION_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                transaction = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        if (
            asset_id_filter
            and transaction.get("asset_id")
            != asset_id_filter
        ):
            continue

        transactions.append(transaction)

    transactions.sort(
        key=lambda item: str(
            item.get("created_at", "")
        ),
        reverse=True,
    )

    transactions = transactions[:limit]

    return {
        "result": "ok",
        "employee": "碳资产运营 Subagent",
        "message": "碳资产交易记录查询完成",
        "record_count": len(transactions),
        "transactions": transactions,
    }


def main() -> None:
    ensure_directories()

    try:
        payload = load_payload()
        action = str(
            payload.get("action", "get_portfolio")
        )

        with LOCK_FILE.open(
            "a+",
            encoding="utf-8",
        ) as lock_file:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX,
            )

            try:
                if action == "register_asset":
                    result = register_asset(payload)
                elif action == "record_transaction":
                    result = record_transaction(payload)
                elif action == "retire_asset":
                    result = retire_asset(payload)
                elif action == "get_asset":
                    result = get_asset(payload)
                elif action == "get_portfolio":
                    result = get_portfolio(payload)
                elif action == "list_transactions":
                    result = list_transactions(payload)
                else:
                    raise ValueError(
                        f"不支持的action：{action}"
                    )
            finally:
                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_UN,
                )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

    except Exception as exc:
        print(
            json.dumps(
                {
                    "result": "error",
                    "employee": "碳资产运营 Subagent",
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
