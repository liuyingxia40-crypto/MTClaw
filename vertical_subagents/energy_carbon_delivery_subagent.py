#!/usr/bin/env python3

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote


BASE_DIR = Path.home() / "moor" / "MTClaw" / "vertical_subagents"

PROJECT_LOADER = BASE_DIR / "project_loader.py"
PROJECT_VALIDATOR = BASE_DIR / "project_data_validator.py"
REPORT_GENERATOR = BASE_DIR / "energy_carbon_report_generator.py"


def output_json(payload: dict[str, Any], exit_code: int = 0) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    sys.exit(exit_code)


def run_python_script(
    script_path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not script_path.is_file():
        output_json(
            {
                "result": "error",
                "error": f"找不到程序：{script_path}",
            },
            1,
        )

    try:
        process = subprocess.run(
            ["python3", str(script_path)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        output_json(
            {
                "result": "error",
                "error": f"程序执行超时：{script_path.name}",
            },
            1,
        )

    stdout = process.stdout.strip()
    stderr = process.stderr.strip()

    if not stdout:
        output_json(
            {
                "result": "error",
                "error": f"程序没有返回结果：{script_path.name}",
                "stderr": stderr,
            },
            1,
        )

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        output_json(
            {
                "result": "error",
                "error": f"程序返回的不是有效JSON：{script_path.name}",
                "stdout": stdout,
                "stderr": stderr,
            },
            1,
        )

    if process.returncode != 0:
        output_json(
            {
                "result": "error",
                "error": f"程序执行失败：{script_path.name}",
                "details": result,
                "stderr": stderr,
            },
            1,
        )

    return result


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

    # 第一步：根据名称自动查找项目
    loader_result = run_python_script(
        PROJECT_LOADER,
        {
            "project_query": project_query,
        },
    )

    if loader_result.get("result") != "ok":
        output_json(loader_result)

    project = loader_result.get("project", {})
    project_dir_value = str(
        loader_result.get("project_dir", "")
    ).strip()

    if not project_dir_value:
        output_json(
            {
                "result": "error",
                "error": "项目目录信息缺失",
            },
            1,
        )

    project_dir = Path(project_dir_value)

    if not project_dir.is_dir():
        output_json(
            {
                "result": "error",
                "error": "项目目录不存在",
                "project_dir": str(project_dir),
            },
            1,
        )

    # 第二步：校验项目数据
    validator_result = run_python_script(
        PROJECT_VALIDATOR,
        {
            "project": project,
        },
    )

    if validator_result.get("can_generate_report") is not True:
        output_json(
            {
                "result": "invalid",
                "subagent": "energy_carbon_delivery_subagent",
                "message": "项目数据校验未通过，暂不能生成报告",
                "project_query": project_query,
                "project_name": project.get("project_name"),
                "project_id": project.get("project_id"),
                "project_dir": str(project_dir),
                "validation": validator_result,
            }
        )

    # 第三步：准备项目独立输出目录
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    generator_payload = dict(project)
    generator_payload["output_dir"] = str(output_dir)

    # 第四步：生成报告
    generator_result = run_python_script(
        REPORT_GENERATOR,
        generator_payload,
    )

    if generator_result.get("result") != "ok":
        output_json(generator_result)

    # 第五步：生成本次交付批次编号
    delivery_time = datetime.now().astimezone()
    delivery_time_iso = delivery_time.isoformat(
        timespec="seconds"
    )
    delivery_batch_id = delivery_time.strftime(
        "DELIVERY-%Y%m%d-%H%M%S"
    )
    delivery_file_suffix = delivery_time.strftime(
        "%Y%m%d_%H%M%S"
    )

    # 第六步：生成客户交付邮件草稿
    project_name = str(project.get("project_name", "项目")).strip()
    report_year = str(project.get("report_year", "")).strip()
    pdf_path = str(generator_result.get("pdf_path", "")).strip()

    # 将PDF复制到公网下载目录，并生成下载地址
    download_dir = Path("/data/energy-carbon-downloads")
    download_dir.mkdir(parents=True, exist_ok=True)

    source_pdf_path = Path(pdf_path)

    if not source_pdf_path.is_file():
        output_json(
            {
                "result": "error",
                "error": "报告PDF文件不存在，无法创建下载链接",
                "pdf_path": pdf_path,
            },
            1,
        )

    download_file_name = source_pdf_path.name
    download_file_path = download_dir / download_file_name
    shutil.copy2(source_pdf_path, download_file_path)

    download_url = (
        "http://8.130.125.178/reports/"
        + quote(download_file_name)
    )

    email_subject = f"{project_name}{report_year}年度能碳诊断报告交付"

    email_body = f"""尊敬的项目负责人：

您好！

{project_name}{report_year}年度能碳诊断工作已经完成，相关报告现已生成。

本次交付内容包括：
1. 项目能耗与碳排放核算结果
2. 主要能碳问题诊断
3. 节能降碳改进建议
4. 能碳诊断报告PDF文件

报告文件：
{pdf_path}

报告下载地址：
{download_url}

请您查收。如需进一步开展节能改造方案设计、负荷分析、三相电异常分析或碳资产机会评估，可继续推进后续工作。

此致
敬礼

企业能碳自主执行工作站
"""

    email_draft_path = (
        output_dir
        / f"delivery_email_{delivery_file_suffix}.txt"
    )
    email_draft_path.write_text(
        f"主题：{email_subject}\n\n{email_body}",
        encoding="utf-8",
    )

    # 第七步：生成交付执行日志
    delivery_log = {
        "result": "ok",
        "subagent": "energy_carbon_delivery_subagent",
        "action": "generate_energy_carbon_delivery",
        "delivery_batch_id": delivery_batch_id,
        "executed_at": delivery_time_iso,
        "project": {
            "project_id": project.get("project_id"),
            "project_name": project_name,
            "report_year": report_year,
            "project_dir": str(project_dir),
            "project_file": loader_result.get("project_file"),
        },
        "validation": {
            "passed": True,
            "completeness_score": validator_result.get(
                "completeness_score"
            ),
            "warnings": validator_result.get("warnings", []),
        },
        "actions_completed": [
            "自动查找项目文件",
            "校验项目基础数据",
            "完成能耗与碳排放计算",
            "生成能碳诊断PDF报告",
            "生成结构化JSON结果",
            "复制报告到公网下载目录",
            "生成报告公网下载地址",
            "生成客户交付邮件草稿",
            "保存交付执行日志",
        ],
        "output_files": {
            "pdf_report": generator_result.get("pdf_path"),
            "json_report": generator_result.get("json_path"),
            "download_file": str(download_file_path),
            "download_url": download_url,
            "email_draft": str(email_draft_path),
        },
        "email": {
            "status": "draft_created",
            "sent": False,
            "requires_confirmation": True,
            "subject": email_subject,
        },
    }

    delivery_log_path = (
        output_dir
        / f"delivery_log_{delivery_file_suffix}.json"
    )
    delivery_log_path.write_text(
        json.dumps(
            delivery_log,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 第八步：返回完整交付结果
    output_json(
        {
            "result": "ok",
            "subagent": "energy_carbon_delivery_subagent",
            "message": "项目能碳诊断报告已自动生成",
            "project_query": project_query,
            "project_name": project.get("project_name"),
            "project_id": project.get("project_id"),
            "project_dir": str(project_dir),
            "project_file": loader_result.get("project_file"),
            "validation": validator_result,
            "pdf_path": generator_result.get("pdf_path"),
            "json_path": generator_result.get("json_path"),
            "download_file_path": str(download_file_path),
            "download_url": download_url,
            "email_draft_path": str(email_draft_path),
            "email_subject": email_subject,
            "delivery_batch_id": delivery_batch_id,
            "delivery_log_path": str(delivery_log_path),
            "executed_at": delivery_time_iso,
            "calculation_results": generator_result.get(
                "calculation_results"
            ),
            "recommendations": generator_result.get(
                "recommendations"
            ),
        }
    )


if __name__ == "__main__":
    main()
