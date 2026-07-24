#!/usr/bin/env python3

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


DEFAULT_OUTPUT_DIR = Path("/data/energy-carbon-reports/generated")

# 第一版测试用排放因子，后续可以替换成正式因子库
DEFAULT_ELECTRICITY_FACTOR = 0.5554  # kgCO2/kWh
DEFAULT_GAS_FACTOR = 2.1622          # kgCO2/m³


def output_error(message: str) -> None:
    print(
        json.dumps(
            {
                "result": "error",
                "tool": "energy_carbon_report_generator",
                "error": message,
            },
            ensure_ascii=False,
        )
    )
    sys.exit(1)


def read_number(
    data: dict[str, Any],
    key: str,
    required: bool = False,
    default: float = 0,
) -> float:
    value = data.get(key)

    if value in (None, ""):
        if required:
            output_error(f"缺少必要参数：{key}")
        return default

    try:
        number = float(value)
    except (TypeError, ValueError):
        output_error(f"{key} 必须是数字")

    if number < 0:
        output_error(f"{key} 不能小于0")

    return number


def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name or "energy_carbon_report"


def calculate_results(data: dict[str, Any]) -> dict[str, Any]:
    area = read_number(data, "building_area_m2", required=True)
    electricity = read_number(data, "electricity_kwh", required=True)
    gas = read_number(data, "natural_gas_m3", default=0)
    energy_cost = read_number(data, "energy_cost_cny", default=0)
    occupancy_rate = read_number(data, "occupancy_rate_percent", default=0)

    if area <= 0:
        output_error("building_area_m2 必须大于0")

    electricity_factor = read_number(
        data,
        "electricity_factor_kgco2_per_kwh",
        default=DEFAULT_ELECTRICITY_FACTOR,
    )

    gas_factor = read_number(
        data,
        "gas_factor_kgco2_per_m3",
        default=DEFAULT_GAS_FACTOR,
    )

    electricity_emissions = electricity * electricity_factor / 1000
    gas_emissions = gas * gas_factor / 1000
    total_emissions = electricity_emissions + gas_emissions

    intensity = total_emissions * 1000 / area

    electricity_share = (
        electricity_emissions / total_emissions * 100
        if total_emissions > 0
        else 0
    )

    gas_share = (
        gas_emissions / total_emissions * 100
        if total_emissions > 0
        else 0
    )

    annual_energy_per_area = electricity / area

    return {
        "building_area_m2": area,
        "electricity_kwh": electricity,
        "natural_gas_m3": gas,
        "energy_cost_cny": energy_cost,
        "occupancy_rate_percent": occupancy_rate,
        "electricity_factor": electricity_factor,
        "gas_factor": gas_factor,
        "electricity_emissions_tco2e": electricity_emissions,
        "gas_emissions_tco2e": gas_emissions,
        "total_emissions_tco2e": total_emissions,
        "carbon_intensity_kgco2e_per_m2": intensity,
        "electricity_share_percent": electricity_share,
        "gas_share_percent": gas_share,
        "electricity_intensity_kwh_per_m2": annual_energy_per_area,
    }


def build_recommendations(results: dict[str, Any]) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []

    electricity_share = results["electricity_share_percent"]
    gas_share = results["gas_share_percent"]
    electricity_intensity = results["electricity_intensity_kwh_per_m2"]
    occupancy_rate = results["occupancy_rate_percent"]

    if electricity_share >= 60:
        recommendations.append(
            {
                "priority": "高",
                "title": "优先优化空调及机电系统运行策略",
                "reason": (
                    f"外购电力约占项目碳排放的"
                    f"{electricity_share:.1f}%，是当前最大排放来源。"
                ),
                "measure": (
                    "建议检查冷机、水泵、风机运行时段，按负荷启停设备，"
                    "优化温度设定，并逐步增加分项计量和自动控制。"
                ),
            }
        )

    if electricity_intensity >= 80:
        recommendations.append(
            {
                "priority": "高",
                "title": "开展重点区域用电分项诊断",
                "reason": (
                    f"年度单位面积用电量约为"
                    f"{electricity_intensity:.1f} kWh/㎡。"
                ),
                "measure": (
                    "建议对空调、照明、厨房、洗衣房、电梯及客房用电"
                    "进行分项分析，识别异常时段与高耗能设备。"
                ),
            }
        )

    if gas_share >= 15:
        recommendations.append(
            {
                "priority": "中",
                "title": "优化燃气锅炉与生活热水系统",
                "reason": (
                    f"天然气约占项目碳排放的"
                    f"{gas_share:.1f}%，具有进一步优化空间。"
                ),
                "measure": (
                    "建议检查锅炉燃烧效率、供回水温度、管道保温和"
                    "生活热水循环策略，并评估热泵替代方案。"
                ),
            }
        )

    if occupancy_rate > 0 and occupancy_rate < 75:
        recommendations.append(
            {
                "priority": "中",
                "title": "建立能耗与入住率联动控制",
                "reason": (
                    f"年度平均入住率约为{occupancy_rate:.1f}%，"
                    "固定运行策略可能造成低入住时段能源浪费。"
                ),
                "measure": (
                    "建议根据入住率、客流和营业时段动态调整空调、"
                    "照明、热水及公共区域设备运行计划。"
                ),
            }
        )

    recommendations.append(
        {
            "priority": "中",
            "title": "建立月度能碳监测与复盘机制",
            "reason": "仅依靠年度总量数据难以及时发现能耗异常。",
            "measure": (
                "建议形成月度能源台账、碳排放台账和异常预警机制，"
                "持续跟踪单位面积能耗和主要设备运行效率。"
            ),
        }
    )

    return recommendations[:5]


def create_pdf(
    data: dict[str, Any],
    results: dict[str, Any],
    recommendations: list[dict[str, str]],
    pdf_path: Path,
) -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ChineseTitle",
        parent=styles["Title"],
        fontName="STSong-Light",
        fontSize=22,
        leading=32,
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    subtitle_style = ParagraphStyle(
        "ChineseSubtitle",
        parent=styles["Normal"],
        fontName="STSong-Light",
        fontSize=12,
        leading=20,
        alignment=TA_CENTER,
    )

    heading_style = ParagraphStyle(
        "ChineseHeading",
        parent=styles["Heading2"],
        fontName="STSong-Light",
        fontSize=15,
        leading=24,
        spaceBefore=10,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "ChineseBody",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=10.5,
        leading=19,
        spaceAfter=6,
    )

    small_style = ParagraphStyle(
        "ChineseSmall",
        parent=body_style,
        fontSize=9,
        leading=15,
    )

    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"{data['project_name']}能碳诊断报告",
        author="MTClaw 能碳报告生成 Subagent",
    )

    story = []

    project_name = data["project_name"]
    report_year = str(data.get("report_year", datetime.now().year))
    location = data.get("location", "未提供")
    project_type = data.get("project_type", "酒店")

    story.append(Spacer(1, 28 * mm))
    story.append(
        Paragraph(
            f"{project_name}<br/>{report_year}年度能碳诊断报告",
            title_style,
        )
    )
    story.append(Spacer(1, 10 * mm))
    story.append(
        Paragraph(
            f"项目类型：{project_type}<br/>"
            f"项目地点：{location}<br/>"
            f"生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M')}",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 35 * mm))
    story.append(
        Paragraph(
            "本报告由 MTClaw 能碳报告生成 Subagent 自动生成。"
            "第一版使用测试排放因子，结果仅用于系统演示和内部测试，"
            "不能替代正式碳核查或节能审计。",
            small_style,
        )
    )

    story.append(PageBreak())

    story.append(Paragraph("一、项目概况", heading_style))

    project_table_data = [
        ["项目名称", project_name],
        ["项目类型", project_type],
        ["项目地点", location],
        ["报告年度", report_year],
        ["建筑面积", f"{results['building_area_m2']:,.0f} ㎡"],
        [
            "平均入住率",
            (
                f"{results['occupancy_rate_percent']:.1f}%"
                if results["occupancy_rate_percent"] > 0
                else "未提供"
            ),
        ],
    ]

    project_table = Table(
        project_table_data,
        colWidths=[42 * mm, 110 * mm],
    )

    project_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#AAB6C2")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(project_table)
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("二、能源消费情况", heading_style))

    energy_table_data = [
        ["能源项目", "年度消费量"],
        ["外购电力", f"{results['electricity_kwh']:,.0f} kWh"],
        ["天然气", f"{results['natural_gas_m3']:,.0f} m³"],
        [
            "年度能源费用",
            (
                f"{results['energy_cost_cny']:,.0f} 元"
                if results["energy_cost_cny"] > 0
                else "未提供"
            ),
        ],
        [
            "单位面积用电量",
            f"{results['electricity_intensity_kwh_per_m2']:.1f} kWh/㎡",
        ],
    ]

    energy_table = Table(
        energy_table_data,
        colWidths=[70 * mm, 82 * mm],
    )

    energy_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEBF7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#AAB6C2")),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(energy_table)
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("三、碳排放核算结果", heading_style))

    carbon_table_data = [
        ["指标", "核算结果"],
        [
            "外购电力碳排放",
            f"{results['electricity_emissions_tco2e']:.2f} tCO₂e",
        ],
        [
            "天然气碳排放",
            f"{results['gas_emissions_tco2e']:.2f} tCO₂e",
        ],
        [
            "年度碳排放总量",
            f"{results['total_emissions_tco2e']:.2f} tCO₂e",
        ],
        [
            "单位面积碳排放强度",
            f"{results['carbon_intensity_kgco2e_per_m2']:.2f} kgCO₂e/㎡",
        ],
        [
            "外购电力排放占比",
            f"{results['electricity_share_percent']:.1f}%",
        ],
        [
            "天然气排放占比",
            f"{results['gas_share_percent']:.1f}%",
        ],
    ]

    carbon_table = Table(
        carbon_table_data,
        colWidths=[78 * mm, 74 * mm],
    )

    carbon_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAD3")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#AAB6C2")),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(carbon_table)
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("四、初步诊断", heading_style))

    diagnosis = (
        f"项目年度碳排放总量约为"
        f"{results['total_emissions_tco2e']:.2f} tCO₂e，"
        f"单位面积碳排放强度约为"
        f"{results['carbon_intensity_kgco2e_per_m2']:.2f} kgCO₂e/㎡。"
        f"其中外购电力排放占比约"
        f"{results['electricity_share_percent']:.1f}%，"
        f"天然气排放占比约"
        f"{results['gas_share_percent']:.1f}%。"
        "当前第一版诊断主要依据项目年度总量数据，"
        "后续应补充逐月及分项数据，提高分析准确性。"
    )

    story.append(Paragraph(diagnosis, body_style))

    story.append(Paragraph("五、节能降碳建议", heading_style))

    for index, item in enumerate(recommendations, start=1):
        story.append(
            Paragraph(
                f"{index}. 【{item['priority']}优先级】{item['title']}",
                body_style,
            )
        )
        story.append(
            Paragraph(
                f"判断依据：{item['reason']}<br/>"
                f"建议措施：{item['measure']}",
                body_style,
            )
        )

    story.append(Paragraph("六、数据缺口与后续工作", heading_style))

    story.append(
        Paragraph(
            "为进一步形成正式、可执行的能碳诊断方案，建议补充："
            "逐月电力和燃气账单、空调及动力系统分项数据、"
            "主要设备台账、每日运行时段、客流或入住率数据、"
            "能源价格及历史改造记录。",
            body_style,
        )
    )

    story.append(Paragraph("七、核算说明", heading_style))

    story.append(
        Paragraph(
            f"本次测试使用的电力排放因子为"
            f"{results['electricity_factor']:.4f} kgCO₂/kWh，"
            f"天然气排放因子为"
            f"{results['gas_factor']:.4f} kgCO₂/m³。"
            "上述因子为第一版系统测试参数，正式项目应根据核算年度、"
            "所在地区和适用标准进行更新。",
            body_style,
        )
    )

    document.build(story)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        output_error("请输入有效的JSON数据")

    required_text_fields = ["project_name"]

    for field in required_text_fields:
        if not str(data.get(field, "")).strip():
            output_error(f"缺少必要参数：{field}")

    output_dir_value = str(data.get("output_dir", "")).strip()

    if output_dir_value:
        output_dir = Path(output_dir_value)
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    results = calculate_results(data)
    recommendations = build_recommendations(results)

    project_name = str(data["project_name"]).strip()
    report_year = str(data.get("report_year", datetime.now().year))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = safe_filename(
        f"{project_name}_{report_year}年度能碳诊断报告_{timestamp}"
    )

    pdf_path = output_dir / f"{base_name}.pdf"
    json_path = output_dir / f"{base_name}.json"

    create_pdf(
        data=data,
        results=results,
        recommendations=recommendations,
        pdf_path=pdf_path,
    )

    record = {
        "project": data,
        "calculation_results": results,
        "recommendations": recommendations,
        "generated_files": {
            "pdf": str(pdf_path),
            "json": str(json_path),
        },
        "generated_at": datetime.now().isoformat(),
        "status": "test_report",
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "result": "ok",
                "tool": "energy_carbon_report_generator",
                "message": "能碳诊断报告已生成",
                "project_name": project_name,
                "pdf_path": str(pdf_path),
                "json_path": str(json_path),
                "calculation_results": results,
                "recommendations": recommendations,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
