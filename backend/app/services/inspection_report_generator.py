"""
Inspection Report Generator: Generates formal JSON and ReportLab PDF inspection reports.
"""
import io
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as PILImage


def generate_json_report(analysis_data: Dict[str, Any]) -> str:
    """Format structured inspection findings as pretty-printed JSON."""
    clean_data = dict(analysis_data)
    # Remove large base64 images from JSON export if desired, or keep light thumbnails
    return json.dumps(clean_data, indent=2)


def generate_pdf_report(analysis_data: Dict[str, Any]) -> bytes:
    """Compile structured inspection findings into an official PDF document."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    story = []

    # Color palette
    PRIMARY_COLOR = colors.HexColor("#0F172A")  # Slate 900
    ACCENT_COLOR = colors.HexColor("#1E3A8A")   # Blue 900
    BORDER_COLOR = colors.HexColor("#E2E8F0")   # Slate 200

    severity = analysis_data.get("visual_severity", {}).get("visual_severity", "undetermined").lower()
    severity_colors = {
        "high": colors.HexColor("#DC2626"),        # Red 600
        "moderate": colors.HexColor("#D97706"),    # Amber 600
        "low": colors.HexColor("#16A34A"),         # Green 600
        "undetermined": colors.HexColor("#64748B") # Slate 500
    }
    sev_color = severity_colors.get(severity, colors.HexColor("#64748B"))

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=PRIMARY_COLOR,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=ACCENT_COLOR,
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#475569"),
    )

    # 1. Header
    domain_label = "Civil Structural / Building Concrete Inspection" if analysis_data.get("domain") == "building" else "Industrial Machinery & Equipment Inspection"
    story.append(Paragraph("AI Inspection & Visual Severity Screening Report", title_style))
    story.append(Paragraph(f"<b>Domain:</b> {domain_label} | <b>System:</b> Antigravity Dual-Domain Vision Fleet", subtitle_style))

    # 2. Metadata & Visual Severity Summary Box
    insp_id = analysis_data.get("inspection_id", "INSP-UNKNOWN")
    ts = analysis_data.get("timestamp", "N/A")
    priority = analysis_data.get("visual_severity", {}).get("inspection_priority", "monitor").replace("_", " ").upper()
    sev_label = severity.upper()

    summary_table_data = [
        [
            Paragraph(f"<b>Inspection ID:</b> {insp_id}<br/><b>Timestamp:</b> {ts}<br/><b>Image Size:</b> {analysis_data.get('image_width')}x{analysis_data.get('image_height')} px", body_style),
            Paragraph(
                f"<font size=8><b>VISUAL SEVERITY RATING:</b></font><br/>"
                f"<font size=16 color='{sev_color.hexval()}'><b>{sev_label}</b></font><br/>"
                f"<font size=8><b>PRIORITY:</b> {priority}</font>",
                body_style
            ),
        ]
    ]
    summary_table = Table(summary_table_data, colWidths=[330, 210])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # 3. Explainable Rationale
    rationales = analysis_data.get("visual_severity", {}).get("rationale", [])
    if rationales:
        story.append(Paragraph("Visual Severity Screening Rationale:", section_heading))
        for r in rationales:
            story.append(Paragraph(f"• {r}", body_style))
        story.append(Spacer(1, 8))

    # 4. Defect Findings Table
    story.append(Paragraph("Detailed Defect Findings:", section_heading))
    defects = analysis_data.get("defects", [])
    if defects:
        defect_rows = [["ID", "Category", "Source Model", "Finding Type", "Confidence", "Associated Component"]]
        for d in defects:
            conf_val = d.get("confidence", 0.0)
            defect_rows.append([
                d.get("defect_id", "-"),
                d.get("category", "-").capitalize(),
                d.get("source_model", "-"),
                d.get("finding_type", "-").replace("_", " "),
                f"{conf_val * 100:.1f}%" if conf_val <= 1.0 else f"{conf_val:.1f}%",
                d.get("associated_component", "unknown").capitalize()
            ])
        t_defects = Table(defect_rows, colWidths=[55, 95, 115, 95, 75, 105])
        t_defects.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        story.append(t_defects)
    else:
        story.append(Paragraph("<i>No discrete defect findings identified within detector sensitivity limits.</i>", body_style))
    story.append(Spacer(1, 10))

    # 5. Component Localization & OCR (if present)
    components = analysis_data.get("components", [])
    ocr_items = analysis_data.get("ocr_results", [])

    if components or ocr_items:
        comp_summary = ", ".join([f"{c['name'].capitalize()} ({c['confidence']:.2f})" for c in components]) if components else "None localized"
        story.append(Paragraph(f"<b>Localized Asset Components:</b> {comp_summary}", body_style))

        if ocr_items:
            ocr_text_summary = "; ".join([f"'{o['text']}' ({o['confidence']:.2f})" for o in ocr_items])
            story.append(Paragraph(f"<b>Nameplate / Tag OCR Extractions:</b> {ocr_text_summary}", body_style))
        story.append(Spacer(1, 10))

    # 6. Embedded Visual Overlay Image
    overlay_b64 = analysis_data.get("images", {}).get("overlay")
    if overlay_b64:
        try:
            img_bytes = base64.b64decode(overlay_b64)
            pil_img = PILImage.open(io.BytesIO(img_bytes))
            w, h = pil_img.size
            max_w, max_h = 420.0, 240.0
            scale = min(max_w / w, max_h / h, 1.0)
            rl_w, rl_h = w * scale, h * scale

            img_buf = io.BytesIO()
            pil_img.save(img_buf, format="JPEG", quality=85)
            img_buf.seek(0)

            story.append(Paragraph("Annotated Visual Findings Overlay:", section_heading))
            story.append(RLImage(img_buf, width=rl_w, height=rl_h))
            story.append(Spacer(1, 10))
        except Exception as e:
            pass

    # 7. Limitations & Caveats
    limitations = analysis_data.get("visual_severity", {}).get("limitations", [])
    if limitations:
        story.append(Paragraph("Technical Limitations & Constraints:", section_heading))
        for lim in limitations:
            story.append(Paragraph(f"• {lim}", disclaimer_style))
        story.append(Spacer(1, 8))

    # 8. Mandatory Disclaimer
    disclaimer_text = analysis_data.get("disclaimer") or (
        "Automated visual inspection results are preliminary heuristics intended solely for "
        "screening and triage prioritization. They do not constitute a certified structural, mechanical, "
        "or safety assessment. Qualified human professional inspection is required."
    )
    disclaimer_table = Table([[Paragraph(f"<b>MANDATORY NOTICE:</b> {disclaimer_text}", disclaimer_style)]], colWidths=[540])
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FCA5A5")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(disclaimer_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
