"""
ReportLab PDF Generation Smoke Test.
Generates a test inspection report PDF and verifies readability.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_smoke_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    story = []

    # Title & Heading
    title_style = ParagraphStyle(
        'SmokeTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=6
    )
    story.append(Paragraph("Task 1: AI Damage Inspection Certificate (Smoke Test)", title_style))
    
    # Inspection Metadata
    meta_style = ParagraphStyle(
        'SmokeMeta',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4B5563')
    )
    story.append(Paragraph("<b>Inspection ID:</b> INSP-20261117-SMOKE-01 | <b>Domain:</b> Building Structural Inspection", meta_style))
    story.append(Paragraph("<b>Timestamp:</b> 2026-11-17 19:35 IST | <b>Inspector Agent:</b> Antigravity Inspection Engine", meta_style))
    story.append(Spacer(1, 14))

    # Findings Table
    table_data = [
        ["Defect ID", "Component", "Defect Type", "Confidence", "Severity Rating", "Action Required"],
        ["DEF-001", "Concrete Column", "concrete_crack", "84.2%", "HIGH", "Immediate Sealing"],
        ["DEF-002", "Concrete Wall", "concrete_crack", "76.5%", "MEDIUM", "Monitor Growth"],
        ["DEF-003", "Base Flange", "corrosion", "91.3%", "MEDIUM", "Wire Brush & Prime"],
        ["DEF-004", "Structural Beam", "exposed_rebar", "94.0%", "CRITICAL", "Structural Review"]
    ]
    t = Table(table_data, colWidths=[65, 100, 95, 75, 95, 110])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
        ('TEXTCOLOR', (4, 4), (4, 4), colors.HexColor('#DC2626')), # Critical red
        ('FONTNAME', (4, 4), (4, 4), 'Helvetica-Bold'),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # Embed sample image if available
    sample_img_path = Path("backend/datasets/damage/processed/images/test/1616.rf.c868709931a671796794fdbb95352c5a.jpg")
    if sample_img_path.exists():
        story.append(Paragraph("<b>Representative Inspection Asset:</b>", styles['Heading3']))
        story.append(Spacer(1, 6))
        story.append(Image(str(sample_img_path), width=240, height=180))

    doc.build(story)
    print(f"PDF generated successfully at: {output_path}")

if __name__ == "__main__":
    out_file = Path("backend/scripts/test_smoke_report.pdf")
    generate_smoke_pdf(out_file)
    print(f"Generated PDF file size: {out_file.stat().st_size} bytes")
