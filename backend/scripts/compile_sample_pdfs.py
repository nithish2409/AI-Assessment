"""
Compile sample inspection reports to PDF using ReportLab in backend/venv.
"""
import sys
import json
from pathlib import Path

# Add backend directory
sys.path.insert(0, str(Path("backend").resolve()))

from app.services.inspection_report_generator import generate_pdf_report

def main():
    bldg_json = Path("reports/task1_final/sample_building_inspection_full.json")
    ind_json = Path("reports/task1_final/sample_industrial_inspection_full.json")

    if bldg_json.exists():
        with open(bldg_json, "r") as f:
            data = json.load(f)
        pdf_bytes = generate_pdf_report(data)
        out_pdf = Path("reports/task1_final/sample_building_report.pdf")
        with open(out_pdf, "wb") as f:
            f.write(pdf_bytes)
        print(f"Building PDF compiled: {out_pdf} ({len(pdf_bytes)} bytes)")

    if ind_json.exists():
        with open(ind_json, "r") as f:
            data = json.load(f)
        pdf_bytes = generate_pdf_report(data)
        out_pdf = Path("reports/task1_final/sample_industrial_report.pdf")
        with open(out_pdf, "wb") as f:
            f.write(pdf_bytes)
        print(f"Industrial PDF compiled: {out_pdf} ({len(pdf_bytes)} bytes)")

if __name__ == "__main__":
    main()
