"""
Generate synthetic insurance claim PDFs with embedded tables.

Uses ReportLab for PDF creation with proper table formatting.
These PDFs test pdfplumber's table extraction capabilities.

Usage:
    python scripts/generate_table_pdfs.py
"""

from pathlib import Path
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
)

DATA_DIR = Path(__file__).parent.parent / "insurance_claims_data"


def create_table_style(has_header: bool = True) -> TableStyle:
    """Create a standard table style with borders and alternating rows."""
    style_commands = [
        # Borders
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
        # Padding
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        # Alignment
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]

    if has_header:
        style_commands.extend([
            # Header row styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            # Alternating row colors for data rows
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ])

    return TableStyle(style_commands)


def create_property_damage_pdf():
    """
    Generate CLM-2024-006001 - Property Damage with Financial Breakdown Table.

    This PDF contains a detailed financial breakdown table with:
    - Item descriptions
    - Categories
    - Amounts
    - Dates
    """
    output_path = DATA_DIR / "CLM_2024_006001.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                  fontSize=16, spaceAfter=12)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                    fontSize=12, spaceAfter=8, spaceBefore=16)
    body_style = styles['Normal']

    story = []

    # Header
    story.append(Paragraph("INSURANCE CLAIM FILE", title_style))
    story.append(Spacer(1, 12))

    # Claim info
    claim_info = """
    <b>Claim Number:</b> CLM-2024-006001<br/>
    <b>Claim Type:</b> Property Damage - Storm<br/>
    <b>Claimant:</b> Michael and Sarah Thompson<br/>
    <b>Policy:</b> HOME-7745892<br/>
    """
    story.append(Paragraph(claim_info, body_style))
    story.append(Spacer(1, 12))

    # Claim Summary
    story.append(Paragraph("CLAIM SUMMARY", heading_style))
    summary_text = """
    <b>Incident Date:</b> September 8, 2024<br/>
    <b>Report Date:</b> September 9, 2024<br/>
    <b>Location:</b> 4521 Maple Drive, Cedar Rapids, IA 52402<br/>
    """
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 8))

    # Incident description
    story.append(Paragraph("INCIDENT DESCRIPTION", heading_style))
    incident_text = """
    On September 8, 2024, a severe thunderstorm with straight-line winds exceeding
    75 mph caused significant damage to the insured property. A large oak tree
    in the backyard was uprooted and fell onto the roof, causing structural damage
    to the main house. Heavy rainfall following the tree impact resulted in water
    intrusion affecting the upstairs bedrooms and hallway.

    Emergency services were contacted at 3:45 PM. The property was inspected by
    adjuster Rebecca Martinez (ADJ-5587) on September 10, 2024. Temporary roof
    tarping was completed by ServiceMaster on September 9, 2024 to prevent further
    water damage.
    """
    story.append(Paragraph(incident_text, body_style))
    story.append(Spacer(1, 16))

    # Financial Breakdown Table - THE KEY TABLE
    story.append(Paragraph("FINANCIAL BREAKDOWN", heading_style))
    story.append(Spacer(1, 8))

    financial_data = [
        ['Item', 'Category', 'Amount', 'Date'],
        ['Roof replacement', 'Structural', '$12,500.00', '2024-09-15'],
        ['Water damage cleanup', 'Restoration', '$3,200.00', '2024-09-12'],
        ['Temporary housing (16 nights)', 'Living Expenses', '$4,800.00', '2024-09-10'],
        ['Contents - Electronics', 'Personal Property', '$2,150.00', '2024-09-20'],
        ['Contents - Furniture', 'Personal Property', '$3,750.00', '2024-09-20'],
        ['Tree removal', 'Debris Removal', '$1,450.00', '2024-09-11'],
        ['Contractor fees', 'Services', '$1,800.00', '2024-09-22'],
        ['Drywall repair', 'Structural', '$2,350.00', '2024-09-18'],
    ]

    col_widths = [2.2*inch, 1.5*inch, 1.2*inch, 1.0*inch]
    table = Table(financial_data, colWidths=col_widths)
    table.setStyle(create_table_style())
    story.append(table)
    story.append(Spacer(1, 12))

    # Summary totals table
    summary_data = [
        ['', '', 'SUBTOTAL', '$32,000.00'],
        ['', '', 'Deductible', '-$1,500.00'],
        ['', '', 'Depreciation', '-$3,800.00'],
        ['', '', 'TOTAL CLAIM', '$26,700.00'],
    ]
    summary_table = Table(summary_data, colWidths=col_widths)
    summary_table.setStyle(TableStyle([
        ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
        ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
        ('FONTNAME', (2, -1), (3, -1), 'Helvetica-Bold'),
        ('LINEABOVE', (2, -1), (3, -1), 1, colors.black),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 16))

    # Timeline
    story.append(Paragraph("TIMELINE OF EVENTS", heading_style))
    timeline_text = """
    <b>September 8, 2024 - 3:15 PM:</b> Storm damage occurs.<br/>
    <b>September 8, 2024 - 3:45 PM:</b> Emergency services contacted.<br/>
    <b>September 9, 2024 - 8:00 AM:</b> Claim filed via phone (Rep: James Wilson).<br/>
    <b>September 9, 2024 - 2:00 PM:</b> Emergency tarping completed by ServiceMaster.<br/>
    <b>September 10, 2024:</b> Property inspection by Adjuster Rebecca Martinez (ADJ-5587).<br/>
    <b>September 11, 2024:</b> Tree removal completed. Cost: $1,450.00.<br/>
    <b>September 12, 2024:</b> Water damage restoration begins.<br/>
    <b>September 15, 2024:</b> Roof replacement contract signed. Amount: $12,500.00.<br/>
    <b>September 22, 2024:</b> Final repairs completed.<br/>
    <b>September 25, 2024:</b> Final inspection passed.<br/>
    """
    story.append(Paragraph(timeline_text, body_style))
    story.append(Spacer(1, 16))

    # Status
    story.append(Paragraph("CLAIM STATUS: SETTLED", heading_style))
    status_text = """
    <b>Settlement Date:</b> September 28, 2024<br/>
    <b>Settlement Amount:</b> $26,700.00<br/>
    <b>Payment Method:</b> Check #48827 mailed to claimant<br/>
    """
    story.append(Paragraph(status_text, body_style))

    doc.build(story)
    print(f"Created: {output_path}")


def create_auto_coverage_pdf():
    """
    Generate CLM-2024-006002 - Auto Insurance with Coverage/Policy Table.

    This PDF contains a coverage breakdown table showing:
    - Coverage types
    - Limits
    - Deductibles
    - Amounts used
    """
    output_path = DATA_DIR / "CLM_2024_006002.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                  fontSize=16, spaceAfter=12)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                    fontSize=12, spaceAfter=8, spaceBefore=16)
    body_style = styles['Normal']

    story = []

    # Header
    story.append(Paragraph("INSURANCE CLAIM FILE", title_style))
    story.append(Spacer(1, 12))

    # Claim info
    claim_info = """
    <b>Claim Number:</b> CLM-2024-006002<br/>
    <b>Claim Type:</b> Auto - Multi-Vehicle Collision<br/>
    <b>Claimant:</b> Daniel J. Harrison<br/>
    <b>Policy:</b> AUTO-3392847<br/>
    """
    story.append(Paragraph(claim_info, body_style))
    story.append(Spacer(1, 12))

    # Claim Summary
    story.append(Paragraph("CLAIM SUMMARY", heading_style))
    summary_text = """
    <b>Incident Date:</b> October 12, 2024<br/>
    <b>Report Date:</b> October 12, 2024<br/>
    <b>Location:</b> Highway 101 Southbound, Mile Marker 47, San Jose, CA<br/>
    <b>Vehicle:</b> 2022 BMW X5 xDrive40i (VIN: 5UXCR6C05N9K78234)<br/>
    """
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 8))

    # Incident description
    story.append(Paragraph("INCIDENT DESCRIPTION", heading_style))
    incident_text = """
    On October 12, 2024, at approximately 5:35 PM, the insured was traveling
    southbound on Highway 101 when a third-party vehicle (2019 Toyota Camry)
    made an unsafe lane change, causing a chain-reaction collision involving
    three vehicles. The insured's vehicle sustained significant front-end
    damage and was deemed repairable.

    CHP Report #2024-CA-87234 was filed at the scene. Officer Maria Santos
    (Badge #7823) documented the incident. The at-fault driver was cited for
    unsafe lane change. Two passengers in the insured vehicle required medical
    attention for minor injuries.
    """
    story.append(Paragraph(incident_text, body_style))
    story.append(Spacer(1, 16))

    # Coverage Table - THE KEY TABLE
    story.append(Paragraph("POLICY COVERAGE BREAKDOWN", heading_style))
    story.append(Spacer(1, 8))

    coverage_data = [
        ['Coverage Type', 'Limit', 'Deductible', 'Amount Used'],
        ['Collision', '$50,000', '$500', '$18,750.00'],
        ['Comprehensive', '$50,000', '$250', '$0.00'],
        ['Liability - Bodily Injury', '$100,000', '$0', '$15,000.00'],
        ['Liability - Property Damage', '$50,000', '$0', '$8,500.00'],
        ['Medical Payments', '$5,000', '$0', '$3,200.00'],
        ['Uninsured Motorist', '$100,000', '$0', '$0.00'],
        ['Rental Reimbursement', '$30/day', '$0', '$720.00'],
    ]

    col_widths = [2.3*inch, 1.1*inch, 1.0*inch, 1.2*inch]
    table = Table(coverage_data, colWidths=col_widths)
    table.setStyle(create_table_style())
    story.append(table)
    story.append(Spacer(1, 16))

    # Medical summary
    story.append(Paragraph("MEDICAL PAYMENTS DETAIL", heading_style))
    medical_text = """
    The Medical Payments coverage of $5,000 per person was utilized for two
    passengers. Coverage details:

    <b>Passenger 1 (Linda Harrison - spouse):</b><br/>
    - Emergency room visit: $1,850.00<br/>
    - Follow-up appointment: $275.00<br/>
    - Prescription medications: $125.00<br/>
    - Total: $2,250.00<br/>
    <br/>
    <b>Passenger 2 (Emma Harrison - daughter):</b><br/>
    - Emergency room visit: $850.00<br/>
    - X-rays: $100.00<br/>
    - Total: $950.00<br/>
    <br/>
    Combined Medical Payments utilized: $3,200.00 of $5,000 limit.
    """
    story.append(Paragraph(medical_text, body_style))
    story.append(Spacer(1, 16))

    # Rental details
    story.append(Paragraph("RENTAL REIMBURSEMENT", heading_style))
    rental_text = """
    Rental vehicle authorized from October 13, 2024 to November 5, 2024 (24 days).
    <br/><br/>
    <b>Rental Provider:</b> Enterprise Rent-A-Car<br/>
    <b>Daily Rate:</b> $45.00 (covered at $30/day policy limit)<br/>
    <b>Days Covered:</b> 24 days<br/>
    <b>Amount Paid by Policy:</b> $720.00 (24 x $30)<br/>
    <b>Claimant Responsibility:</b> $360.00 (24 x $15 overage)<br/>
    """
    story.append(Paragraph(rental_text, body_style))
    story.append(Spacer(1, 16))

    # Status
    story.append(Paragraph("CLAIM STATUS: SETTLED", heading_style))
    status_text = """
    <b>Settlement Date:</b> November 8, 2024<br/>
    <b>Total Claim Value:</b> $46,170.00<br/>
    <b>Subrogation:</b> Initiated against at-fault driver's insurer (Progressive Policy #PRG-8847231)<br/>
    <b>Payment Method:</b> Direct deposit to claimant account ending in 4492<br/>
    """
    story.append(Paragraph(status_text, body_style))

    doc.build(story)
    print(f"Created: {output_path}")


def create_workers_comp_pdf():
    """
    Generate CLM-2024-006003 - Workers Comp with Timeline/Events Table.

    This PDF contains a treatment timeline table with:
    - Dates
    - Events/procedures
    - Costs
    - Payment status
    """
    output_path = DATA_DIR / "CLM_2024_006003.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                  fontSize=16, spaceAfter=12)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                    fontSize=12, spaceAfter=8, spaceBefore=16)
    body_style = styles['Normal']

    story = []

    # Header
    story.append(Paragraph("WORKERS COMPENSATION CLAIM FILE", title_style))
    story.append(Spacer(1, 12))

    # Claim info
    claim_info = """
    <b>Claim Number:</b> CLM-2024-006003<br/>
    <b>Claim Type:</b> Workers Compensation - Workplace Injury<br/>
    <b>Claimant:</b> Christopher R. Martinez<br/>
    <b>Employer:</b> Pacific Manufacturing Inc.<br/>
    <b>Policy:</b> WC-8874521<br/>
    """
    story.append(Paragraph(claim_info, body_style))
    story.append(Spacer(1, 12))

    # Claim Summary
    story.append(Paragraph("CLAIM SUMMARY", heading_style))
    summary_text = """
    <b>Injury Date:</b> October 1, 2024<br/>
    <b>Report Date:</b> October 1, 2024<br/>
    <b>Location:</b> Pacific Manufacturing - Warehouse B, 2200 Industrial Blvd, Oakland, CA<br/>
    <b>Job Title:</b> Forklift Operator<br/>
    <b>Supervisor:</b> Robert Chen<br/>
    """
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 8))

    # Incident description
    story.append(Paragraph("INJURY DESCRIPTION", heading_style))
    incident_text = """
    On October 1, 2024, at 10:15 AM, the claimant was operating a forklift in
    Warehouse B when a pallet of materials shifted unexpectedly during transport.
    The claimant attempted to stabilize the load but was struck in the left
    shoulder by falling boxes weighing approximately 45 lbs each.

    Immediate first aid was administered by on-site safety officer Patricia Wells.
    The claimant was transported to Kaiser Permanente Oakland Medical Center for
    evaluation. Initial diagnosis indicated a left rotator cuff strain with
    possible partial tear.

    OSHA Incident Report #OSHA-2024-CA-44821 was filed. The incident has been
    classified as a recordable workplace injury.
    """
    story.append(Paragraph(incident_text, body_style))
    story.append(Spacer(1, 16))

    # Treatment Timeline Table - THE KEY TABLE
    story.append(Paragraph("TREATMENT TIMELINE", heading_style))
    story.append(Spacer(1, 8))

    timeline_data = [
        ['Date', 'Event', 'Cost', 'Status'],
        ['2024-10-01', 'Initial ER visit', '$1,250.00', 'Paid'],
        ['2024-10-03', 'MRI - Left shoulder', '$2,800.00', 'Paid'],
        ['2024-10-05', 'Orthopedic consultation - Dr. Amanda Lee', '$450.00', 'Paid'],
        ['2024-10-08', 'Prescription medications', '$185.00', 'Paid'],
        ['2024-10-10', 'Physical therapy - Session 1', '$175.00', 'Paid'],
        ['2024-10-15', 'Physical therapy - Session 2', '$175.00', 'Paid'],
        ['2024-10-20', 'Physical therapy - Session 3', '$175.00', 'Pending'],
        ['2024-10-25', 'Follow-up X-ray', '$320.00', 'Pending'],
        ['2024-10-28', 'Physical therapy - Session 4', '$175.00', 'Scheduled'],
        ['2024-11-01', 'Orthopedic follow-up', '$450.00', 'Scheduled'],
        ['2024-11-05', 'Physical therapy - Session 5', '$175.00', 'Scheduled'],
        ['2024-11-10', 'Physical therapy - Session 6', '$175.00', 'Scheduled'],
    ]

    col_widths = [1.2*inch, 2.8*inch, 1.0*inch, 0.9*inch]
    table = Table(timeline_data, colWidths=col_widths)
    table.setStyle(create_table_style())
    story.append(table)
    story.append(Spacer(1, 12))

    # Cost summary
    story.append(Paragraph("MEDICAL COST SUMMARY", heading_style))
    cost_summary = """
    <b>Total Medical Costs to Date:</b> $6,505.00<br/>
    <b>Paid:</b> $5,035.00<br/>
    <b>Pending:</b> $495.00<br/>
    <b>Scheduled:</b> $975.00<br/>
    """
    story.append(Paragraph(cost_summary, body_style))
    story.append(Spacer(1, 16))

    # Indemnity benefits
    story.append(Paragraph("INDEMNITY BENEFITS", heading_style))
    indemnity_text = """
    <b>Average Weekly Wage:</b> $1,380.00<br/>
    <b>Temporary Total Disability (TTD) Rate:</b> $920.00/week<br/>
    <b>TTD Period:</b> October 2, 2024 - November 15, 2024 (estimated)<br/>
    <b>Total TTD Benefits:</b> $5,980.00 (6.5 weeks)<br/>
    <br/>
    The claimant has been placed on light duty restrictions pending full medical
    clearance. Return to full duty is anticipated by November 18, 2024.
    """
    story.append(Paragraph(indemnity_text, body_style))
    story.append(Spacer(1, 16))

    # MRI Results detail (needle data)
    story.append(Paragraph("MRI FINDINGS", heading_style))
    mri_text = """
    MRI performed on October 3, 2024 at Kaiser Permanente Oakland Medical Center.
    Cost: $2,800.00. Radiologist: Dr. Kevin Park.

    <b>Findings:</b>
    - Partial thickness tear of the supraspinatus tendon (Grade II)
    - Mild subacromial bursitis
    - No evidence of complete rotator cuff tear
    - Mild degenerative changes in the acromioclavicular joint

    <b>Recommendation:</b> Conservative treatment with physical therapy.
    Surgical intervention not indicated at this time.
    """
    story.append(Paragraph(mri_text, body_style))
    story.append(Spacer(1, 16))

    # Status
    story.append(Paragraph("CLAIM STATUS: OPEN", heading_style))
    status_text = """
    <b>Current Status:</b> Active treatment - Physical therapy ongoing<br/>
    <b>Expected MMI Date:</b> November 15, 2024<br/>
    <b>Claims Examiner:</b> Sandra Williams (Ext. 4421)<br/>
    <b>Next Review Date:</b> November 5, 2024<br/>
    """
    story.append(Paragraph(status_text, body_style))

    doc.build(story)
    print(f"Created: {output_path}")


def main():
    """Generate all three table PDFs."""
    print("Generating synthetic PDFs with embedded tables...")
    print(f"Output directory: {DATA_DIR}")
    print("-" * 50)

    create_property_damage_pdf()
    create_auto_coverage_pdf()
    create_workers_comp_pdf()

    print("-" * 50)
    print("Done! Created 3 new PDFs with tables.")
    print("\nNew files:")
    print("  - CLM_2024_006001.pdf (Property Damage - Financial Breakdown Table)")
    print("  - CLM_2024_006002.pdf (Auto Insurance - Coverage/Policy Table)")
    print("  - CLM_2024_006003.pdf (Workers Comp - Treatment Timeline Table)")


if __name__ == "__main__":
    main()
