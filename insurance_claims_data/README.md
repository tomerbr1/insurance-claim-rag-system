# Insurance Claims Dataset

Synthetic dataset of 13 insurance claim documents for testing RAG systems with multi-agent architectures.

## Documents Overview

| Claim ID | Type | Claimant | Total Value |
|----------|------|----------|-------------|
| CLM-2024-001847 | Auto Accident | Robert J. Mitchell | $14,050.33 |
| CLM-2024-002156 | Homeowner - Water Damage | Jennifer & Thomas Blackwood | $22,450.00 |
| CLM-2024-002589 | Health Insurance - Surgery | David Chen | $24,578.40 |
| CLM-2024-003012 | Commercial Liability (Slip & Fall) | Patricia Vaughn | $142,500.00 |
| CLM-2024-003458 | Auto - Total Loss | Michelle Torres | $24,255.00 |
| CLM-2024-003891 | Homeowner - Theft | Gregory & Susan Palmer | $21,762.99 |
| CLM-2024-004127 | Workers Compensation | James Rodriguez | $55,803.62 |
| CLM-2024-004583 | Life Insurance - Death Benefit | Estate of William Harrison | $500,547.95 |
| CLM-2024-004891 | Professional Liability (E&O) | Vertex Technologies Inc. | $1,550,000 (reserve) |
| CLM-2024-005234 | Travel Insurance - Trip Cancellation | Dr. Amanda & Kevin Foster | $16,650.00 |

## Table-Based Claims (NEW)

These documents contain embedded tables to test pdfplumber table extraction:

| Claim ID | Type | Table Type | Key Needle Data |
|----------|------|------------|-----------------|
| CLM-2024-006001 | Property Damage - Storm | Financial Breakdown | Roof: $12,500.00 on 2024-09-15 |
| CLM-2024-006002 | Auto - Multi-Vehicle | Coverage/Policy | Medical Payments limit: $5,000 |
| CLM-2024-006003 | Workers Comp - Workplace | Treatment Timeline | MRI: $2,800.00 on 2024-10-03 |

### Table Extraction Test Queries
- "What was the exact cost of roof replacement in the property damage claim?" (needle)
- "What is the Medical Payments coverage limit in the auto claim?" (needle)
- "How much did the MRI cost in the workers comp claim?" (needle)
- "Summarize all costs in the property claim CLM-2024-006001" (summary)

## Needle Data (Hard-to-Find Facts)

These are specific details buried within dense text - ideal for testing "needle in a haystack" retrieval:

| Claim ID | Needle Data | Location in Document |
|----------|-------------|---------------------|
| CLM-2024-001847 | Tow Invoice #T-8827 costs exactly $185.00 | Timeline section |
| CLM-2024-001847 | Officer Daniel Thompson Badge #4421 | Incident description |
| CLM-2024-001847 | Physical therapy: 6 sessions at RehabFirst Clinic | Timeline, April entries |
| CLM-2024-002156 | ServiceMaster arrived at exactly 8:45 PM | Timeline section |
| CLM-2024-002156 | Industrial drying equipment was in place for 8 days | Timeline section |
| CLM-2024-002156 | Pipe failure was due to manufacturing defect | Incident description |
| CLM-2024-002589 | Pre-authorization number: PA-2024-88472 | Insurance processing |
| CLM-2024-002589 | Member total responsibility: $6,144.60 | Financial summary |
| CLM-2024-002589 | Workers comp appeal denied July 2, 2024 | Coordination section |
| CLM-2024-003012 | Coffee spill was present for exactly 8 minutes before incident | Liability analysis |
| CLM-2024-003012 | Last floor inspection was at 10:45 AM | Liability analysis |
| CLM-2024-003012 | Defense costs: $12,450.00 | Settlement summary |
| CLM-2024-003458 | Salvage winning bid: $4,200.00 by JM Auto Parts, Miami FL | Salvage disposition |
| CLM-2024-003458 | CCC valuation initially $21,475.00, adjusted to $22,150.00 | Timeline section |
| CLM-2024-003458 | Lienholder payoff to Capital One: $8,447.23 | Financial summary |
| CLM-2024-003891 | Gold Rolex Submariner ref. 116618LB valued at $38,500.00 | Jewelry inventory |
| CLM-2024-003891 | Jewelry sublimit capped claim at $5,000 (actual loss $55,850) | Settlement calculation |
| CLM-2024-003891 | 1909-S VDB Lincoln Cent valued at $1,150.00 | Coin collection |
| CLM-2024-004127 | Permanent impairment rating: 8% left upper extremity | MMI determination |
| CLM-2024-004127 | Average weekly wage: $1,247.00, TTD rate: $831.33/week | Indemnity benefits |
| CLM-2024-004127 | Injury occurred at 10:45 AM on August 5, 2024 | Timeline section |
| CLM-2024-004583 | Interest paid from date of death: $547.95 | Payment details |
| CLM-2024-004583 | Wire reference: WPT-2024-889234 | Payment details |
| CLM-2024-004583 | Last physical April 12, 2024 - BP 142/86 | Underwriting review |
| CLM-2024-004891 | UAT sign-off email from Tom Henderson on March 28, 2024 | Timeline section |
| CLM-2024-004891 | 47 unauthorized changes by Vertex staff (April 5-30, 2024) | Defense position |
| CLM-2024-004891 | Initial reserve reduced from $1,500,000 to $1,200,000 | Reserve information |
| CLM-2024-005234 | Appendectomy performed at exactly 8:30 AM on October 28, 2024 | Timeline section |
| CLM-2024-005234 | Surgeon: Dr. Michael Torres at Stanford Medical Center | Timeline section |
| CLM-2024-005234 | Airline credit expires September 2025 | Refund status |

## Sample Test Queries

### Summary-Level Questions (for Summarization Agent)
- "What happened in claim CLM-2024-003012?"
- "Give me an overview of all auto-related claims"
- "Which claims are still open?"
- "Summarize the water damage claim timeline"

### Needle-in-Haystack Questions (for Precision Agent)
- "What was the exact towing cost in the auto accident claim CLM-2024-001847?"
- "How long was the coffee spill on the floor before the slip and fall incident?"
- "What is the wire transfer reference number for the life insurance payout?"
- "What was the impairment rating for the workers comp claim?"
- "Who signed off on the UAT for the DataCore ERP implementation?"
- "What time did the appendectomy surgery start?"

## Data Characteristics

- **Timeline density**: Each document contains 10-20 dated events
- **Financial complexity**: Multiple line items, subtotals, deductibles, sublimits
- **Cross-references**: Policy numbers, claim IDs, invoice numbers, badge numbers
- **Nested details**: Important facts buried in dense paragraphs
- **Varied document lengths**: 1-3 pages per claim
