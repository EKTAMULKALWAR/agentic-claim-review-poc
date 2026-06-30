"""
Payer policy knowledge base — 15 policies grounded in 2026 CMS guidelines.
Sources: Medicare Program Integrity Manual (PIM), CMS Claims Processing Manual,
AMA CPT 2024/2026, NCCI Policy Manual, Medicare Benefit Policy Manual.
"""

POLICIES = [
    {
        "id": "POL-001",
        "title": "E/M Level Selection — Medical Decision Making (2024+)",
        "source": "CMS Claims Processing Manual Ch. 12; AMA CPT 2024",
        "effective": "2024-01-01",
        "text": (
            "Effective January 2024, Evaluation & Management codes 99202–99215 are selected "
            "by either Medical Decision Making (MDM) complexity or total physician time. "
            "MDM levels require: straightforward (99202/99212), low (99203/99213), moderate "
            "(99204/99214, requires independent interpretation of tests or prescription drug "
            "management), or high (99205/99215, requires drug therapy requiring intensive monitoring "
            "or decision regarding hospitalization). Billing a higher E/M level than MDM or time "
            "documentation supports constitutes upcoding. Auditors compare billed code distribution "
            "against specialty peer benchmarks; providers billing >80% at 99215 warrant review."
        ),
    },
    {
        "id": "POL-002",
        "title": "NCCI — Unbundling of Procedure Codes",
        "source": "CMS NCCI Policy Manual for Medicare Services, v31.0 (2025); CMS-1500 guidelines",
        "effective": "2025-01-01",
        "text": (
            "The National Correct Coding Initiative (NCCI) establishes Column 1/Column 2 code "
            "pairs where the Column 2 code is a component of the Column 1 comprehensive code. "
            "Billing both codes without a valid NCCI-associated modifier (25, 59, XE, XP, XS, XU) "
            "is unbundling and is not separately reimbursable. Mutually exclusive edits prohibit "
            "billing two codes that represent services that cannot reasonably be performed together. "
            "High claim counts of NCCI-paired codes from the same provider on the same date of "
            "service, combined with modifier 59 overuse, trigger post-payment review."
        ),
    },
    {
        "id": "POL-003",
        "title": "Excessive Service Volume — Utilization Outliers",
        "source": "CMS Program Integrity Manual (PIM) Ch. 4 §4.3; OIG Work Plan 2025–2026",
        "effective": "2023-01-01",
        "text": (
            "Providers billing the same HCPCS/CPT code at volumes materially exceeding "
            "the 95th percentile of peers with the same specialty and procedure code require "
            "medical review. A services-per-beneficiary ratio exceeding 5.0 for most procedure "
            "categories, or total service counts more than 3 standard deviations above the "
            "specialty+HCPCS peer mean, is a primary Unified Program Integrity Contractor (UPIC) "
            "trigger. OIG 2026 work plan specifically targets high-volume billing of oncology "
            "drug administration, home health, and durable medical equipment codes."
        ),
    },
    {
        "id": "POL-004",
        "title": "Medicare Frequency Limits on Covered Services",
        "source": "Medicare Benefit Policy Manual Ch. 15; LCD/NCD database (CMS.gov)",
        "effective": "2026-01-01",
        "text": (
            "CMS establishes frequency limitations for preventive and repeat therapeutic services: "
            "screening colonoscopy every 120 months (60 months for high-risk patients); "
            "cardiovascular disease screening every 5 years; bone density measurement every "
            "24 months; initial preventive physical examination (IPPE/Welcome to Medicare) once "
            "per beneficiary lifetime; annual wellness visit once per calendar year. "
            "Billing the same preventive service multiple times per beneficiary per year, "
            "or within the prohibited frequency window, indicates medically unnecessary services "
            "or duplicate billing. Review is triggered when services-per-beneficiary exceeds the "
            "clinically plausible annual frequency for the procedure."
        ),
    },
    {
        "id": "POL-005",
        "title": "Medical Necessity — Coverage and Documentation Standards",
        "source": "Social Security Act §1862(a)(1)(A); CMS PIM Ch. 13; LCD framework",
        "effective": "2023-01-01",
        "text": (
            "Medicare covers services that are reasonable and necessary for the diagnosis or "
            "treatment of illness or injury. Documentation must establish: (1) a covered diagnosis "
            "consistent with the service billed, (2) that the service was actually performed, "
            "(3) the clinical rationale for the service. Local Coverage Determinations (LCDs) "
            "specify diagnosis codes that support medical necessity for specific procedures. "
            "Claims where the billed HCPCS code is inconsistent with the provider's documented "
            "diagnoses, or where high-cost procedures are billed without supporting comorbidity "
            "profiles typical for the specialty, are flagged for prepayment or post-payment review."
        ),
    },
    {
        "id": "POL-006",
        "title": "High-Cost Drug Administration — J-Code Billing (ASP Methodology)",
        "source": "CMS ASP Drug Pricing Files Q1 2026; Medicare Claims Processing Manual Ch. 17",
        "effective": "2026-01-01",
        "text": (
            "Separately payable drugs administered in physician offices or outpatient settings "
            "are reimbursed at Average Sales Price (ASP) + 6% under the competitive acquisition "
            "program. J-code claims must reflect the drug, dosage, and route of administration "
            "as specified in the HCPCS descriptor. Billing for quantities exceeding what is "
            "clinically documented, billing at list price (AWP) rather than ASP, or billing "
            "for drug wastage without proper documentation violates CMS overpayment rules. "
            "A charge-to-payment ratio exceeding 2.0 for ASP-priced drugs indicates potential "
            "billing at inflated acquisition cost. High total service counts for costly oncology "
            "drugs (e.g., oxaliplatin, pembrolizumab, bevacizumab) across a small beneficiary "
            "panel trigger drug integrity review."
        ),
    },
    {
        "id": "POL-007",
        "title": "Specialty-Inappropriate Billing — Scope of Practice",
        "source": "CMS Provider Enrollment, Chain and Ownership System (PECOS); PIM Ch. 4",
        "effective": "2023-01-01",
        "text": (
            "Providers must bill only codes that fall within their enrolled specialty's scope "
            "of clinical practice. CMS cross-references billed HCPCS codes against the provider's "
            "registered specialty in PECOS. Examples of specialty-inappropriate billing include: "
            "an internal medicine physician billing high volumes of surgical procedure codes "
            "(CPT 10000–69999); a physical therapist billing independent laboratory or radiology "
            "interpretations; a podiatrist billing cardiac catheterization codes. "
            "Specialty mismatch combined with high charge amounts or high service volumes "
            "triggers UPIC probe audits and may result in revocation of Medicare billing privileges."
        ),
    },
    {
        "id": "POL-008",
        "title": "Place of Service Mismatch — Facility vs. Non-Facility Rates",
        "source": "CMS Claims Processing Manual Ch. 26 §10.5; 42 CFR §414.22",
        "effective": "2023-01-01",
        "text": (
            "Medicare reimburses professional services at different rates depending on the place "
            "of service (POS): non-facility (POS 11 — office) rates are higher because the "
            "physician bears practice expense, while facility rates (POS 21 hospital inpatient, "
            "POS 22 outpatient, POS 23 emergency) are lower because the facility absorbs overhead. "
            "Billing POS 11 for services rendered in a hospital or ambulatory surgical center "
            "results in overpayment and is a top CMS target. High average submitted charges for "
            "procedure codes that are predominantly performed in facility settings signal potential "
            "POS miscoding."
        ),
    },
    {
        "id": "POL-009",
        "title": "Split/Shared E/M Visits — 2024 Final Rule",
        "source": "CMS CY2024 Physician Fee Schedule Final Rule (88 FR 78818); CPT 99202–99215",
        "effective": "2024-01-01",
        "text": (
            "For hospital inpatient, outpatient, and nursing facility E/M visits rendered by "
            "a physician and a non-physician practitioner (NPP) in the same group, the claim "
            "must be billed under the provider who performed the substantive portion — defined "
            "as more than half the total time of the combined visit. The supervising physician's "
            "NPI must appear in the ordering/referring field. Billing both the physician and NPP "
            "separately for the same encounter, or billing under the physician when the NPP "
            "performed the majority of the visit, constitutes duplicate billing. "
            "High E/M volumes from a single physician NPI with corresponding NPP credentials "
            "suggest improper split-billing attribution."
        ),
    },
    {
        "id": "POL-010",
        "title": "Incident-to Billing — Direct Supervision Requirements",
        "source": "Medicare Benefit Policy Manual Ch. 15 §60; 42 CFR §410.26",
        "effective": "2023-01-01",
        "text": (
            "Services billed incident-to a physician must meet all of: (1) performed by auxiliary "
            "personnel employed by the physician or group, (2) integral and incidental to the "
            "physician's professional service, (3) rendered under the physician's direct personal "
            "supervision (physician physically present in the office suite), (4) for an established "
            "patient with an established condition. New problems require direct physician "
            "involvement and cannot be billed incident-to. Incident-to billing is not permitted "
            "in institutional settings. Auditors flag providers with high E/M volumes billed "
            "under the physician NPI where payroll records do not support corresponding auxiliary "
            "staff capacity."
        ),
    },
    {
        "id": "POL-011",
        "title": "Global Surgery Package — Included Services",
        "source": "CMS Claims Processing Manual Ch. 12 §40; CPT Global Period definitions",
        "effective": "2023-01-01",
        "text": (
            "The global surgical package payment includes all pre-operative services (day before "
            "major, same day for minor), intra-operative services, complications not requiring "
            "additional trips to OR, post-operative visits within the global period (90 days for "
            "major surgery, 10 days for minor surgery), and related medical or surgical services "
            "by the same surgeon. Separately billing post-operative E/M visits within the global "
            "period without modifier 24 (unrelated condition) or 79 (unrelated procedure), "
            "or billing for services included in the surgical package, constitutes unbundling. "
            "High E/M claim counts from surgical specialists within 90 days of major procedures "
            "on the same beneficiary trigger global period edit reviews."
        ),
    },
    {
        "id": "POL-012",
        "title": "Submitted Charge Inflation — Charge-to-Allowed Ratio Outliers",
        "source": "CMS OIG Advisory Opinion; PIM Ch. 4 §4.3.3; HEAT Task Force Guidelines",
        "effective": "2023-01-01",
        "text": (
            "Submitted charges that exceed the Medicare allowed amount by a factor of 5 or more "
            "are considered outlier charges and trigger automated review. While Medicare does not "
            "limit submitted charges, systematic charge inflation — particularly when combined "
            "with high service volume or specialty-code mismatch — indicates potential balance "
            "billing, fraudulent charge master manipulation, or billing for services not rendered. "
            "The Healthcare Fraud Prevention Partnership (HFPP) benchmarks charge-to-allowed "
            "ratios by specialty and HCPCS; providers in the top 1% of this ratio within their "
            "peer group are referred to UPIC for investigation."
        ),
    },
    {
        "id": "POL-013",
        "title": "Modifier Misuse — Modifiers 25, 59, and XE/XP/XS/XU",
        "source": "CMS Modifier 59 Article (MM8863); NCCI Policy Manual v31.0",
        "effective": "2015-01-01",
        "text": (
            "Modifier 25 appended to an E/M code on the same day as a procedure signifies a "
            "significant, separately identifiable service unrelated to the procedure performed. "
            "Documentation must support that the E/M addressed a separate problem, not pre/post "
            "procedural care. Modifier 59 indicates a distinct procedural service; CMS created "
            "selective NCCI-associated modifiers (XE, XP, XS, XU) for specificity. "
            "Systematic overuse of modifier 59 to bypass NCCI edits without clinical justification "
            "constitutes abuse. Providers with modifier 25 or 59 appended to >50% of procedure "
            "claims, or NCCI edit bypass rates exceeding specialty peer averages, are flagged "
            "for medical review."
        ),
    },
    {
        "id": "POL-014",
        "title": "Telehealth Billing Requirements — CY2026",
        "source": "CMS CY2026 Physician Fee Schedule Proposed Rule; Consolidated Appropriations Act 2024",
        "effective": "2026-01-01",
        "text": (
            "For CY2026, Medicare telehealth services are covered for beneficiaries at any "
            "geographic location and originating site following post-PHE permanency provisions. "
            "Synchronous audio-video is required for most services; audio-only visits (G2252) "
            "are permitted only when video is technically unavailable and the patient cannot "
            "access video technology, with required patient consent documentation. "
            "Mental health telehealth requires an in-person visit within 12 months of the first "
            "telehealth service and annually thereafter. Billing standard E/M codes without "
            "telehealth place of service (POS 02 or 10) for remotely rendered services, "
            "or billing audio-only visits as standard synchronous telehealth, is incorrect billing."
        ),
    },
    {
        "id": "POL-015",
        "title": "Anti-Markup Rule — Purchased Diagnostic Tests",
        "source": "42 CFR §414.50; Medicare Claims Processing Manual Ch. 1 §30.2.9",
        "effective": "2009-01-01",
        "text": (
            "When a physician or group bills Medicare for a diagnostic test (laboratory, radiology, "
            "or other) that was performed by an outside supplier, the billing entity may not mark "
            "up the price. Reimbursement is the lower of: the performing supplier's net charge to "
            "the billing physician, the Medicare fee schedule amount, or the billing physician's "
            "actual charge. The anti-markup rule applies whenever the performing supplier is not "
            "an employee of the billing physician or does not share a practice location. "
            "Providers with high average submitted charges for purchased diagnostic services "
            "that significantly exceed the Medicare allowed amount indicate anti-markup violations."
        ),
    },
]
