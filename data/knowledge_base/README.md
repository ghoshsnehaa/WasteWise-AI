# WasteWise AI Knowledge Base

This directory contains the curated recycling and disposal knowledge base used by the RAG (Retrieval-Augmented Generation) pipeline in **WasteWise AI**.

## Regulatory & Geographic Scope
The disposal guidance in this knowledge base is aligned with **Indian Environmental Regulatory Frameworks**:
* **MoEFCC — Solid Waste Management Rules, 2026** (Ministry of Environment, Forest and Climate Change, Govt. of India).
* **MoEFCC & CPCB — Plastic Waste Management Rules & EPR** (Central Pollution Control Board).
* **MoHUA — Swachh Bharat Mission Urban (SBM-U 2.0)** (Ministry of Housing and Urban Affairs).

## Verification Semantics & Status Definitions

Every category in `categories.json` includes explicit verification metadata defining its status:

* **`verified_general_guidance`**: The broad waste-management principle (e.g., source segregation into Wet Waste vs. Dry Waste) is supported by an official government regulatory framework (MoEFCC / CPCB / MoHUA). Item-specific preparation steps represent general practical guidelines aligned with these principles.
* **`partially_verified`**: The category represents a broad, heterogeneous class (e.g., `miscellaneous_trash`) or local-rule-dependent stream (e.g., `textile_trash`) where specific handling requires local verification.

## Key Responsible AI Principles & Policy Guidelines

### 1. Decision Support, Not Municipal Authority
* WasteWise AI is designed solely as an educational **decision-support tool**. It does not replace official municipal solid waste management authorities or local Urban Local Body (ULB) rules.

### 2. Local ULB Rule Dependence
* Specific collection practices, door-to-door collection schedules, scrap collectors (Kabadiwalas), and Dry Waste Collection Centers (DWCCs) vary by city and municipality.
* Users are explicitly reminded that local municipal authority rules take precedence.

### 3. No Universal Bin Color Claims
* Collection bin color-coding schemes (e.g., green for wet waste, blue for dry waste, black for domestic hazardous waste) differ across various Urban Local Bodies (e.g., MCD, MCGM, BBMP).
* WasteWise AI avoids claiming universal bin color rules and focuses on core material stream segregation (Wet Waste vs. Dry Waste).

### 4. Special Handling for Miscellaneous Trash (`miscellaneous_trash`)
* `miscellaneous_trash` is a heterogeneous computer-vision class containing composite objects, non-standard packaging, and mixed materials.
* The system **never** issues a blanket instruction such as *"this always belongs in general trash."* It instructs users to inspect for separable recyclable components and to keep hazardous/e-waste out.

### 5. Special Waste Streams (E-Waste & Batteries)
* Electronic waste, batteries, and hazardous chemicals fall **outside** the 9 trained RealWaste classification classes.
* They are mentioned in guidance only as examples of materials that must be diverted from regular mixed waste streams into specialized hazardous/e-waste collection channels.

## 9 RealWaste Categories Verification Status
1. **Cardboard** — `verified_general_guidance` (MoEFCC — Solid Waste Management Rules, 2026)
2. **Food Organics** — `verified_general_guidance` (MoEFCC — Solid Waste Management Rules, 2026)
3. **Glass** — `verified_general_guidance` (CPCB — Solid Waste Management)
4. **Metal** — `verified_general_guidance` (CPCB — Solid Waste Management)
5. **Miscellaneous Trash** — `partially_verified` (MoEFCC & CPCB — Solid Waste Management)
6. **Paper** — `verified_general_guidance` (MoEFCC — Solid Waste Management Rules, 2026)
7. **Plastic** — `verified_general_guidance` (MoEFCC & CPCB — Plastic Waste Management Rules & EPR)
8. **Textile Trash** — `partially_verified` (MoHUA — Swachh Bharat Mission Urban SBM-U 2.0)
9. **Vegetation** — `verified_general_guidance` (MoEFCC & CPCB — Solid Waste Management)
