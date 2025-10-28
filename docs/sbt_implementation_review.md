# SBT Implementation Review: Comparison with `jc_snigdha_def.md`

This document outlines the discrepancies between the current SBT implementation (in `code/pySBT.py` and `code/02_SBT_*.ipynb` notebooks) and the updated specifications outlined in `docs/jc_snigdha_def.md`.

### Overall Summary

The logic for **SBT Delivery Criteria** is largely consistent with the new specifications. However, there are several significant mismatches in the **SBT Eligibility Criteria**, which will require substantial code modifications to align.

---

### Detailed Comparison

#### SBT Eligibility Criteria

| Feature | Specification (`jc_snigdha_def.md`) | Current Implementation | Gap Analysis |
| :--- | :--- | :--- | :--- |
| **1. Controlled Mode Window** | Patient on IMV in ICU for **6 cumulative hours** between 10 PM and **8 AM**. | Patient on IMV in ICU for **6 cumulative hours** between 10 PM and **6 AM**. | **Mismatch.** The time window in the code is 2 hours shorter than specified. |
| **2. Stability Check** | **1 continuous hour** of stability within a **dynamic ±2 hour window** around the hospital's median SAT assessment time. | Stability criteria are bundled with the IMV flag, and this combined condition is checked for **6 cumulative hours** over the entire 10 PM-6 AM window. | **Major Mismatch.** The implementation is fundamentally different. It does not use a dynamic window, nor does it check for a separate, continuous 1-hour period of stability. |
| **3. Hemodynamic Stability** | Patient is on no **more than one** vasoactive drug. | A Norepinephrine-Equivalent (`NEE`) dose is calculated from all active vasoactive drugs, and the patient is considered stable if `NEE <= 0.2`. | **Major Mismatch.** The specification uses a simple **count** of active medications, while the code uses a complex **dose-equivalence calculation**. |
| **4. Respiratory Stability** | FiO₂ ≤ 50%, PEEP ≤ 8 cmH₂O, and SpO₂ ≥ 88%. | `(fio2_set <= 0.5) & (peep_set <= 8) & (spo2 >= 88)` | **Match.** The implementation aligns with the specification. |
| **5. Tracheostomy Exclusion** | Patients are permanently excluded from the cohort *from the day forward* that `tracheostomy = 1`. | In `00_cohort_id.ipynb`, any patient with a tracheostomy record is **completely removed** from the initial cohort. | **Mismatch.** The code performs a global exclusion at the start, rather than the specified day-forward exclusion. |

#### SBT Delivery Criteria

| Feature | Specification (`jc_snigdha_def.md`) | Current Implementation | Gap Analysis |
| :--- | :--- | :--- | :--- |
| **1. EHR Delivery** | Switch to PS/CPAP (with PEEP/PS ≤ 8) or T-piece, sustained for **30 minutes**. A 2-minute version is also requested. | The `process_diagnostic_flip_sbt_optimized_v2` function in `pySBT.py` identifies a switch to PS/CPAP (with PEEP/PS ≤ 8) or T-piece and checks if it is sustained for 2 and 30 minutes, creating separate flags. | **Match.** The implementation correctly identifies the specified ventilator modes and checks for both 2-minute and 30-minute durations. |
| **2. Flowsheet Delivery** | `sbt_delivery_pass_fail` field is not missing. | This is not part of the signature detection logic but is used in the analysis notebooks as a "ground truth" for comparison against the EHR-derived flags. | **Match.** The implementation allows for this comparison as intended. |

---

### Summary of Required Changes

To align with the new specifications, the following changes are necessary:

1.  **Implement Dynamic Stability Window**: The entire eligibility logic in `pySBT.py` needs to be refactored to first determine the hospital-specific median SAT assessment time and then check for a **continuous 1-hour** period of stability within the new ±2 hour window.
2.  **Update Hemodynamic Stability Logic**: The `NEE` calculation should be replaced with a check that ensures no more than one specified vasoactive medication is active at any given time.
3.  **Correct Time Windows**: The overnight window for the initial controlled mode check must be extended from 6 AM to 8 AM.
4.  **Revise Tracheostomy Exclusion**: The logic should be moved from a global filter in `00_cohort_id.ipynb` to a day-forward exclusion within the analysis, ensuring patients are only removed from the cohort after the date of their tracheostomy.
