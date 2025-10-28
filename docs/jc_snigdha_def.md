**SBT Eligibility Criteria**
A patient is eligible for an SBT if all the following conditions are met:
1. IMV in a Controlled Mode for at Least 6 Hours Between 10 PM and 8 AM
The patient must be in a controlled ventilation mode (mode_category = "Assist Control-Volume Control", "Pressure Control", "Pressure-Regulated Volume Control", etc.) for at least 6 cumulative hours between 10 PM and 8 AM. We may be called out for over-sensitivity with 6 hours as one might argue that 6 hours on the vent is not enough to justify SBT. Trials used 12 hours. I think we could build a secondary 12 hour criteria to allow looking at this
The patient must remain in the ICU (location_category = "ICU") during this period, recorded in the adt table.

2. Hemodynamic and Respiratory Stability Based on sat_screen or sat_delivery_flowsheet Median Time
New Timing Window: Instead of using a fixed 6 AM–5 PM window, we determine the median recorded time of either:
sat_screen OR sat_delivery_flowsheet in patient_assessments per hospital.
Once the hospital-specific median SBT-related assessment time (median_sbt_time) is determined, we use a ±2-hour window around it:
Example: If median_sbt_time = 3 AM, then the evaluation window is 1 AM to 5 AM. Looks good.
The patient must meet both Hemodynamic and Respiratory Stability criteria for at least 1 continuous hour within this new time window.

**Hemodynamic Stability (Updated)**
Vasoactive Medication Limitation: The patient must not be on more than one vasoactive drug at any given time.
Allowed vasoactive drugs: "dopamine", "dobutamine", "epinephrine", "norepinephrine", "phenylephrine", "vasopressin", "milrinone", "angiotensin".
This is determined from the medication_admin_continuous table (med_group = "vasoactives").

**Respiratory Stability (Updated)**
FiO2: ≤ 50% (fio2_set in respiratory_support).
Positive End-Expiratory Pressure (PEEP): ≤ 8 cmH2O (peep_set in respiratory_support).
Pulse Oximetry (SpO2): ≥ 88% (vital_category = "spo2" in vitals).

3. Exclude Patients with Active Tracheostomy (Updated Criteria)
Patients remain eligible for an SBT until they receive a tracheostomy (tracheostomy = 1 in respiratory_support). What about patients admitted with a tracheostomy, i.e., day 0?
Once a patient has a tracheostomy, they are permanently excluded from the cohort from that day forward.
Example: If a patient has tracheostomy = 0 from Day -17 to Day 8, they remain in the cohort.
If tracheostomy = 1 on Day 9, they are dropped from the cohort starting Day 9 forward, and ventilation data is no longer considered.
 
**SBT Delivery Criteria**
If a patient is eligible, an SBT is considered delivered if any one of the following conditions is met:
4. SBT via EHR (sbt_ehr_delivered)
The ventilator mode switches from a controlled mode to one of the following: Looks good
Pressure Support (PS) or Continuous Positive Airway Pressure (CPAP) with:
peep_set ≤ 8
pressure_support_set ≤ 8
OR a T-piece trial, where the ventilator is disconnected, and the patient breathes spontaneously on a T-piece connected to an oxygen source.
The new mode must be maintained for at least 30 minutes (mode_category = "Pressure Support/CPAP" in respiratory_support for ≥ 30 min).

Code 2 min and 30 min version of this.

5. SBT via Flowsheet (sbt_flowsheet_delivered)
The flowsheet field sbt_delivery_pass_fail is not missing (assessment_category = "sbt_screen_pass_fail" in patient_assessments).

**SBT Success/Failure Criteria**
Success
Defined as transition from "Pressure Support/CPAP" (mode_category = "Pressure Support/CPAP") to any other oxygen delivery device.
device_category transitions to **"Nasal Cannula", "Face Mask", "High Flow NC", "Trach Collar", "Room Air"`, etc.
Failure
Defined as return to any mechanical ventilation mode (mode_category = "Assist Control-Volume Control", "Pressure Control", "SIMV", etc.) after SBT_2min or SBT_30min.
 
Timing Assessment Hierarchy for Time to Check for SBT eligible as noted above
Primary Reference: Median recorded_dttm from patient_assessments where assessment_category = 'sat_screen' OR assessment_category = 'sat_delivery_flowsheet', determined per hospital.
Secondary Reference: Site median time if hospital-specific median is unavailable.
Default Window: If median_sbt_time is not available, default evaluation window remains 5:00 - 8:00 AM.
 
Logical Flow with AND/OR Statements
SBT Eligibility
Controlled mode for 6 hours (10 PM to 8 AM):
mode_category = "Controlled" AND cumulative duration ≥ 6 hours AND location_category = "ICU".
Hemodynamic Stability:
Only one active vasoactive drug at any given time.
Respiratory Stability:
FiO2 ≤ 50% AND
PEEP ≤ 8 cmH2O AND
SpO2 ≥ 88%.
SBT Pre-Qualification (Max 2-Min Check):
PEEP ≤ 8 cmH2O AND PS – Set PEEP ≤ 8 cmH2O
Stability Evaluation Time Window:
Median of sat_screen OR sat_delivery_flowsheet ±2 hours.
Example: If median_sbt_time = 3 AM, check criteria between 1 AM and 5 AM.
Exclude Tracheostomy:
tracheostomy = 0 until the first day it becomes 1.
After tracheostomy = 1, exclude patient permanently from cohort.