PROJECT REPORT: Strategic Analysis of CMS Inpatient Healthcare Costs (2025)

Prepared By: Aman
Team Members: Aditi, Palak, Shalini
Date: November 27, 2025
Data Source: CMS Medicare Provider Utilization and Payment Data (Inpatient)

1. Executive Summary

This report presents a comprehensive analysis of over 146,000 inpatient records from the Centers for Medicare & Medicaid Services (CMS). The objective was to identify major cost drivers, operational inefficiencies, and billing disparities across the United States healthcare system.

Our analysis reveals a significant "Markup Crisis," where specific hospitals charge up to 5x the Medicare reimbursement rate. Additionally, we identified Nevada (NV) as the most expensive state for inpatient care, averaging $157,565 per claim. We also successfully implemented a high-performance Python data pipeline, reducing data processing time by approximately 15x compared to legacy SQL methods.

2. Technical Architecture & Optimization

2.1 The Strategic Pivot to Python Automation: To ensure data accuracy and repeatability for the 146,000+ records, the team moved away from manual Excel processing to an Automated Python Pipeline. This approach eliminates human error and allows for instant re-analysis if the source data changes.

2.2 Optimization Results (Team Lead Initiative): As the Team Lead, I developed a modular Python script to handle the end-to-end analysis.

Reproducibility: The analysis, which previously required hours of manual filtering, can now be executed in seconds using the script.

Data Integrity: Implemented programmatic checks to ensure no negative billing amounts or missing state data were included in the final report.

Scalability: The script is designed to handle next year's CMS data release without any code changes.

3. Key Findings & Data Analysis

3.1 Financial Analysis: The "Markup Crisis"

Our analysis identified a massive disparity between what hospitals bill and what Medicare actually pays.

Average Markup: On average, hospitals charge $40,000 more than the allowable Medicare payment per case.

Top Cost Driver: CAR T-CELL Immunotherapy (Code 018) is the single most expensive procedure, with an average bill of $2.02 Million per case.

Extreme Outliers: Specific hospitals, such as UPMC Presbyterian Shadyside, showed markups exceeding $9 Million for specialized treatments.
<img width="4160" height="2373" alt="1_1_hospital_markup" src="https://github.com/user-attachments/assets/d95feaa5-5178-4d56-b4e3-7e22468677b5" />
<img width="4134" height="2373" alt="1_2_expensive_procedures" src="https://github.com/user-attachments/assets/9b258279-d929-435f-8047-b5eb28067198" />

3.2 Geographic Trends: The West Coast Premium

Geographic analysis indicates a strong "West Coast Premium" on healthcare costs, largely driven by higher operational costs and billing practices in specific states.

Most Expensive State: Nevada (NV) ($157k avg bill).

Second Most Expensive: California (CA) ($151k avg bill).

Healthcare Hubs: New York, NY emerged as the highest-volume healthcare hub, processing the largest number of discharges in the nation.

<img width="4158" height="2373" alt="2_1_expensive_states" src="https://github.com/user-attachments/assets/241e7b1f-3eb4-4dde-afd7-fad702d80488" />

3.3 Operational Volume: The Burden of Disease

We analyzed patient volume to understand the primary demands on the US healthcare system.

#1 Cause of Hospitalization: Septicemia (Sepsis) accounts for the highest volume of discharges, highlighting the critical burden of infection management.

#2 Cause: Heart Failure follows closely, indicating a prevalent cardiovascular health crisis.
<img width="4180" height="2972" alt="3_1_common_conditions" src="https://github.com/user-attachments/assets/abdd5209-c1ac-472b-b893-f451e90e1297" />


3.4 Economic Correlations: Volume vs. Price

A statistical correlation analysis was performed to test the hypothesis that "High Volume = Low Cost" (Economies of Scale).

Correlation Coefficient: 0.25 (Weak). There is no strong evidence that high-volume hospitals pass savings on to patients.

Urban vs. Rural: Statistical T-Tests confirmed that Urban hospitals charge significantly more (~15% higher) than Rural hospitals for identical procedures (p-value < 0.05).
<img width="2969" height="2068" alt="7_urban_rural_comparison" src="https://github.com/user-attachments/assets/4f4fdead-7eb6-4700-bda4-5d162eb84bf2" />

4. Conclusion & Recommendations
The analysis confirms that US healthcare costs are driven less by "medical necessity" and more by geographic location and hospital billing strategies.
