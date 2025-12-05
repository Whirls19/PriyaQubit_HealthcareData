# 🏥 Hospital Bed Utilization & Revenue Optimization System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Tableau](https://img.shields.io/badge/Tableau-Dashboard-orange)
![Machine Learning](https://img.shields.io/badge/ML-Gradient%20Boosting-green)

> **A Full-Stack Data Science Project optimizing hospital throughput and financial efficiency using Machine Learning and Business Intelligence.**

---

## 📋 Table of Contents
- [Executive Summary](#-executive-summary)
- [Data Source](#-data-source)
- [Project Architecture](#-project-architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Team](#-team)

---

## 🚀 Executive Summary
Hospital "Bed Blocking"—patients staying longer than medically necessary—causes severe bottlenecks in patient flow and significant revenue leakage. 

This project analyzes patient discharge records to identify operational inefficiencies. We developed an end-to-end solution combining a **Gradient Boosting Machine Learning Model** to predict the "Medically Necessary Length of Stay" (LOS) with an interactive **Command Center Dashboard** to visualize the financial impact of excess stays.

**Key Goals:**
* Predict patient discharge dates with high precision.
* Quantify the "Opportunity Cost" of occupied beds.
* Identify specific diagnoses and admission types driving the backlog.

---

## 💾 Data Source
This project utilizes the **Hospital Inpatient Discharges (SPARCS)** dataset, provided by the **New York State Department of Health**.

* **Dataset Name:** Statewide Planning and Research Cooperative System (SPARCS) De-Identified Data.
* **Description:** Contains patient-level detail on patient characteristics, diagnoses, treatments, services, and charges for every hospital discharge in New York State.
* **Access:** The raw data is publicly available at [Health.data.ny.gov](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/u4ud-w55t).

> **Note:** Due to file size limits (>1GB), the raw data is not included in this repository. A processed sample (`hospital_data_30pct.csv`) is provided for demonstration purposes.

---

## 🏗 Project Architecture

The solution follows a linear data pipeline:

1.  **Data Ingestion:** Loading raw regulatory data containing patient demographics, APR-DRG diagnoses, and financial metrics.
2.  **ETL & Engineering:** Cleaning data, handling outliers, and creating features like `Severity_Risk_Interaction` and `Group_Statistics`.
3.  **Machine Learning:** Training a **Gradient Boosting Regressor** to predict the expected Length of Stay.
4.  **Logic Layer:** Calculating `Variance` (Actual - Predicted) and flagging `Bed_Blockers` based on statistical thresholds.
5.  **Deployment:**
    * **Tableau:** For strategic executive reporting and root cause analysis.
    * **Streamlit:** For real-time clinical prediction and scenario planning.

---

## ✨ Key Features

### 1. AI-Powered Discharge Predictor
A generic ML model that estimates the target discharge date for incoming patients.
* **Input:** Age, Diagnosis, Severity, Admission Type.
* **Output:** Predicted Stay & Risk Alert (Low/Medium/High).
* **Safety Layer:** Includes clinical overrides for "Extreme" severity cases to ensure patient safety.

### 2. The "Command Center" Dashboard (Tableau)
An interactive dashboard answering three critical business questions:
* **Executive View:** Visualizes total revenue opportunity and wasted bed days.
* **Clinical View:** Drills down into specific diagnoses (e.g., Septicemia, Heart Failure) causing delays.
* **Technical View:** Validates model performance by comparing Predicted vs. Actual stays.

### 3. "What-If" Simulation Engine
A dynamic parameter tool allowing stakeholders to calculate ROI by adjusting efficiency targets (e.g., *"What if we reduce variance by 10%?"*).

---

## 🛠 Tech Stack

* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Gradient Boosting Regressor, LabelEncoder)
* **Visualization:** Tableau Public (Embedded), Streamlit
* **Database:** SQL (Data storage & Audit queries)

---

## ⚙ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/hospital-optimization.git](https://github.com/your-username/hospital-optimization.git)
    cd hospital-optimization
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

*This project was developed as part of the Data Science Internship (2025).*
