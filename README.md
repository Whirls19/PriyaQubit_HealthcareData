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
<img width="1920" height="924" alt="{D9E55A89-1241-4521-8DC5-A2118B63225A}" src="https://github.com/user-attachments/assets/4ce9c2c9-4378-4005-bf9e-4fd73d4fe782" />


### 2. The "Command Center" Dashboard (Tableau)
An interactive dashboard answering three critical business questions:
* **Executive View:** Visualizes total revenue opportunity and wasted bed days.
* **Clinical View:** Drills down into specific diagnoses (e.g., Septicemia, Heart Failure) causing delays.
* **Technical View:** Validates model performance by comparing Predicted vs. Actual stays.
  
<img width="1358" height="767" alt="{3A951AA2-3AE4-4F68-B021-8233E0E9951A}" src="https://github.com/user-attachments/assets/8c5af10b-d412-4de9-8280-c8881cb9cf54" />
<img width="1362" height="763" alt="{C68FDFB6-6711-41E1-9724-28961AF8A0E7}" src="https://github.com/user-attachments/assets/7280bea5-e840-4398-a58e-5eb43d2acd3a" />
<img width="1370" height="770" alt="{3A929768-3070-4234-8BB8-043B3E3E5A4A}" src="https://github.com/user-attachments/assets/6657097b-dc53-44ac-99cd-5c17801c3e71" />


### 3. What-If Simulation Engine
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
    streamlit run stream_db.py
    ```

*This project was developed as part of the Data Science Internship (2025).*
