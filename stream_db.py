import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Hospital LOS Analytics", layout="wide")

def create_advanced_features(df):
    df_enhanced = df.copy()
    
    # 1. Age Group Features
    age_mapping = {
        '0 to 17': 8.5,
        '18 to 29': 23.5,
        '30 to 49': 39.5,
        '50 to 69': 59.5,
        '70 or Older': 77.5
    }
    df_enhanced['Age_Numeric'] = df_enhanced['Age Group'].map(age_mapping)
    df_enhanced['Is_Elderly'] = (df_enhanced['Age Group'] == '70 or Older').astype(int)
    df_enhanced['Is_Pediatric'] = (df_enhanced['Age Group'] == '0 to 17').astype(int)
    
    # 2. Admission Type Features
    df_enhanced['Is_Emergency'] = (df_enhanced['Type of Admission'] == 'Emergency').astype(int)
    df_enhanced['Is_Elective'] = (df_enhanced['Type of Admission'] == 'Elective').astype(int)
    df_enhanced['Is_Urgent'] = (df_enhanced['Type of Admission'] == 'Urgent').astype(int)
    
    # 3. Severity and Risk Features
    df_enhanced['Severity_Risk_Interaction'] = (
        df_enhanced['APR Severity of Illness Code'].fillna(0) * 
        df_enhanced['APR Risk of Mortality'].map({'Minor': 1, 'Moderate': 2, 'Major': 3, 'Extreme': 4}).fillna(1)
    )
    df_enhanced['Is_High_Severity'] = (df_enhanced['APR Severity of Illness Code'] >= 3).astype(int)
    df_enhanced['Is_High_Risk'] = (df_enhanced['APR Risk of Mortality'].isin(['Major', 'Extreme'])).astype(int)
    
    # 4. Medical vs Surgical
    df_enhanced['Is_Surgical'] = (df_enhanced['APR Medical Surgical Description'] == 'Surgical').astype(int)
    df_enhanced['Had_Procedure'] = (df_enhanced['CCS Procedure Code'] != 0).astype(int)
    
    # 5. Emergency Department Flag
    df_enhanced['Came_Through_ED'] = (df_enhanced['Emergency Department Indicator'] == 'Y').astype(int)
    df_enhanced['Emergency_And_ED'] = df_enhanced['Is_Emergency'] * df_enhanced['Came_Through_ED']
    
    # 6. Disposition Features
    df_enhanced['Discharged_Home'] = (df_enhanced['Patient Disposition'] == 'Home or Self Care').astype(int)
    df_enhanced['Discharged_Facility'] = (df_enhanced['Patient Disposition'].isin([
        'Skilled Nursing Home', 'Inpatient Rehabilitation Facility', 'Home w/ Home Health Services'
    ])).astype(int)
    
    # 7. DRG and MDC Group Statistics (powerful features!)
    drg_stats = df_enhanced.groupby('APR DRG Code')['Length of Stay'].agg(['mean', 'median', 'std']).reset_index()
    drg_stats.columns = ['APR DRG Code', 'DRG_Avg_LOS', 'DRG_Median_LOS', 'DRG_Std_LOS']
    df_enhanced = df_enhanced.merge(drg_stats, on='APR DRG Code', how='left')
    
    mdc_stats = df_enhanced.groupby('APR MDC Code')['Length of Stay'].agg(['mean', 'median']).reset_index()
    mdc_stats.columns = ['APR MDC Code', 'MDC_Avg_LOS', 'MDC_Median_LOS']
    df_enhanced = df_enhanced.merge(mdc_stats, on='APR MDC Code', how='left')
    
    # 8. Diagnosis Category Features
    ccs_stats = df_enhanced.groupby('CCS Diagnosis Code')['Length of Stay'].agg(['mean', 'median']).reset_index()
    ccs_stats.columns = ['CCS Diagnosis Code', 'Diagnosis_Avg_LOS', 'Diagnosis_Median_LOS']
    df_enhanced = df_enhanced.merge(ccs_stats, on='CCS Diagnosis Code', how='left')
    
    # 9. Facility-level Features
    facility_stats = df_enhanced.groupby('Facility Name')['Length of Stay'].agg(['mean', 'median']).reset_index()
    facility_stats.columns = ['Facility Name', 'Facility_Avg_LOS', 'Facility_Median_LOS']
    df_enhanced = df_enhanced.merge(facility_stats, on='Facility Name', how='left')
    
    # 10. Cost-based Features
    df_enhanced['Cost_Per_Day_Historic'] = df_enhanced['Total Costs'] / df_enhanced['Length of Stay']
    df_enhanced['Cost_Per_Day_Historic'] = df_enhanced['Cost_Per_Day_Historic'].replace([np.inf, -np.inf], np.nan)
    df_enhanced['Charge_to_Cost_Ratio'] = df_enhanced['Total Charges'] / (df_enhanced['Total Costs'] + 1)
    df_enhanced['Log_Total_Charges'] = np.log1p(df_enhanced['Total Charges'].fillna(0))
    
    return df_enhanced

def encode_categorical_features(df):
    df_encoded = df.copy()
    label_encoders = {}
    
    categorical_features = [
        'Age Group',
        'Gender',
        'Type of Admission',
        'APR DRG Description',
        'APR Severity of Illness Description',
        'APR Risk of Mortality',
        'APR Medical Surgical Description',
        'CCS Diagnosis Description',
        'Patient Disposition',
        'Health Service Area'
    ]
    
    for feature in categorical_features:
        if feature in df.columns:
            le = LabelEncoder()
            df_encoded[f'{feature}_Encoded'] = le.fit_transform(
                df_encoded[feature].fillna('Unknown')
            )
            label_encoders[feature] = le
    
    return df_encoded, label_encoders

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('Hospital_Inpatient_Discharges_(SPARCS_De-Identified)__2012_20251204.csv')
    
    # Clean financial columns
    df['Total Charges'] = df['Total Charges'].str.replace('$', '', regex=False)
    df['Total Charges'] = df['Total Charges'].str.replace(',', '', regex=False)
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    
    df['Total Costs'] = df['Total Costs'].str.replace('$', '', regex=False)
    df['Total Costs'] = df['Total Costs'].str.replace(',', '', regex=False)
    df['Total Costs'] = pd.to_numeric(df['Total Costs'], errors='coerce')
    
    df['Length of Stay'] = pd.to_numeric(df['Length of Stay'], errors='coerce')
    
    # Sample 30%
    df_sample = df.sample(frac=0.3, random_state=42)
    return df_sample

@st.cache_resource
def train_enhanced_model():
    with st.spinner("Training model..."):
        df = load_and_clean_data()
        
        df = create_advanced_features(df)
        df, label_encoders = encode_categorical_features(df)
        
        # Keep ALL features including cost features for training
        feature_columns = [
            'Age Group_Encoded',
            'Gender_Encoded',
            'Type of Admission_Encoded',
            'APR DRG Description_Encoded',
            'APR Severity of Illness Description_Encoded',
            'APR Risk of Mortality_Encoded',
            'APR Medical Surgical Description_Encoded',
            'CCS Diagnosis Description_Encoded',
            'Patient Disposition_Encoded',
            'Health Service Area_Encoded',
            'Age_Numeric',
            'APR Severity of Illness Code',
            'APR DRG Code',
            'APR MDC Code',
            'CCS Diagnosis Code',
            'CCS Procedure Code',
            'Is_Elderly',
            'Is_Pediatric',
            'Is_Emergency',
            'Is_Elective',
            'Is_Urgent',
            'Is_Surgical',
            'Had_Procedure',
            'Came_Through_ED',
            'Emergency_And_ED',
            'Discharged_Home',
            'Discharged_Facility',
            'Is_High_Severity',
            'Is_High_Risk',
            'Severity_Risk_Interaction',
            'DRG_Avg_LOS',
            'DRG_Median_LOS',
            'DRG_Std_LOS',
            'MDC_Avg_LOS',
            'MDC_Median_LOS',
            'Diagnosis_Avg_LOS',
            'Diagnosis_Median_LOS',
            'Facility_Avg_LOS',
            'Facility_Median_LOS',
            'Log_Total_Charges',
            'Charge_to_Cost_Ratio'
        ]
        
        available_features = [f for f in feature_columns if f in df.columns]
        
        # Prepare data
        df_model = df.dropna(subset=['Length of Stay']).copy()
        for col in available_features:
            if df_model[col].isna().any():
                df_model[col] = df_model[col].fillna(df_model[col].median())
        
        X = df_model[available_features]
        y = df_model['Length of Stay']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Store actual record count
        training_records = len(X_train)
        
        # Original model settings
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            max_features='sqrt',
            random_state=42,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate DRG-specific charges/costs for better prediction
        drg_cost_stats = df.groupby('APR DRG Code').agg({
            'Total Charges': 'median',
            'Total Costs': 'median'
        }).reset_index()
        drg_cost_stats.columns = ['APR DRG Code', 'DRG_Median_Charges', 'DRG_Median_Costs']
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': available_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Get top diagnoses and lookup
        top_diagnoses = df['APR DRG Description'].value_counts().head(30).index.tolist()
        
        drg_lookup = df[['APR DRG Description', 'APR DRG Code', 'APR MDC Code', 'CCS Diagnosis Code',
                         'DRG_Avg_LOS', 'DRG_Median_LOS', 'DRG_Std_LOS',
                         'MDC_Avg_LOS', 'MDC_Median_LOS',
                         'Diagnosis_Avg_LOS', 'Diagnosis_Median_LOS',
                         'Facility_Avg_LOS', 'Facility_Median_LOS']].drop_duplicates()
        
        # Merge cost stats into lookup
        drg_lookup = drg_lookup.merge(drg_cost_stats, on='APR DRG Code', how='left')
        
        health_areas = df['Health Service Area'].dropna().unique().tolist()
        
        return model, label_encoders, available_features, r2, mae, top_diagnoses, drg_lookup, health_areas, feature_importance, training_records

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Dashboard", "Predictor"])

if page == "Dashboard":
    st.title("Hospital Dashboard")
    view = st.selectbox("Select Dashboard", ["Executive Summary", "Clinical Operations", "AI Model Validation"])
    
    urls = {
        "Executive Summary": "https://public.tableau.com/views/Healthcare__17648521079380/FinancialImpactDB?:showVizHome=no&:embed=true",
        "Clinical Operations": "https://public.tableau.com/views/Healthcare__17648521079380/ClincalOperationDB?:showVizHome=no&:embed=true",
        "AI Model Validation": "https://public.tableau.com/views/Healthcare__17648521079380/ModelValidation?:showVizHome=no&:embed=true"
    }
    
    components.html(f'<iframe src="{urls[view]}" width="100%" height="850"></iframe>', height=850)

else:
    st.title("Discharge Predictor")
    
    model, encoders, features, r2, mae, top_dx, lookup, areas, feat_imp, num_records = train_enhanced_model()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.2f} days")
    c3.metric("Training Records", f"{num_records:,}")
    
    with st.expander("View Top Features"):
        st.dataframe(feat_imp.head(10), use_container_width=True)
    
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.selectbox("Age Group", ['0 to 17', '18 to 29', '30 to 49', '50 to 69', '70 or Older'])
        gender = st.radio("Gender", ['M', 'F'])
        adm_type = st.selectbox("Admission Type", ['Elective', 'Emergency', 'Urgent', 'Newborn', 'Not Available', 'Trauma'])
        area = st.selectbox("Health Service Area", areas)
    
    with col2:
        sev = st.selectbox("APR Severity", ['Minor', 'Moderate', 'Major', 'Extreme'])
        risk = st.selectbox("APR Risk", ['Minor', 'Moderate', 'Major', 'Extreme'])
        proc = st.selectbox("Medical/Surgical", ['Medical', 'Surgical'])
        dx = st.selectbox("Diagnosis", top_dx)
    
    adm_date = st.date_input("Admission Date")
    
    if st.button("Predict Discharge Date"):
        try:
            # Get DRG info
            info = lookup[lookup['APR DRG Description'] == dx].iloc[0]
            
            # Build input exactly like training
            inp = pd.DataFrame({
                'Age Group': [age],
                'Gender': [gender],
                'Type of Admission': [adm_type],
                'APR DRG Description': [dx],
                'APR Severity of Illness Description': [sev],
                'APR Risk of Mortality': [risk],
                'APR Medical Surgical Description': [proc],
                'Health Service Area': [area],
                'CCS Diagnosis Description': ['Unknown'],
                'Patient Disposition': ['Home or Self Care']
            })
            
            # Apply same feature engineering
            age_map = {'0 to 17': 8.5, '18 to 29': 23.5, '30 to 49': 39.5, '50 to 69': 59.5, '70 or Older': 77.5}
            inp['Age_Numeric'] = age_map[age]
            inp['Is_Elderly'] = int(age == '70 or Older')
            inp['Is_Pediatric'] = int(age == '0 to 17')
            inp['Is_Emergency'] = int(adm_type == 'Emergency')
            inp['Is_Elective'] = int(adm_type == 'Elective')
            inp['Is_Urgent'] = int(adm_type == 'Urgent')
            inp['Is_Surgical'] = int(proc == 'Surgical')
            inp['Had_Procedure'] = 0
            inp['Came_Through_ED'] = int(adm_type == 'Emergency')
            inp['Emergency_And_ED'] = inp['Is_Emergency'] * inp['Came_Through_ED']
            inp['Discharged_Home'] = 1
            inp['Discharged_Facility'] = 0
            
            sev_map = {'Minor': 1, 'Moderate': 2, 'Major': 3, 'Extreme': 4}
            inp['APR Severity of Illness Code'] = sev_map[sev]
            inp['Is_High_Severity'] = int(sev_map[sev] >= 3)
            inp['Is_High_Risk'] = int(risk in ['Major', 'Extreme'])
            inp['Severity_Risk_Interaction'] = sev_map[sev] * sev_map[risk]
            
            inp['APR DRG Code'] = info['APR DRG Code']
            inp['APR MDC Code'] = info['APR MDC Code']
            inp['CCS Diagnosis Code'] = info['CCS Diagnosis Code']
            inp['CCS Procedure Code'] = 0
            inp['DRG_Avg_LOS'] = info['DRG_Avg_LOS']
            inp['DRG_Median_LOS'] = info['DRG_Median_LOS']
            inp['DRG_Std_LOS'] = info['DRG_Std_LOS']
            inp['MDC_Avg_LOS'] = info['MDC_Avg_LOS']
            inp['MDC_Median_LOS'] = info['MDC_Median_LOS']
            inp['Diagnosis_Avg_LOS'] = info['Diagnosis_Avg_LOS']
            inp['Diagnosis_Median_LOS'] = info['Diagnosis_Median_LOS']
            inp['Facility_Avg_LOS'] = info['Facility_Avg_LOS']
            inp['Facility_Median_LOS'] = info['Facility_Median_LOS']
            
            # Use DRG-specific cost values (much better than global median)
            drg_charges = info.get('DRG_Median_Charges', 25000)
            drg_costs = info.get('DRG_Median_Costs', 10000)
            inp['Log_Total_Charges'] = np.log1p(drg_charges)
            inp['Charge_to_Cost_Ratio'] = drg_charges / (drg_costs + 1)
            
            # Encode
            for col, enc in encoders.items():
                if col in inp.columns:
                    try:
                        inp[col + '_Encoded'] = enc.transform(inp[col])
                    except:
                        inp[col + '_Encoded'] = 0
            
            # Predict
            pred = model.predict(inp[features])[0]
            
            # Apply intelligent adjustments based on clinical factors
            adjustment = 0
            
            # Severity adjustments
            if sev == 'Extreme':
                adjustment += 1.5
            elif sev == 'Major':
                adjustment += 0.8
            
            # Risk adjustments
            if risk == 'Extreme':
                adjustment += 1.2
            elif risk == 'Major':
                adjustment += 0.6
            
            # Emergency admissions tend to stay longer
            if adm_type == 'Emergency':
                adjustment += 0.3
            
            # Elderly patients often stay longer
            if age == '70 or Older':
                adjustment += 0.4
            
            # Pediatric cases are often shorter
            if age == '0 to 17':
                adjustment -= 0.3
            
            # Surgical cases tend to need recovery time
            if proc == 'Surgical':
                adjustment += 0.5
            
            # Apply adjustment
            pred = pred + adjustment
            pred = max(1, round(pred, 1))
            disc = adm_date + timedelta(days=int(pred))
            
            st.success("Prediction Complete!")
            st.markdown(f"## Predicted LOS: {pred} Days")
            st.markdown(f"### Expected Discharge: {disc.strftime('%B %d, %Y')}")
            
            # Show what influenced the prediction
            if adjustment != 0:
                st.info(f"Clinical adjustment applied: {adjustment:+.1f} days based on patient risk factors")
            
            c1, c2 = st.columns(2)
            c1.info(f"Model Accuracy: {r2:.1%}")
            c1.info(f"Typical Error: ±{mae:.1f} days")
            c1.info(f"Prediction Range: {max(1, pred-mae):.1f} - {pred+mae:.1f} days")
            c2.info(f"Similar Patients (DRG) Avg: {info['DRG_Avg_LOS']:.1f} days")
            c2.info(f"Diagnosis Group Median: {info['Diagnosis_Median_LOS']:.1f} days")
            c2.info(f"Historical Variability: ±{info['DRG_Std_LOS']:.1f} days")
            
            # Risk factors
            st.markdown("---")
            st.subheader("Clinical Context")
            
            # Show key factors affecting prediction
            factors = []
            if sev in ['Major', 'Extreme']:
                factors.append(f"High severity ({sev}) increases expected stay")
            if risk in ['Major', 'Extreme']:
                factors.append(f"Elevated mortality risk ({risk}) requires extended monitoring")
            if adm_type == 'Emergency':
                factors.append("Emergency admission typically requires stabilization period")
            if age == '70 or Older':
                factors.append("Elderly patient may need additional recovery time")
            if proc == 'Surgical':
                factors.append("Surgical procedure requires post-operative recovery")
            
            if factors:
                for factor in factors:
                    st.warning(factor)
            else:
                st.success("Standard risk profile - routine care expected")
            
            # Compare to baseline
            variance = abs(pred - info['DRG_Avg_LOS'])
            if variance > 2:
                if pred > info['DRG_Avg_LOS']:
                    st.warning(f"Predicted stay is {variance:.1f} days LONGER than typical cases due to risk factors")
                else:
                    st.success(f"Predicted stay is {variance:.1f} days SHORTER than typical cases")
            else:
                st.info("Predicted stay is consistent with similar cases")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

