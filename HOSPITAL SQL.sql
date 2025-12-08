USE Hospital;

#QUERY1
SELECT  

    APR_DRG_Description, 

    COUNT(*) AS total_cases, 

    AVG(total_charges_num) AS average_charges 

FROM 

     discharge_data 

GROUP BY  

    APR_DRG_Description 

ORDER BY  

    average_charges DESC 

LIMIT 10; 


#QUERY2
SELECT 
    facility_name, 
    COUNT(*) AS total_discharges, 
    SUM(CASE WHEN CAST(REPLACE(REPLACE(total_costs, '$', ''), ',', '') AS decimal) > CAST(REPLACE(REPLACE(total_charges, '$', ''), ',', '') AS decimal) THEN 1 ELSE 0 END) AS loss_count, 
    (SUM(CASE WHEN CAST(REPLACE(REPLACE(total_costs, '$', ''), ',', '') AS decimal) > CAST(REPLACE(REPLACE(total_charges, '$', ''), ',', '') AS decimal) THEN 1 ELSE 0 END) * 100.0) / COUNT(*) AS percent_loss_discharges 
FROM 
    discharge_data 
GROUP BY 
    facility_name 
HAVING 
    COUNT(*) > 500  
ORDER BY 
    percent_loss_discharges DESC 
LIMIT 5; 

#QUERY3
 SELECT 
    HOSPITAL_county, 
    COUNT(*) AS total_discharges, 
    # Total Aggregate Loss (summing only the loss amounts) 
    SUM( 
        CASE 
            WHEN CAST(REPLACE(REPLACE(total_costs, '$', ''), ',', '') AS DECIMAL) > CAST(REPLACE(REPLACE(total_charges, '$', ''), ',', '') AS DECIMAL) 
            THEN (CAST(REPLACE(REPLACE(total_costs, '$', ''), ',', '') AS DECIMAL) - CAST(REPLACE(REPLACE(total_charges, '$', ''), ',', '') AS DECIMAL)) 
            ELSE 0  
        END 
    ) AS total_aggregate_financial_loss 
FROM 
    discharge_data
    GROUP BY 
    HOSPITAL_county 
ORDER BY 
    total_aggregate_financial_loss DESC; 
    
    
    #QUERY4
    SELECT 
    age_group, 
    gender, 
    COUNT(*) AS total_discharges, 
    # Total Aggregate Loss (summing only the loss amounts) 
    SUM( 
        CASE 
            WHEN CAST(REPLACE(REPLACE(total_costs, '$', ''), ',', '') AS DECIMAL) > CAST(REPLACE(REPLACE(total_charges, '$', ''), ',', '') AS DECIMAL) 
            THEN (CAST(REPLACE(REPLACE(total_costs, '$', ''), ',', '') AS DECIMAL) - CAST(REPLACE(REPLACE(total_charges, '$', ''), ',', '') AS DECIMAL)) 
            ELSE 0  
        END 
    ) AS total_aggregate_financial_loss 
FROM 
    discharge_DATA 
GROUP BY 
    age_group,
    gender 
ORDER BY 
    total_aggregate_financial_loss DESC; 
    
    
    #QUERY5
    SELECT 
    APR_DRG_DESCRIPTION, 
    COUNT(*) AS total_cases,
    AVG( 
        CAST(REPLACE(REPLACE(total_costs, '$', ''), ',', '') AS DECIMAL) - 
        CAST(REPLACE(REPLACE(total_charges, '$', ''), ',', '') AS DECIMAL) 
    ) AS average_financial_loss_severe 
FROM 
   discharge_data 
WHERE 
    apr_severity_OF_ILLNESS_code = '4' 
GROUP BY 
  APR_DRG_DESCRIPTION 
HAVING 
    AVG(CAST(REPLACE(REPLACE(total_costs, '$', ''), ',', '') AS DECIMAL) - CAST(REPLACE(REPLACE(total_charges, '$', ''), ',', '') AS DECIMAL)) > 0 
ORDER BY 
    average_financial_loss_severe DESC 
LIMIT 10; 


#QUERY6
SELECT 
    facility_name, 
    COUNT(*) AS total_discharges, 
#Total Aggregate Financial Loss (interpreting this as 'Opportunity_Cost_Lost') 
    SUM( 
        CASE 
# loss only when Cost > Charge 
            WHEN CAST(REPLACE(REPLACE(total_costs, '$', ''), ',', '') AS DECIMAL) > CAST(REPLACE(REPLACE(total_charges, '$', ''), ',', '') AS DECIMAL) 
            THEN (CAST(REPLACE(REPLACE(total_costs, '$', ''), ',', '') AS DECIMAL) - CAST(REPLACE(REPLACE(total_charges, '$', ''), ',', '') AS DECIMAL)) 
            ELSE 0  
        END 
    ) AS total_opportunity_cost_lost 
FROM 
    discharge_data 
GROUP BY 
    facility_name 
ORDER BY 
    total_opportunity_cost_lost DESC; 
    
    
    #QUERY7
    SELECT 
    type_of_admission, 
    COUNT(*) AS total_discharges, 
    SUM(CASE WHEN Bed_Blocker_Flag = 'True' THEN 1 ELSE 0 END) AS total_bed_blockers, 
# percentage of discharges that are flagged as 'Bed Blockers' within that admission type 
    (CAST(SUM(CASE WHEN Bed_Blocker_Flag = 'True' THEN 1 ELSE 0 END) AS decimal) * 100.0) / COUNT(*) AS percentage_bed_blockers 
FROM 
discharge_data 
WHERE 
    type_of_admission IN ('Emergency', 'Elective') 
    AND type_of_admission IS NOT NULL 
    AND type_of_admission <> '' 
    AND Bed_Blocker_Flag IS NOT NULL 
    AND Bed_Blocker_Flag <> '' 
GROUP BY 
    type_of_admission 
ORDER BY 
    percentage_bed_blockers DESC; 
    
    
    #QUERY8
    SELECT 
    apr_drg_description, 
    COUNT(*) AS total_cases, 
# Average Actual LOS and Average Predicted LOS 
    AVG(CAST(length_of_stay AS decimal)) AS average_actual_los, 
    AVG(CAST(predicted_los AS decimal)) AS average_predicted_los, 
    #Average Excess Days (Actual LOS - Predicted LOS) 
    (AVG(CAST(length_of_stay AS decimal)) - AVG(CAST(predicted_los AS decimal))) AS average_excess_days 
FROM 
   discharge_data 
WHERE 
    length_of_stay IS NOT NULL  
    AND length_of_stay <> '' 
    AND predicted_los IS NOT NULL 
    AND predicted_los <> '' 
    AND CAST(length_of_stay AS decimal) > 0 
    AND CAST(predicted_los AS decimal) > 0 
GROUP BY 
    apr_drg_description 
#show diagnoses where the Actual LOS was, on average, longer than the Predicted LOS 
HAVING 
    (AVG(CAST(length_of_stay AS decimal)) - AVG(CAST(predicted_los AS decimal))) > 0 
ORDER BY 
    average_excess_days DESC 
LIMIT 10; 


#QUERY9
SELECT 
    apr_severity_of_illness_description, 
    COUNT(*) AS total_discharges, 
# Average Actual Length of Stay (LOS) 
    AVG(CAST(length_of_stay AS decimal)) AS average_actual_los, 
# Average Predicted Length of Stay (Predicted LOS) 
    AVG(CAST(predicted_los AS decimal)) AS average_predicted_los, 
# Difference between actual and predicted LOS 
    (AVG(CAST(length_of_stay AS decimal)) - AVG(CAST(predicted_los AS decimal))) AS average_los_difference 
FROM 
    discharge_data 
WHERE 
    # grouping column itself is not null 
    apr_severity_of_illness_description IS NOT NULL 
    AND apr_severity_of_illness_description <> '' 
# non-null values for the fields being cast and averaged 
    AND length_of_stay IS NOT NULL  
    AND length_of_stay <> '' 
    AND predicted_los IS NOT NULL 
    AND predicted_los <> '' 
    # only valid, positive lengths of stay are included 
    AND CAST(length_of_stay AS decimal) > 0 
    AND CAST(predicted_los AS decimal) > 0 
GROUP BY 
    apr_severity_of_illness_description 
ORDER BY 
    average_los_difference DESC;