# my_heart_recommendations.py - Complete Professional Heart Recommendations

class HeartRecommendationSystem:
    """
    Professional Heart Disease Recommendation System
    Based on clinical guidelines and evidence-based medicine
    """
    
    def __init__(self):
        self.recommendations = self.setup_recommendations()
    
    def setup_recommendations(self):
        """Complete professional recommendations for all heart disease features"""
        
        recommendations = {
            
            # 1. Age - سن
            'Age': {
                'general': "Age is a non-modifiable risk factor for cardiovascular disease. Risk increases exponentially after age 45 in men and 55 in women due to arterial stiffening, endothelial dysfunction, and accumulated oxidative stress.",
                'age_categories': {
                    '<40': {
                        'risk_level': 'Low',
                        'recommendations': [
                            "Establish baseline cardiovascular risk profile",
                            "Focus on primary prevention and lifestyle optimization",
                            "Annual BP screening, lipid panel every 5 years if normal"
                        ]
                    },
                    '40-54': {
                        'risk_level': 'Moderate',
                        'recommendations': [
                            "Calculate 10-year ASCVD risk score using pooled cohort equations",
                            "Consider coronary calcium scoring if intermediate risk",
                            "Lipid panel every 3-5 years, more frequent if abnormal"
                        ]
                    },
                    '55-64': {
                        'risk_level': 'High',
                        'recommendations': [
                            "Intensive risk factor modification and monitoring",
                            "Consider stress testing if multiple risk factors present",
                            "Annual comprehensive cardiovascular assessment"
                        ]
                    },
                    '≥65': {
                        'risk_level': 'Very High',
                        'recommendations': [
                            "Multidisciplinary approach with geriatric considerations",
                            "Balance aggressive treatment with life expectancy and frailty",
                            "Biannual cardiology follow-up recommended"
                        ]
                    }
                },
                'professional': [
                    "Age-adjusted risk stratification using validated calculators (ASCVD Risk Calculator)",
                    "Consider biological vs chronological age in treatment decisions",
                    "Implement age-appropriate exercise prescriptions (target HR = 220-age)",
                    "Screen for age-related comorbidities affecting cardiovascular health"
                ],
                'medical_referral': "Refer for cardiology consultation if age ≥65 with ≥2 additional risk factors, or any age with concerning symptoms or abnormal screening tests."
            },
            
            # 2. Sex - جنسیت
            'Sex': {
                'general': "Sex significantly influences cardiovascular disease presentation, progression, and outcomes. Men develop CAD 7-10 years earlier than women, but women have higher mortality rates post-MI.",
                'male_specific': [
                    "Higher risk of early-onset CAD (age 45+)",
                    "Classic chest pain presentation more common",
                    "Lower HDL levels typically observed"
                ],
                'female_specific': [
                    "Protective effect of estrogen until menopause",
                    "Atypical symptoms more common (fatigue, nausea, jaw pain)",
                    "Higher risk of microvascular disease and heart failure with preserved EF"
                ],
                'professional': [
                    "Apply sex-specific risk calculators and normal values for diagnostic tests",
                    "Consider hormone replacement therapy effects on cardiovascular risk",
                    "Screen for sex-specific risk factors (PCOS in women, low testosterone in men)",
                    "Implement tailored lifestyle interventions based on sex-specific barriers"
                ],
                'medical_referral': "Women: Consider cardiology referral for atypical symptoms, especially post-menopause. Men: Earlier screening and intervention starting age 40 if additional risk factors present."
            },
            
            # 3. ChestPainType - نوع درد قفسه سینه  
            'ChestPainType': {
                'general': "Chest pain classification is crucial for CAD diagnosis. Typical angina has 95% specificity for obstructive CAD, while atypical presentations require careful evaluation.",
                'pain_types': {
                    'Typical Angina': {
                        'characteristics': 'Substernal chest discomfort, provoked by exertion/stress, relieved by rest/nitroglycerin',
                        'cad_probability': '85-95%',
                        'action': 'High suspicion for obstructive CAD'
                    },
                    'Atypical Angina': {
                        'characteristics': 'Meets 2 of 3 typical angina criteria',
                        'cad_probability': '15-85%',
                        'action': 'Intermediate probability, requires risk stratification'
                    },
                    'Non-anginal': {
                        'characteristics': 'Meets ≤1 typical angina criteria',
                        'cad_probability': '<15%',
                        'action': 'Consider alternative diagnoses'
                    },
                    'Asymptomatic': {
                        'characteristics': 'No chest pain symptoms',
                        'cad_probability': 'Variable based on risk factors',
                        'action': 'Focus on risk factor modification'
                    }
                },
                'professional': [
                    "Use validated chest pain classification systems (Diamond-Forrester criteria)",
                    "Document pain character, location, triggers, duration, and relief factors",
                    "Consider sex and age variations in anginal presentations",
                    "Evaluate for anginal equivalents (dyspnea, fatigue, arm/jaw discomfort)"
                ],
                'medical_referral': "Immediate referral for typical angina or any chest pain with concerning features. Urgent cardiology consultation for crescendo angina or rest pain."
            },
            
            # 4. RestingBP - فشار خون
            'RestingBP': {
                'general': "Hypertension affects 45% of adults and is the leading modifiable risk factor for cardiovascular disease. Every 20 mmHg increase in SBP doubles cardiovascular risk.",
                'bp_categories': {
                    'Normal': '<120/<80 mmHg - Continue healthy lifestyle',
                    'Elevated': '120-129/<80 mmHg - Lifestyle modifications',
                    'Stage 1 HTN': '130-139/80-89 mmHg - Lifestyle + consider medication',
                    'Stage 2 HTN': '≥140/≥90 mmHg - Lifestyle + antihypertensive therapy',
                    'Hypertensive Crisis': '>180/>120 mmHg - Immediate medical attention'
                },
                'professional': [
                    "Use proper BP measurement technique (AHA guidelines): rest 5 min, appropriate cuff size, 2+ readings",
                    "Confirm diagnosis with out-of-office BP monitoring (ABPM or home monitoring)",
                    "Assess for target organ damage (LVH, retinopathy, proteinuria, CKD)",
                    "Calculate cardiovascular risk to guide treatment intensity and targets"
                ],
                'medical_referral': "Refer for BP ≥180/≥120 mmHg, suspected secondary hypertension, resistant hypertension (>3 drugs), or target organ damage."
            },
            
            # 5. Cholesterol - کلسترول
            'Cholesterol': {
                'general': "Elevated cholesterol, particularly LDL-C, directly correlates with atherosclerotic plaque formation. Each 1 mmol/L (39 mg/dL) LDL-C reduction decreases cardiovascular events by ~20%.",
                'lipid_targets': {
                    'Primary Prevention': 'LDL-C <100 mg/dL (2.6 mmol/L), <70 mg/dL if high risk',
                    'Secondary Prevention': 'LDL-C <70 mg/dL (1.8 mmol/L), <55 mg/dL if very high risk',
                    'HDL-C': 'Men >40 mg/dL, Women >50 mg/dL',
                    'Triglycerides': '<150 mg/dL (1.7 mmol/L)',
                    'Non-HDL-C': '<130 mg/dL primary prevention, <100 mg/dL secondary prevention'
                },
                'professional': [
                    "Obtain complete lipid panel after 9-12 hour fast for accurate triglyceride measurement",
                    "Calculate non-HDL cholesterol and apolipoprotein B if triglycerides >200 mg/dL",
                    "Consider advanced lipid testing (LDL particle number, Lp(a)) in high-risk patients",
                    "Implement evidence-based statin therapy according to ACC/AHA guidelines"
                ],
                'medical_referral': "Refer to lipid specialist for familial hypercholesterolemia (LDL-C >190 mg/dL), statin intolerance, or failure to achieve targets despite maximum tolerated therapy."
            },
            
            # 6. FastingBS - قند خون ناشتا
            'FastingBS': {
                'general': "Diabetes mellitus increases cardiovascular risk 2-4 fold through accelerated atherosclerosis, endothelial dysfunction, and increased thrombotic tendency.",
                'glucose_categories': {
                    'Normal': '<100 mg/dL (5.6 mmol/L)',
                    'Prediabetes': '100-125 mg/dL (5.6-6.9 mmol/L)',
                    'Diabetes': '≥126 mg/dL (7.0 mmol/L) on two occasions',
                    'HbA1c Targets': '<7% for most adults, <6.5% if low hypoglycemia risk'
                },
                'professional': [
                    "Screen annually if prediabetes, every 3 years if normal glucose and low risk",
                    "Use HbA1c for diagnosis and monitoring (reflects 2-3 month glucose control)",
                    "Implement intensive cardiovascular risk reduction in diabetic patients",
                    "Consider SGLT2 inhibitors or GLP-1 agonists for cardiovascular protection"
                ],
                'medical_referral': "Refer to endocrinology for newly diagnosed diabetes, HbA1c >9%, frequent hypoglycemia, or cardiovascular complications requiring specialized diabetes management."
            },
            
            # 7. RestingECG - نوار قلب
            'RestingECG': {
                'general': "Resting ECG provides valuable information about cardiac rhythm, conduction, chamber enlargement, and ischemic changes. Normal ECG doesn't exclude CAD but abnormal findings increase cardiovascular risk.",
                'ecg_findings': {
                    'Normal': 'Normal sinus rhythm, rate 60-100 bpm, normal intervals and morphology',
                    'ST-T Abnormalities': 'May indicate ischemia, electrolyte imbalance, or structural heart disease',
                    'Left Ventricular Hypertrophy': 'Suggests chronic pressure overload, associated with increased cardiovascular events',
                    'Q-waves': 'May indicate prior myocardial infarction',
                    'Arrhythmias': 'Atrial fibrillation increases stroke risk 5-fold'
                },
                'professional': [
                    "Obtain baseline ECG in all adults ≥40 years or with cardiovascular risk factors",
                    "Use standardized ECG interpretation criteria (AHA/ACC guidelines)",
                    "Compare with prior ECGs to identify new changes suggesting acute pathology",
                    "Consider advanced ECG techniques (signal-averaged ECG, heart rate variability) in high-risk patients"
                ],
                'medical_referral': "Immediate referral for acute ST-elevation, new Q-waves, or concerning arrhythmias. Routine cardiology referral for LVH, significant conduction abnormalities, or ischemic changes."
            },
            
            # 8. MaxHR - حداکثر ضربان قلب
            'MaxHR': {
                'general': "Maximum heart rate reflects cardiac reserve and exercise capacity. Age-predicted max HR = 220-age, but individual variation is significant. Chronotropic incompetence suggests underlying CAD.",
                'hr_assessment': {
                    'Normal Response': 'Achieves ≥85% age-predicted max HR during exercise',
                    'Chronotropic Incompetence': 'Fails to achieve 85% predicted max HR or has blunted HR response',
                    'Exercise Capacity': 'Max HR correlates with cardiovascular fitness and prognosis',
                    'Recovery HR': 'Failure to decrease ≥12 bpm in first minute suggests autonomic dysfunction'
                },
                'professional': [
                    "Assess chronotropic response during exercise stress testing",
                    "Calculate heart rate reserve (max HR - resting HR) for exercise prescription",
                    "Monitor heart rate recovery as prognostic indicator",
                    "Consider medication effects on heart rate response (beta-blockers, calcium channel blockers)"
                ],
                'medical_referral': "Refer for exercise stress testing if chronotropic incompetence suspected, or for comprehensive evaluation of exercise intolerance with concerning symptoms."
            },
            
            # 9. ExerciseAngina - درد قفسه سینه هنگام ورزش
            'ExerciseAngina': {
                'general': "Exercise-induced angina indicates supply-demand mismatch and suggests functionally significant coronary stenosis. Presence increases likelihood of obstructive CAD to >90%.",
                'angina_assessment': {
                    'Exercise-Induced': 'Chest discomfort occurring during physical exertion',
                    'Reproducible Threshold': 'Symptoms occur at consistent workload/heart rate',
                    'Relief Pattern': 'Symptoms resolve with rest within 2-5 minutes',
                    'Functional Significance': 'Indicates hemodynamically significant stenosis (>70%)'
                },
                'professional': [
                    "Document specific exercise triggers, intensity, and symptom characteristics",
                    "Assess functional capacity using METs (metabolic equivalents) or standardized questionnaires",
                    "Consider exercise stress testing to objectively evaluate exercise-induced ischemia",
                    "Implement appropriate anti-anginal therapy and activity modifications"
                ],
                'medical_referral': "Urgent cardiology referral for exercise-induced chest pain, especially if new onset or worsening pattern. Consider invasive evaluation if high-risk features present."
            },
            
            # 10. Oldpeak - افت ST
            'Oldpeak': {
                'general': "ST depression (Oldpeak) during exercise represents myocardial ischemia. ≥1mm horizontal or downsloping ST depression is considered positive for ischemia and indicates significant CAD.",
                'st_interpretation': {
                    'Normal': '<0.5mm ST depression or elevation',
                    'Borderline': '0.5-1.0mm ST depression',
                    'Positive': '≥1.0mm horizontal or downsloping ST depression',
                    'Highly Positive': '≥2.0mm ST depression or early onset (<6 METs)',
                    'ST Elevation': 'May indicate transmural ischemia or aneurysm'
                },
                'professional': [
                    "Measure ST depression 60-80ms after J-point in appropriate leads",
                    "Consider lead location, timing of onset, and recovery characteristics",
                    "Integrate with clinical symptoms and hemodynamic response",
                    "Account for baseline ECG abnormalities that may affect ST interpretation"
                ],
                'medical_referral': "Immediate referral for ≥2mm ST depression, early positive test (<6 METs), or ST depression with hypotensive response. Consider coronary angiography for high-risk findings."
            },
            
            # 11. ST_Slope - شیب ST
            'ST_Slope': {
                'general': "ST segment slope during exercise provides crucial diagnostic information. Upsloping ST depression is generally benign, while flat or downsloping patterns suggest significant ischemia.",
                'slope_patterns': {
                    'Upsloping': 'Generally considered normal or benign finding',
                    'Flat/Horizontal': 'Abnormal, suggests myocardial ischemia',
                    'Downsloping': 'Most concerning pattern, high specificity for CAD',
                    'Functional Significance': 'Slope pattern affects diagnostic accuracy of stress testing'
                },
                'professional': [
                    "Assess ST slope in conjunction with magnitude of depression and clinical context",
                    "Use computer-assisted ST analysis when available for objective measurement",
                    "Consider additional imaging if ST changes are equivocal or borderline",
                    "Recognize that certain medications and conditions can affect ST morphology"
                ],
                'medical_referral': "Refer for flat or downsloping ST depression ≥1mm, especially if accompanied by symptoms or hemodynamic instability. Consider stress imaging for equivocal results."
            }
        }
        
        return recommendations
    
    def get_age_category_recommendation(self, age):
        """Get age-specific recommendations"""
        age_recs = self.recommendations['Age']['age_categories']
        
        if age < 40:
            return age_recs['<40']
        elif 40 <= age <= 54:
            return age_recs['40-54']
        elif 55 <= age <= 64:
            return age_recs['55-64']
        else:
            return age_recs['≥65']
    
    def get_recommendation(self, feature_name, patient_value=None):
        """Get recommendation for a specific feature"""
        if feature_name in self.recommendations:
            rec = self.recommendations[feature_name]
            result = {
                'feature': feature_name,
                'value': patient_value,
                'general_advice': rec['general'],
                'professional_recommendations': rec['professional'],
                'medical_referral': rec['medical_referral']
            }
            
            # Add age-specific recommendations if applicable
            if feature_name == 'Age' and patient_value is not None:
                result['age_specific'] = self.get_age_category_recommendation(patient_value)
            
            return result
        else:
            return None
    
    def calculate_cardiovascular_risk_score(self, patient_data):
        """Calculate a simplified cardiovascular risk score"""
        score = 0
        risk_factors = []
        
        # Age scoring
        age = patient_data.get('Age', 0)
        if age >= 65:
            score += 4
            risk_factors.append('Advanced age (≥65)')
        elif age >= 55:
            score += 3
            risk_factors.append('Older age (55-64)')
        elif age >= 45:
            score += 2
            risk_factors.append('Middle age (45-54)')
        
        # Gender scoring (assuming 1=male, 0=female)
        if patient_data.get('Sex') == 1 and age >= 45:
            score += 1
            risk_factors.append('Male gender ≥45 years')
        elif patient_data.get('Sex') == 0 and age >= 55:
            score += 1
            risk_factors.append('Female gender ≥55 years')
        
        # Blood pressure scoring
        bp = patient_data.get('RestingBP', 0)
        if bp >= 180:
            score += 4
            risk_factors.append('Stage 2 Hypertension (severe)')
        elif bp >= 140:
            score += 3
            risk_factors.append('Stage 2 Hypertension')
        elif bp >= 130:
            score += 2
            risk_factors.append('Stage 1 Hypertension')
        elif bp >= 120:
            score += 1
            risk_factors.append('Elevated blood pressure')
        
        # Cholesterol scoring
        chol = patient_data.get('Cholesterol', 0)
        if chol >= 300:
            score += 4
            risk_factors.append('Very high cholesterol (≥300)')
        elif chol >= 240:
            score += 3
            risk_factors.append('High cholesterol (240-299)')
        elif chol >= 200:
            score += 2
            risk_factors.append('Borderline high cholesterol (200-239)')
        
        # Fasting blood sugar scoring
        fbs = patient_data.get('FastingBS', 0)
        if fbs >= 126:
            score += 3
            risk_factors.append('Diabetes (FBS ≥126)')
        elif fbs >= 100:
            score += 2
            risk_factors.append('Prediabetes (FBS 100-125)')
        
        # Exercise angina
        if patient_data.get('ExerciseAngina') == 1:
            score += 3
            risk_factors.append('Exercise-induced angina')
        
        # ST depression (Oldpeak)
        oldpeak = patient_data.get('Oldpeak', 0)
        if oldpeak >= 2.0:
            score += 3
            risk_factors.append('Significant ST depression (≥2.0mm)')
        elif oldpeak >= 1.0:
            score += 2
            risk_factors.append('Positive ST depression (≥1.0mm)')
        
        # Risk categorization
        if score >= 15:
            risk_category = 'VERY HIGH RISK'
            urgency = 'IMMEDIATE medical attention required'
        elif score >= 10:
            risk_category = 'HIGH RISK'
            urgency = 'Urgent cardiology referral recommended'
        elif score >= 6:
            risk_category = 'MODERATE RISK'
            urgency = 'Cardiology consultation advised'
        elif score >= 3:
            risk_category = 'LOW-MODERATE RISK'
            urgency = 'Enhanced monitoring and prevention'
        else:
            risk_category = 'LOW RISK'
            urgency = 'Continue preventive measures'
        
        return {
            'total_score': score,
            'risk_category': risk_category,
            'urgency': urgency,
            'risk_factors': risk_factors
        }
    
    def generate_comprehensive_report(self, patient_data, important_features=None):
        """Generate complete professional report"""
        if important_features is None:
            important_features = list(patient_data.keys())
        
        # Calculate risk score first
        risk_assessment = self.calculate_cardiovascular_risk_score(patient_data)
        
        report = []
        report.append("="*90)
        report.append("COMPREHENSIVE CARDIOVASCULAR RISK ASSESSMENT & RECOMMENDATIONS")
        report.append("="*90)
        
        # Risk Summary
        report.append(f"\n🎯 CARDIOVASCULAR RISK SUMMARY")
        report.append("-" * 50)
        report.append(f"Risk Score: {risk_assessment['total_score']}/20")
        report.append(f"Risk Category: {risk_assessment['risk_category']}")
        report.append(f"Clinical Action: {risk_assessment['urgency']}")
        
        if risk_assessment['risk_factors']:
            report.append(f"\n⚠️  Identified Risk Factors:")
            for factor in risk_assessment['risk_factors']:
                report.append(f"   • {factor}")
        
        # Detailed feature analysis
        for i, feature in enumerate(important_features, 1):
            if feature in self.recommendations:
                value = patient_data.get(feature, 'Not provided')
                rec = self.get_recommendation(feature, value)
                
                report.append(f"\n\n{i}. {feature.upper()} ANALYSIS")
                report.append("=" * 60)
                report.append(f"Current Value: {value}")
                
                report.append(f"\n📋 CLINICAL OVERVIEW:")
                report.append(f"   {rec['general_advice']}")
                
                # Add age-specific recommendations if available
                if 'age_specific' in rec:
                    age_spec = rec['age_specific']
                    report.append(f"\n🎂 AGE-SPECIFIC ASSESSMENT:")
                    report.append(f"   Risk Level: {age_spec['risk_level']}")
                    report.append(f"   Specific Recommendations:")
                    for j, age_rec in enumerate(age_spec['recommendations'], 1):
                        report.append(f"      {j}. {age_rec}")
                
                report.append(f"\n🏥 PROFESSIONAL RECOMMENDATIONS:")
                for j, prof_rec in enumerate(rec['professional_recommendations'], 1):
                    report.append(f"   {j}. {prof_rec}")
                
                report.append(f"\n👨‍⚕️ MEDICAL REFERRAL CRITERIA:")
                report.append(f"   {rec['medical_referral']}")
        
        # Evidence-based treatment recommendations
        report.append(f"\n\n💊 EVIDENCE-BASED TREATMENT RECOMMENDATIONS")
        report.append("=" * 60)
        
        if risk_assessment['total_score'] >= 10:
            report.append("High-intensity interventions recommended:")
            report.append("• Statin therapy (high-intensity): Atorvastatin 40-80mg or Rosuvastatin 20-40mg")
            report.append("• ACE inhibitor/ARB if hypertensive or diabetic")
            report.append("• Antiplatelet therapy consideration (aspirin 75-100mg daily)")
            report.append("• Beta-blocker if post-MI or heart failure")
        elif risk_assessment['total_score'] >= 6:
            report.append("Moderate-intensity interventions recommended:")
            report.append("• Moderate-intensity statin therapy")
            report.append("• Antihypertensive therapy if BP ≥130/80")
            report.append("• Diabetes management with cardiovascular benefits")
        else:
            report.append("Primary prevention focus:")
            report.append("• Lifestyle modifications (diet, exercise, smoking cessation)")
            report.append("• Risk factor monitoring and control")
            report.append("• Consider statin if additional risk factors present")
        
        # Follow-up recommendations
        report.append(f"\n📅 FOLLOW-UP SCHEDULE")
        report.append("-" * 30)
        if risk_assessment['risk_category'] == 'VERY HIGH RISK':
            report.append("• Cardiology: 1-2 weeks")
            report.append("• Primary care: 2-4 weeks")
            report.append("• Laboratory: 4-6 weeks (lipids, metabolic panel)")
        elif risk_assessment['risk_category'] == 'HIGH RISK':
            report.append("• Cardiology: 2-4 weeks")
            report.append("• Primary care: 4-6 weeks")
            report.append("• Laboratory: 6-8 weeks")
        else:
            report.append("• Primary care: 3-6 months")
            report.append("• Laboratory: 6-12 months")
            report.append("• Specialist referral as clinically indicated")
        
        # Medical disclaimer
        report.append("\n" + "="*90)
        report.append("⚠️  IMPORTANT MEDICAL DISCLAIMER")
        report.append("="*90)
        report.append("• This assessment is for educational purposes and clinical decision support only")
        report.append("• All recommendations must be interpreted by qualified healthcare professionals")
        report.append("• Individual patient factors may modify these general recommendations")
        report.append("• Emergency symptoms (chest pain, SOB, syncope) require immediate medical attention")
        report.append("• Treatment decisions should consider patient preferences, comorbidities, and contraindications")
        report.append("="*90)
        
        return "\n".join(report)
    
    def print_feature_list(self):
        """Display available features with clinical significance"""
        print("🫀 CARDIOVASCULAR RISK FACTORS & CLINICAL FEATURES:")
        print("=" * 60)
        for i, feature in enumerate(self.recommendations.keys(), 1):
            print(f"{i:2d}. {feature:<15} - {self.get_clinical_significance(feature)}")
    
    def get_clinical_significance(self, feature):
        """Get brief clinical significance of each feature"""
        significance = {
            'Age': 'Non-modifiable risk factor, exponential risk increase',
            'Sex': 'Gender-specific risk patterns and presentations',
            'ChestPainType': 'Primary symptom for CAD diagnosis',
            'RestingBP': 'Leading modifiable cardiovascular risk factor',
            'Cholesterol': 'Direct correlation with atherosclerotic burden',
            'FastingBS': 'Diabetes increases CV risk 2-4 fold',
            'RestingECG': 'Cardiac structure, rhythm, and ischemic changes',
            'MaxHR': 'Exercise capacity and chronotropic competence',
            'ExerciseAngina': 'Functional significance of coronary stenosis',
            'Oldpeak': 'Exercise-induced ischemia quantification',
            'ST_Slope': 'Pattern analysis for CAD diagnosis'
        }
        return significance.get(feature, 'Clinical significance varies')


# Professional Usage Example
if __name__ == "__main__":
    # Create professional recommendation system
    rec_system = HeartRecommendationSystem()
    
    # Display available clinical features
    print("🏥 PROFESSIONAL CARDIOVASCULAR ASSESSMENT SYSTEM")
    rec_system.print_feature_list()
    
    # High-risk patient example
    high_risk_patient = {
        'Age': 68,
        'Sex': 1,  # Male
        'ChestPainType': 1,  # Typical Angina
        'RestingBP': 165,
        'Cholesterol': 285,
        'FastingBS': 135,  # Diabetic
        'RestingECG': 1,  # ST-T abnormalities
        'MaxHR': 125,  # Reduced for age
        'ExerciseAngina': 1,  # Present
        'Oldpeak': 2.1,  # Significant ST depression
        'ST_Slope': 2  # Downsloping
    }
    
    print(f"\n📊 HIGH-RISK PATIENT CASE STUDY:")
    print("=" * 50)
    for key, value in high_risk_patient.items():
        print(f"{key}: {value}")
    
    # Generate comprehensive professional report
    print(f"\n📋 COMPREHENSIVE CARDIOVASCULAR ASSESSMENT:")
    comprehensive_report = rec_system.generate_comprehensive_report(
        high_risk_patient, 
        ['Age', 'ChestPainType', 'RestingBP', 'Cholesterol', 'ExerciseAngina', 'Oldpeak']
    )
    print(comprehensive_report)
    
    # Risk score calculation
    risk_score = rec_system.calculate_cardiovascular_risk_score(high_risk_patient)
    print(f"\n🎯 RISK STRATIFICATION SUMMARY:")
    print(f"Total Risk Score: {risk_score['total_score']}/20")
    print(f"Risk Category: {risk_score['risk_category']}")
    print(f"Clinical Urgency: {risk_score['urgency']}")

