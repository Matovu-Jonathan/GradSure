###############################################################
# Graduation Assurance Product Simulation (Uganda)
# Author: Group 9 (Final Professional Version with Refined Exposure Pricing)
# Notes:
#  - Professional risk-based pricing with refined exposure component
#  - Separate risk and exposure components for transparency
#  - CLAIM_CAP set to UGX 15,000,000
#  - MATURITY_BENEFIT set to 500,000 UGX
###############################################################

import numpy as np
import pandas as pd

rng = np.random.default_rng(12345)  # reproducibility

# ----------------------------------------------------------------
#  PORTFOLIO PARAMETERS
# ----------------------------------------------------------------
N_POLICIES = 10_000          
SEMESTER_MONTHS = 4
SEMESTER_YEARS = SEMESTER_MONTHS / 12
BASE_PREMIUM = 180_000        
CLAIM_CAP = 15_000_000       
MATURITY_BENEFIT = 500_000   
MEDIAN_TUITION = 2_500_000   # UGX median tuition for exposure calculation

# ----------------------------------------------------------------
#  UGANDA ASSURED LIVES MORTALITY TABLE (2015–2019)
# ----------------------------------------------------------------
ua_data = {
    20:(0.001563,0.002015),21:(0.001822,0.002103),22:(0.002053,0.002174),
    23:(0.002251,0.002227),24:(0.002405,0.002260),25:(0.002508,0.002272),
    26:(0.002554,0.002259),27:(0.002554,0.002228),28:(0.002520,0.002180),
    29:(0.002461,0.002123),30:(0.002393,0.002060),31:(0.002325,0.001995),
    32:(0.002270,0.001936),33:(0.002239,0.001888),34:(0.002246,0.001857),
    35:(0.002297,0.001847),36:(0.002390,0.001858),37:(0.002518,0.001888),
    38:(0.002674,0.001932),39:(0.002855,0.001988),40:(0.003049,0.002052),
    41:(0.003253,0.002121),42:(0.003458,0.002194),43:(0.003656,0.002266),
    44:(0.003843,0.002336),45:(0.004015,0.002512),46:(0.004177,0.002671),
    47:(0.004330,0.002840),48:(0.004475,0.003021),49:(0.004613,0.003213),
    50:(0.004746,0.003417),51:(0.004877,0.003861),52:(0.005006,0.004362),
    53:(0.005135,0.004924),54:(0.005264,0.005551),55:(0.005394,0.006252),
    56:(0.006056,0.007033),57:(0.006958,0.007903),58:(0.007991,0.008872),
    59:(0.009178,0.009948),60:(0.010542,0.011143),61:(0.012059,0.012468),
    62:(0.013796,0.013936),63:(0.015751,0.015559),64:(0.017947,0.017352),
    65:(0.020409,0.019331),66:(0.023163,0.021513),67:(0.026236,0.023915),
    68:(0.029657,0.026556),69:(0.033459,0.029457),70:(0.037672,0.032639)
}

def get_qx(age, sex):
    age_int = int(min(max(round(age), 20), 70))
    q_m, q_f = ua_data[age_int]
    return q_m if sex == "M" else q_f

# ----------------------------------------------------------------
#  PROFESSIONAL RISK SCORING FUNCTIONS
# ----------------------------------------------------------------
def calculate_actuarial_risk_score(sponsor_age, sponsor_sex, income_band, 
                                   scholarship_flag, sponsor_relationship, expected_years):
    risk_score = 0
    if sponsor_age >= 65: risk_score += 4.0
    elif sponsor_age >= 55: risk_score += 2.5
    elif sponsor_age >= 45: risk_score += 1.5
    elif sponsor_age >= 35: risk_score += 0.5
    if income_band == "low": risk_score += 2.0
    elif income_band == "high": risk_score -= 1.0
    if scholarship_flag == 1: risk_score -= 1.5
    if sponsor_relationship == "grandparent": risk_score += 1.0
    elif sponsor_relationship == "other_relative": risk_score += 0.5
    elif sponsor_relationship == "other": risk_score += 1.5
    if expected_years == 5: risk_score += 1.0
    elif expected_years == 3: risk_score -= 0.5
    return max(0, risk_score)

def classify_risk_tier(risk_score):
    if risk_score <= 1.0: return "PREFERRED"
    elif risk_score <= 2.5: return "STANDARD"
    elif risk_score <= 4.5: return "SUBSTANDARD"
    else: return "HIGH_RISK"

def calculate_risk_premium(risk_tier, tuition_per_sem, base_premium=BASE_PREMIUM):
    tier_multipliers = {
        "PREFERRED": 0.75,
        "STANDARD": 1.00,
        "SUBSTANDARD": 1.40,
        "HIGH_RISK": 1.80
    }
    
    # Risk component (personal risk factors - 85% of pricing)
    risk_component = base_premium * tier_multipliers[risk_tier]
    
    # Exposure component (tuition-based - 15% of pricing) - capped and gradual
    # Only charge for tuition above median, max UGX 75,000 loading
    exposure_loading = min(75_000, max(0, (tuition_per_sem - MEDIAN_TUITION) * 0.03))
    
    return risk_component + exposure_loading

# ----------------------------------------------------------------
#  GENERATE SPONSOR & STUDENT CHARACTERISTICS
# ----------------------------------------------------------------
sponsor_sex = rng.choice(["M","F"], size=N_POLICIES, p=[0.6,0.4])
age_bands = rng.choice(["30_39","40_60","61_70"], size=N_POLICIES, p=[0.15,0.65,0.20])
sponsor_age = [int(rng.integers(30,40)) if b=="30_39" else int(rng.integers(40,61)) if b=="40_60" else int(rng.integers(61,71)) for b in age_bands]

sponsor_relationship = []
for age in sponsor_age:
    if age >= 61:
        sponsor_relationship.append(rng.choice(["parent","grandparent","other_relative","other"], p=[0.35,0.45,0.15,0.05]))
    else:
        sponsor_relationship.append(rng.choice(["parent","grandparent","other_relative","other"], p=[0.70,0.07,0.15,0.08]))

student_age = rng.integers(18,26,size=N_POLICIES)
student_sex = rng.choice(["M","F"], size=N_POLICIES, p=[0.55,0.45])
income_band = rng.choice(["low","medium","high"], size=N_POLICIES, p=[0.4,0.4,0.2])
scholarship_flag = rng.integers(0,2,size=N_POLICIES)
tuition_per_sem = np.round(rng.uniform(1.2,3.5,size=N_POLICIES)*1e6)
expected_years = rng.choice([3,4,5], size=N_POLICIES, p=[0.3,0.5,0.2])

# ----------------------------------------------------------------
#  CALCULATE RISK SCORES AND PREMIUMS
# ----------------------------------------------------------------
risk_scores, risk_tiers, premiums_per_sem = [], [], []
for i in range(N_POLICIES):
    score = calculate_actuarial_risk_score(sponsor_age[i], sponsor_sex[i], income_band[i],
                                           scholarship_flag[i], sponsor_relationship[i], expected_years[i])
    tier = classify_risk_tier(score)
    premium = calculate_risk_premium(tier, tuition_per_sem[i])
    
    risk_scores.append(score)
    risk_tiers.append(tier)
    premiums_per_sem.append(premium)

policies = pd.DataFrame({
    "policy_id": np.arange(1,N_POLICIES+1),
    "sponsor_age": sponsor_age,
    "sponsor_sex": sponsor_sex,
    "sponsor_relationship": sponsor_relationship,
    "student_age": student_age,
    "student_sex": student_sex,
    "income_band": income_band,
    "scholarship_flag": scholarship_flag,
    "tuition_per_sem": tuition_per_sem,
    "expected_years": expected_years,
    "risk_score": risk_scores,
    "risk_tier": risk_tiers,
    "premium_per_sem": premiums_per_sem
})

# ----------------------------------------------------------------
#  SEMESTER PANEL GENERATION
# ----------------------------------------------------------------
records = []
for i in range(N_POLICIES):
    sems = int(policies.loc[i,"expected_years"]*2)
    age = policies.loc[i,"sponsor_age"]
    sex = policies.loc[i,"sponsor_sex"]
    income = policies.loc[i,"income_band"]
    scholar = policies.loc[i,"scholarship_flag"]
    tuition = policies.loc[i,"tuition_per_sem"]
    
    for s in range(1,sems+1):
        q_annual = get_qx(age, sex)
        q_sem = 1 - (1 - q_annual)**SEMESTER_YEARS
        sponsor_died = rng.binomial(1, q_sem)
        sponsor_shock = rng.binomial(1, 0.035)
        
        base_drop = 0.015
        inc_mult = 2.0 if income=="low" else 0.6 if income=="high" else 1.0
        sch_mult = 0.3 if scholar==1 else 1.0
        sem_decay = max(0.4,1-0.05*(s-1))
        dropout_prob = base_drop*inc_mult*sch_mult*sem_decay
        dropout_flag = rng.binomial(1, dropout_prob)
        
        claim_flag = 1 if (sponsor_died==1 or (sponsor_shock==1 and rng.random()<0.2)) else 0
        tuition_paid_flag = rng.binomial(1,0.96 if income!="low" else 0.864)
        remaining_tuition = tuition*(sems-s+1)
        
        if claim_flag==1:
            raw_claim = remaining_tuition*np.exp(rng.normal(0,0.2))
            claim_amount = min(raw_claim, CLAIM_CAP)
        else:
            claim_amount = 0.0
        
        records.append([i+1,s,age,sponsor_died,sponsor_shock,dropout_flag,claim_flag,tuition_paid_flag,claim_amount,remaining_tuition])
        age += SEMESTER_YEARS
        if claim_flag==1 or dropout_flag==1:
            break

semester_panel = pd.DataFrame(records,columns=["policy_id","semester_number","sponsor_age","sponsor_died",
                                               "sponsor_shock","dropout_flag","claim_flag","tuition_paid_flag",
                                               "claim_amount","remaining_tuition"])

# ----------------------------------------------------------------
#  MATURITY BENEFIT CALCULATION
# ----------------------------------------------------------------
maturity_payments = []
for i, policy in policies.iterrows():
    sems = int(policy["expected_years"]*2)
    policy_sem_panel = semester_panel[semester_panel["policy_id"]==policy["policy_id"]]
    if policy_sem_panel["claim_flag"].sum()==0 and policy_sem_panel["dropout_flag"].sum()==0:
        maturity_payments.append(MATURITY_BENEFIT)
    else:
        maturity_payments.append(0)
policies["maturity_benefit"]=maturity_payments

# ----------------------------------------------------------------
#  SAVE OUTPUTS
# ----------------------------------------------------------------
policies.to_csv("policies(python).csv",index=False)
semester_panel.to_csv("semester_panel(python).csv",index=False)

# ----------------------------------------------------------------
#  SUMMARY OUTPUT
# ----------------------------------------------------------------
total_claims = semester_panel["claim_amount"].sum()
total_maturity = sum(maturity_payments)
total_claims_with_maturity = total_claims + total_maturity
total_premiums = (policies["premium_per_sem"]*policies["expected_years"]*2).sum()
loss_ratio_with_maturity = (total_claims_with_maturity/total_premiums)*100

print("Simulation complete ")
print(f"Policies: {len(policies):,}")
print(f"Semester rows: {len(semester_panel):,}")
print(f"Claims: {semester_panel['claim_flag'].sum():,}")
print(f"Dropouts: {semester_panel['dropout_flag'].sum():,}")
print(f"Total Claims (UGX): {total_claims:,.0f}")
print(f"Total Maturity Benefits (UGX): {total_maturity:,.0f}")
print(f"Total Claims + Maturity (UGX): {total_claims_with_maturity:,.0f}")
print(f"Total Premiums (UGX): {total_premiums:,.0f}")
print(f"Loss Ratio w/ Maturity Benefit: {loss_ratio_with_maturity:.2f}%")

print("\n-- Quick breakdowns --")
print("Avg tuition per sem (UGX):", int(policies["tuition_per_sem"].mean()))
print("Avg premium per sem (UGX):", int(policies["premium_per_sem"].mean()))
print("Avg expected years:", round(policies["expected_years"].mean(),2))
print("Mean claim amount (given claim):", int(
    semester_panel.loc[semester_panel['claim_flag']==1,'claim_amount'].mean()
    if semester_panel['claim_flag'].sum()>0 else 0))

# ----------------------------------------------------------------
#  PROFESSIONAL RISK & PRICING ANALYSIS
# ----------------------------------------------------------------
print("\n" + "="*70)
print("PROFESSIONAL RISK-BASED PRICING ANALYSIS")
print("="*70)

# Risk score analysis
print(f"\n RISK SCORE ANALYSIS:")
print(f"Average Risk Score:     {policies['risk_score'].mean():.2f}")
print(f"Median Risk Score:      {policies['risk_score'].median():.2f}")
print(f"Risk Score Range:       {policies['risk_score'].min():.1f} to {policies['risk_score'].max():.1f}")
print(f"Standard Deviation:     {policies['risk_score'].std():.2f}")

# Risk tier distribution and performance
print(f"\n RISK TIER DISTRIBUTION & PERFORMANCE:")
print("-" * 65)

for tier in ["PREFERRED", "STANDARD", "SUBSTANDARD", "HIGH_RISK"]:
    tier_policies = policies[policies["risk_tier"] == tier]
    tier_count = len(tier_policies)
    
    if tier_count > 0:
        # Calculate claim experience for this tier
        tier_semesters = semester_panel[semester_panel["policy_id"].isin(tier_policies["policy_id"])]
        claim_count = tier_semesters["claim_flag"].sum()
        claim_rate = (claim_count / len(tier_semesters)) * 100 if len(tier_semesters) > 0 else 0
        
        avg_premium = tier_policies["premium_per_sem"].mean()
        avg_risk_score = tier_policies["risk_score"].mean()
        percentage = (tier_count / N_POLICIES) * 100
        
        print(f"{tier:12} | {tier_count:4} policies ({percentage:5.1f}%) | "
              f"Avg Risk: {avg_risk_score:.2f} | "
              f"Premium: UGX {avg_premium:,.0f} | "
              f"Claim Rate: {claim_rate:.2f}%")

# Premium statistics
print(f"\n PREMIUM STATISTICS:")
print(f"Minimum Premium:       UGX {policies['premium_per_sem'].min():,.0f}")
print(f"25th Percentile:       UGX {policies['premium_per_sem'].quantile(0.25):,.0f}")
print(f"Median Premium:        UGX {policies['premium_per_sem'].median():,.0f}")
print(f"75th Percentile:       UGX {policies['premium_per_sem'].quantile(0.75):,.0f}")
print(f"Maximum Premium:       UGX {policies['premium_per_sem'].max():,.0f}")
print(f"Standard Deviation:    UGX {policies['premium_per_sem'].std():,.0f}")
print(f"Premium Range:         UGX {policies['premium_per_sem'].max() - policies['premium_per_sem'].min():,.0f}")

# Exposure analysis
print(f"\n EXPOSURE ANALYSIS:")
print(f"Median Tuition:        UGX {MEDIAN_TUITION:,.0f}")
print(f"Average Tuition:       UGX {policies['tuition_per_sem'].mean():,.0f}")
print(f"Tuition Range:         UGX {policies['tuition_per_sem'].min():,.0f} to UGX {policies['tuition_per_sem'].max():,.0f}")

# Financial metrics with expense and profit analysis
expense_ratio = 0.15  # 15% for mandatory university model
profit_margin = max(0, (100 - loss_ratio_with_maturity - expense_ratio))

print(f"\n PROFITABILITY ANALYSIS:")
print(f"Loss Ratio:            {loss_ratio_with_maturity:.1f}%")
print(f"Expense Ratio:         {expense_ratio:.1%}")
print(f"Profit Margin:         {profit_margin:.1f}%")
print(f"Estimated ROE:         {profit_margin * 3:.1f}%")  # Assuming 3x leverage
