#===============================================================================
# LONGITUDINAL EHR SIMULATION FOR XAI BENCHMARKING/EVALUATION
# 
# Logic: Outcome is driven by 3 specific "Phenotypes"
# 1. CHRONIC: High frequency of specific codes (Easy for any model)
# 2. TRIGGER: ICD_A followed by ATC_B (Sequence-dependent, hard for LR)
# 3. ESCALATION: Sudden spike in codes at the end (Temporal-dependent)
#===============================================================================

# Set seed for reproducibility
set.seed(3105)

# Load required packages
library(dplyr)
library(tidyr)
library(reshape2)
library(ggplot2)


#...............................................................................
# 1. GLOBAL PARAMETERS ####
#...............................................................................

N_patients <- 3000 # number of patients
N_visits   <- 20 # number of visits
N_ICD      <- 100 # number of diagnosis codes (ICD)
N_ATC      <- 40 # number of medication codes (ATC)

# Define Ground Truth codes
CHRONIC_ICDS      <- c(1, 2, 3)
TRIGGER_ICD       <- 10  # Risk Trigger
TRIGGER_ATC       <- 5   # Risk Response
PROT_RISK_ICD     <- 20  # Risk-Neutralization Condition
PROT_ANTIDOTE_ATC <- 15  # Risk-Neutralization Antidote
PURE_PROT_ICD     <- 30  # Purely Protective Condition
PURE_PROT_ATC     <- 25  # Purely Protective Treatment
ESCALATION_WINDOW <- 3 

#...............................................................................
# 2. GENERATE LONGITUDINAL DATA ####
#...............................................................................
ICD_arr <- array(0L, dim = c(N_patients, N_visits, N_ICD))
ATC_arr <- array(0L, dim = c(N_patients, N_visits, N_ATC))

# 2.1 Background Noise
for (i in 1:N_patients) {
  for (t in 1:N_visits) {
    ICD_arr[i, t, ] <- rbinom(N_ICD, 1, 0.01)
    ATC_arr[i, t, ] <- rbinom(N_ATC, 1, 0.01)
  }
}

# 2.2 Targeted Injection of Patterns
# Scenario A: Sequential Risk Trigger (15%)
trigger_ids <- sample(1:N_patients, size = N_patients * 0.15)
for(i in trigger_ids) {
  t_idx <- sample(1:10, 1)
  ICD_arr[i, t_idx, TRIGGER_ICD] <- 1
  ATC_arr[i, t_idx + 1, TRIGGER_ATC] <- 1
}

# # Scenario B: Risk-Neutralization (10% total: 5% Risk only, 5% Risk + Antidote)
# neutral_ids <- sample(setdiff(1:N_patients, trigger_ids), size = N_patients * 0.10)
# antidote_ids <- neutral_ids[1:(length(neutral_ids)/2)] # Half get the cure
# 
# for(i in neutral_ids) {
#   t_idx <- sample(5:10, 1)
#   ICD_arr[i, t_idx, PROT_RISK_ICD] <- 1 # Everyone in this group gets the risk
#   if(i %in% antidote_ids) {
#     ATC_arr[i, t_idx + 1, PROT_ANTIDOTE_ATC] <- 1 # Only half get the antidote
#   }
# }

# Scenario C: Purely Protective Sequence (10%)
# pure_prot_ids <- sample(setdiff(1:N_patients, c(trigger_ids, neutral_ids)), size = N_patients * 0.10)
# for(i in pure_prot_ids) {
#   t_idx <- sample(1:15, 1)
#   ICD_arr[i, t_idx, PURE_PROT_ICD] <- 1
#   ATC_arr[i, t_idx + 1, PURE_PROT_ATC] <- 1
# }

# 2.3 Static & Lab Trend (Keep your existing logic)
static_bin  <- rbinom(N_patients, 1, 0.5)
static_cont <- runif(N_patients, 18, 90) / 90
lab_values  <- array(0, dim = c(N_patients, N_visits))
for(i in 1:N_patients) {
  base_val <- rnorm(1, 100, 10)
  trend <- if(i %% 10 == 0) 1.5 else rnorm(1, 0, 0.5) 
  lab_values[i, ] <- base_val + (1:N_visits * trend) + rnorm(N_visits, 0, 2)
}

#...............................................................................
# 3. CALCULATE PHENOTYPE SCORES (The Ground Truth) ####
#...............................................................................
phenotype_contributions <- data.frame(
  patient_id = 1:N_patients, chronic_val = 0, trigger_val = 0,
  escalation_val = 0, static_val = 0, trend_val = 0,
  neutralization_val = 0#, pure_protective_val = 0
)

truth_map_icd <- array(0, dim = c(N_patients, N_visits, N_ICD))
truth_map_atc <- array(0, dim = c(N_patients, N_visits, N_ATC))

for (i in 1:N_patients) {
  # 1. Chronic Logic (Same)
  phenotype_contributions$chronic_val[i] <- sum(colSums(ICD_arr[i, , CHRONIC_ICDS, drop=FALSE])) * 0.5
  truth_map_icd[i, , CHRONIC_ICDS] <- ICD_arr[i, , CHRONIC_ICDS] * 0.25
  
  # 2. Trigger Logic (Same)
  t_icd <- which(ICD_arr[i, , TRIGGER_ICD] == 1)
  t_atc <- which(ATC_arr[i, , TRIGGER_ATC] == 1)
  if(length(t_icd) > 0 && any(t_atc > min(t_icd) & t_atc <= (min(t_icd) + 2))) {
    phenotype_contributions$trigger_val[i] <- 10.0
    truth_map_icd[i, t_icd, TRIGGER_ICD] <- 1.0
    truth_map_atc[i, t_atc[t_atc > min(t_icd)], TRIGGER_ATC] <- 1.0
  }
  
  # # 3. Risk-Neutralization (Antidote)
  # t_risk_icd <- which(ICD_arr[i, , PROT_RISK_ICD] == 1)
  # t_anti_atc <- which(ATC_arr[i, , PROT_ANTIDOTE_ATC] == 1)
  # if(length(t_risk_icd) > 0) {
  #   phenotype_contributions$neutralization_val[i] <- 5.0 # Initial Risk
  #   truth_map_icd[i, t_risk_icd, PROT_RISK_ICD] <- 0.5
  #   # Antidote check
  #   if(any(t_anti_atc > min(t_risk_icd) & t_anti_atc <= (min(t_risk_icd) + 2))) {
  #     phenotype_contributions$neutralization_val[i] <- -2.0 # Flips to Protective
  #     truth_map_atc[i, t_anti_atc[t_anti_atc > min(t_risk_icd)], PROT_ANTIDOTE_ATC] <- -1.0 # BLUE
  #   }
  # }
  
  # # 4. Purely Protective Sequence
  # t_p_icd <- which(ICD_arr[i, , PURE_PROT_ICD] == 1)
  # t_p_atc <- which(ATC_arr[i, , PURE_PROT_ATC] == 1)
  # if(length(t_p_icd) > 0 && any(t_p_atc > min(t_p_icd) & t_p_atc <= (min(t_p_icd) + 2))) {
  #   phenotype_contributions$pure_protective_val[i] <- -5.0
  #   truth_map_icd[i, t_p_icd, PURE_PROT_ICD] <- -0.5 # BLUE
  #   truth_map_atc[i, t_p_atc[t_p_atc > min(t_p_icd)], PURE_PROT_ATC] <- -0.5 # BLUE
  # }
  
  # 5. Escalation, Static, and Trend
  late_idx <- (N_visits - ESCALATION_WINDOW + 1):N_visits
  if(sum(ICD_arr[i, late_idx, ]) > (sum(ICD_arr[i, 1:(N_visits-3), ])/17 + 2)) {
    phenotype_contributions$escalation_val[i] <- 2.5
    truth_map_icd[i, late_idx, ] <- ICD_arr[i, late_idx, ] * 0.5
  }
  phenotype_contributions$static_val[i] <- (static_bin[i] * 0.8) + (static_cont[i] * 0.5)
  if(lab_values[i, 20] - lab_values[i, 1] > 10) phenotype_contributions$trend_val[i] <- 3.0
}

#...............................................................................
# 4. OUTCOME GENERATION (Calibration handled automatically) ####
#...............................................................................
total_risk_signal <- rowSums(phenotype_contributions[,-1])
target_prevalence <- 0.30
find_intercept <- function(intercept) mean(1 / (1 + exp(-(intercept + total_risk_signal)))) - target_prevalence
calibrated_intercept <- uniroot(find_intercept, interval = c(-50, 10))$root
final_probs <- 1 / (1 + exp(-(calibrated_intercept + total_risk_signal)))
outcomes <- rbinom(N_patients, 1, final_probs)


#...............................................................................
# 5. DATA EXPORT  ####
#...............................................................................


#....................... 5.1 Discrete Events (Long format) .....................

event_list <- list()

# Scan the sparse 3D matrices (ICD_arr and ATC_arr) to find every instance where 
# a medical code occurred (Value of 1)

# Convert those coordinates into a "long format" table

# The resulting file/dataset contains the sequence of medical events for every 
# patient, which the DL models will use to learn the temporal "trigger"

for(i in 1:N_patients) {
  
  icd_pos <- which(ICD_arr[i,,] == 1, arr.ind = TRUE)
  if(nrow(icd_pos) > 0) event_list[[length(event_list)+1]] <- 
      data.frame(patient_id = i, time = icd_pos[,1], token = paste0("ICD_", icd_pos[,2]))
  
  atc_pos <- which(ATC_arr[i,,] == 1, arr.ind = TRUE)
  if(nrow(atc_pos) > 0) event_list[[length(event_list)+1]] <- 
    data.frame(patient_id = i, time = atc_pos[,1], token = paste0("ATC_", atc_pos[,2]))
  
}

write.csv(do.call(rbind, event_list), "ehr_events.csv", row.names = FALSE)



#.............................. 5.2 Multimodal Files ...........................

write.csv(data.frame(patient_id = 1:N_patients, static_bin, static_cont, outcome = outcomes), # wide format
          "static_data.csv", 
          row.names = FALSE)

# Take the 2D lab matrix and turn it into a long list
write.csv(melt(lab_values, varnames = c("patient_id", "time")), 
          "lab_data.csv", 
          row.names = FALSE)

#........................ 5.3 Ground Truth (for XAI validation) ................

saveRDS(list(truth_icd = truth_map_icd, truth_atc = truth_map_atc, outcomes = outcomes), 
        "simulation_truth.rds")



# Complete signal with prevalence check: 
cat("Simulation Complete. Prevalence:", mean(outcomes), "\n")



