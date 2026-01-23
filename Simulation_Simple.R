#===============================================================================
# LONGITUDINAL EHR SIMULATION FOR XAI BENCHMARKING
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

# Define specific "Ground Truth" codes
CHRONIC_ICDS <- c(1, 2, 3)    # If these appear often, risk increases
TRIGGER_ICD  <- 10            # The "Cause"
TRIGGER_ATC  <- 5             # The "Response" (must follow Cause)
ESCALATION_WINDOW <- 3        # Last 3 visits


#...............................................................................
# 2. GENERATE LONGITUDINAL DATA ####
# Builds the history for each patient (raw data)
#...............................................................................

# Create "empty" arrays
ICD_arr <- array(0L, dim = c(N_patients, N_visits, N_ICD))
ATC_arr <- array(0L, dim = c(N_patients, N_visits, N_ATC))


#..................... 2.1 Generate random background noise ....................

# Use binomial distribution to randomly scatter codes throughout the visits at a 
# 1% probability. 
# -> Represents "noisy" medical history that doesn't cause the outcome.
for (i in 1:N_patients) {
  
  for (t in 1:N_visits) {
    
    ICD_arr[i, t, ] <- rbinom(N_ICD, 1, 0.01) # low prob. of p=0.01 
    ATC_arr[i, t, ] <- rbinom(N_ATC, 1, 0.01)
    
  }
  
}

#....................... 2.2 Targeted Trigger Injection ........................

# In 15% of patients, force a specific sequence:
# ICD_10 occurs, followed immediately by ATC_5
# -> Simple models like LR should struggle to catch that, whereas RNNs should be
# able to do so

# Force the sequence into 15% of patients to ensure a strong correlation
trigger_target_ids <- sample(1:N_patients, size = N_patients * 0.15)

for(i in trigger_target_ids) {
  
  t_idx <- sample(1:10, 1) # Occurs in the first half of the timeline
  ICD_arr[i, t_idx, TRIGGER_ICD] <- 1
  ATC_arr[i, t_idx + 1, TRIGGER_ATC] <- 1
  
}

#....................... 2.3 Generate Static Variables .........................

# Add static variables (e.g., sex and age)

static_bin  <- rbinom(N_patients, 1, 0.5)                 # e.g., sex
static_cont <- runif(N_patients, 18, 90) / 90             # e.g., normalized age



#.................... 2.4 Generate Lab Values (Continuous Trend) ...............

# 10% of patients are assigned a "rising risk trend", where their lab values 
# increase over time, simulating physiological deterioration

lab_values  <- array(0, dim = c(N_patients, N_visits))

for(i in 1:N_patients) {
  
  # Base value
  base_val <- rnorm(1, 100, 10)
  
  # If patient is in that 10% group, they get a fixed postive "trend" value of 1.5
  # -> Their lab value increases every single visit
  # Other 90% get a random trend near zero, without clear trend
  trend <- if(i %% 10 == 0) 1.5 else rnorm(1, 0, 0.5) 
  
  # Construct the actual values for all 20 visits
  # 1. Base value is the starting point (e.g., 100)
  # 2. Linear growth: For 10% this adds 1.5, 3.0,...,30.0 over time, for 90% this 
  #    adds almost nothing.
  # 3. Measurement noise: Adds noise to every visit so the lines aren't perfectly 
  #    straight -> more realistic
  lab_values[i, ] <- base_val + (1:N_visits * trend) + rnorm(N_visits, 0, 2)
  
}



#...............................................................................
# 3. CALCULATE PHENOTYPE SCORES (The "Ground Truth") ####
# Track exactly why a patient got the outcome
#...............................................................................

# Create "empty" data frame to store information about ground truth
phenotype_contributions <- data.frame(
  patient_id = 1:N_patients,
  chronic_val = 0,
  trigger_val = 0,
  escalation_val = 0,
  static_val = 0, 
  trend_val = 0
)

# Pre-allocation of the truth maps:
# Create two 3D tensors (one for diagnoses, one for medications)
# Initially, they only contain zeros
# As risk-driving events are identified, importance values are set into the
# specific patient-visit-code coordinates
truth_map_icd <- array(0, dim = c(N_patients, N_visits, N_ICD))
truth_map_atc <- array(0, dim = c(N_patients, N_visits, N_ATC))


for (i in 1:N_patients) {
  
  #............................... 1. Chronic ..................................
  
  # Risk is linear and additive; the more often ICD 1, 2, or 3 appear, the higher
  # the patient's risk
  phenotype_contributions$chronic_val[i] <- 
    sum(colSums(ICD_arr[i, , CHRONIC_ICDS, drop=FALSE])) * 0.5
  
  for(v in 1:N_visits) { 
    
    # Every time one of these codes appears, the ground truth is marked with a 
    # value of 0.25
    truth_map_icd[i, v, CHRONIC_ICDS] <- ICD_arr[i, v, CHRONIC_ICDS] * 0.25 
    
    }
  
  
  
  #........................... 2. Trigger (Sequence) ...........................
  
  # "Smoking gun": Look for an ICD 10 code (which is the cause), followed by an 
  # ATC 5 (response) within 2 visits
  # If the sequence is found, the patient gets a massive risk score (10.0)
  # -> Both ICD 10 and ATC 5 get both marked with a high importance weight of 1.0 
  #    in the truth map
  # -> Test if DL models can see the relationship between two different time points
  
  times_icd <- which(ICD_arr[i, , TRIGGER_ICD] == 1)
  times_atc <- which(ATC_arr[i, , TRIGGER_ATC] == 1)
  
  if(length(times_icd) > 0 && length(times_atc) > 0) {
    
    for(t_icd in times_icd) {
      
      future_atc <- times_atc[times_atc > t_icd & times_atc <= (t_icd + 2)]
      
      if(length(future_atc) > 0) {
        
        phenotype_contributions$trigger_val[i] <- 10.0
        truth_map_icd[i, t_icd, TRIGGER_ICD] <- 1.0
        
        for(t_atc in future_atc) { 
          
          truth_map_atc[i, t_atc, TRIGGER_ATC] <- 1.0 
          
        }
        
      }
      
    }
    
  }
  
  
  
  #.......................... 3. Escalation (Late Spike) .......................
  
  # Detects a sudden spike in clinical activity during the final 3 visits (late window)
  # If detected, the risk score is increased
  # -> Mark all codes occurring during those final visits as important (0.5),
  # -> Timing of the codes is what drove the risk, not just the codes themselves
  
  late_idx <- (N_visits - ESCALATION_WINDOW + 1):N_visits
  
  if(sum(ICD_arr[i, late_idx, ]) > (sum(ICD_arr[i, 1:(N_visits-3), ])/17 + 2)) {
    
    phenotype_contributions$escalation_val[i] <- 2.5
    
    truth_map_icd[i, late_idx, ] <- ICD_arr[i, late_idx, ] * 0.5
    
  }
  
  
  
  #...................... 4. Multimodal Logic (Static + Trend) .................
  
  # Adds risk based on non-temporal factors (static) and the velocity of the 
  # lab values (trend)
  # Note: Because they are continuous or static features, they aren't marked in
  # the discrete truth_map_icd tensor, but they contribute to the final probability
  # calculation, providing "background risk" that the DL models must distinguish 
  # from the "medical event" risk
  
  phenotype_contributions$static_val[i] <- 
    (static_bin[i] * 0.8) + (static_cont[i] * 0.5)
  
  if(lab_values[i, 20] - lab_values[i, 1] > 10) {
    
    phenotype_contributions$trend_val[i] <- 3.0
    
  }
  
}


#...............................................................................
# 4. BINARY OUTCOME GENERATION ####
#...............................................................................


#............................... Aggregate the risk ............................

# Sum up the chronic risk, trigger risk, escalation risk etc.
total_risk_signal <- rowSums(phenotype_contributions[,-1])


#........................... Prevalence calibration ............................

# Helper function to find the intercept that hits our target
find_intercept <- function(intercept) {
  
  # total_risk_signal: Determines relative risk (who is sicker than whom)
  # intercept: Determines absolute risk (how easy it is for anyone to get sick)
  probs <- 1 / (1 + exp(-(intercept + total_risk_signal)))
  
  return(mean(probs) - target_prevalence)
  
}

# Target prevalence: We want roughly 30% of patients to be "Positive"
target_prevalence <- 0.30

# Solve for the intercept 
calibrated_intercept <- uniroot(find_intercept, interval = c(-50, 10))$root

# Generate final outcomes
final_probs <- 1 / (1 + exp(-(calibrated_intercept + total_risk_signal)))
outcomes <- rbinom(N_patients, 1, final_probs)

# Check Stats 
cat("Correlations with Outcome:\n")
print(cor(phenotype_contributions[,2:4], outcomes))
print(summary(final_probs))





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




#...............................................................................

# Updated Plotting for Patient 1802 (Includes ATCs)
pt_truth_combined <- cbind(truth_map_icd[1802,,], truth_map_atc[1802,,])
colnames(pt_truth_combined) <- c(paste0("ICD_", 1:N_ICD), paste0("ATC_", 1:N_ATC))
active_cols <- which(colSums(abs(pt_truth_combined)) > 0)

if(length(active_cols) > 0){
  plot_data <- melt(pt_truth_combined[, active_cols])
  colnames(plot_data) <- c("Visit", "Code_Name", "Weight")
  
  ggplot(plot_data, aes(x = Visit, y = Code_Name, fill = Weight)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red") +
    theme_minimal() +
    labs(title = "Ground Truth Heatmap (ICD + ATC)")
}



# Load the truth maps we saved earlier
# truth_map_icd shape: [N_patients, N_visits, N_ICD]
patient_idx <- 1802

# 1. Extract this patient's truth matrix
# Combine ICD and ATC truth for a full view
pt_truth_icd <- truth_map_icd[patient_idx, , ]
pt_truth_atc <- truth_map_atc[patient_idx, , ]

# 2. Identify which codes actually have non-zero truth values
# This helps us zoom in on the important rows
active_icds <- which(colSums(abs(pt_truth_icd)) > 0)
active_atcs <- which(colSums(abs(pt_truth_atc)) > 0)

# 3. Create a clean dataframe for plotting
library(ggplot2)
library(reshape2)

plot_data <- melt(pt_truth_icd[, active_icds])
colnames(plot_data) <- c("Visit", "Code_Idx", "Weight")
plot_data$Code_Name <- paste0("ICD_", active_icds[plot_data$Code_Idx])

# 4. Plot the "Gold Standard"
ggplot(plot_data, aes(x = Visit, y = Code_Name, fill = Weight)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red") +
  theme_minimal() +
  labs(title = paste("Ground Truth Importance: Patient", patient_idx),
       x = "Visit Number", y = "Medical Code")


# Add this to visualize the Lab Trend for the same patient (1802)
pt_lab <- data.frame(Visit = 1:N_visits, Value = lab_values[patient_idx, ])

ggplot(pt_lab, aes(x = Visit, y = Value)) +
  geom_line(color = "darkgreen", size = 1) +
  geom_point() +
  theme_minimal() +
  labs(title = paste("Lab Value Trend: Patient", patient_idx),
       y = "Continuous Lab Value", x = "Visit")






# Combined Truth Plotting Logic
pt_truth_combined <- cbind(pt_truth_icd, pt_truth_atc)
colnames(pt_truth_combined) <- c(paste0("ICD_", 1:N_ICD), paste0("ATC_", 1:N_ATC))

active_cols <- which(colSums(abs(pt_truth_combined)) > 0)
final_plot_matrix <- pt_truth_combined[, active_cols]

plot_data <- melt(final_plot_matrix)
colnames(plot_data) <- c("Visit", "Code_Name", "Weight")

ggplot(plot_data, aes(x = Visit, y = Code_Name, fill = Weight)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red") +
  theme_minimal()
