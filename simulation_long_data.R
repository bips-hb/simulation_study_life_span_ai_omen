
library(dplyr)

#...............................................................................
# 1. Set up parameters ####
#...............................................................................

set.seed(123)

# Number of patients and visits
N_patients <- 1000
N_visits   <- 10

# Number of codes (adjustable for high-dimensionality)
N_ICD <- 50       # start small
N_ATC <- 20       # start small

# Latent health factors
N_factors <- 5    # controls code correlation

# Sparsity control: expected codes per visit
avg_codes_per_visit <- 5

# Temporal dependence
alpha <- 0.3      # 0 = independent visits, 1 = fully persistent



#...............................................................................
# 2. Generate latent patient factors (ground truth) ####
#...............................................................................

# Each patient has a binary vector of latent factors
F_matrix <- matrix(
  rbinom(N_patients * N_factors, 1, 0.2), 
  nrow = N_patients, 
  ncol = N_factors)


# Each row = patient
# Each column = latent condition
# Probability = prevalence of each condition (0.2)


#...............................................................................
# 3. Static covariates ####
#...............................................................................

# Binary static variable (e.g. sex)
sex <- rbinom(N_patients, 1, 0.5)

# Continuous static variable (e.g., baseline risk)
baseline_risk <- runif(N_patients, -1, 1)


# OPTIONAL FOR LATER: CORRELATE STATIC FEATURES WITH LATENT FACTORS

#...............................................................................
# 4. Map factors to code (ground truth) ####
#...............................................................................

# Factor-to-code activation probability (ICD)
W_ICD <- matrix(runif(N_factors * N_ICD, 0.1, 0.6), 
                nrow = N_factors, 
                ncol = N_ICD)

# Factor-to-code activation probability (ATC)
W_ATC <- matrix(runif(N_factors * N_ATC, 0.1, 0.6), 
                nrow = N_factors, 
                ncol = N_ATC)

# Each factor influences multiple codes probabilistically
# Sparsity emerges naturally: not all codes activated per patient/visit




#...............................................................................
# 5. Initialize wide dataset first ####
#...............................................................................

wide <- data.frame(
  patient_id = 1:N_patients,
  sex = sex,
  baseline_risk = baseline_risk
)




#...............................................................................
# 6. Simulate per-visit code occurrences ####
#...............................................................................

simulate_visit <- function(patient_factors, prev_codes = NULL) {
  
  p_ICD <- 1 - apply(1 - (patient_factors %*% W_ICD), 2, prod)
  p_ATC <- 1 - apply(1 - (patient_factors %*% W_ATC), 2, prod)
  
  if (!is.null(prev_codes)) {
    p_ICD <- p_ICD + alpha * prev_codes$ICD
    p_ATC <- p_ATC + alpha * prev_codes$ATC
  }
  
  p_ICD <- pmin(p_ICD, 1)
  p_ATC <- pmin(p_ATC, 1)
  
  ICD <- rbinom(N_ICD, 1, p_ICD)
  ATC <- rbinom(N_ATC, 1, p_ATC)
  
  if (sum(ICD) > avg_codes_per_visit) {
    ICD[sample(which(ICD == 1), sum(ICD) - avg_codes_per_visit)] <- 0
  }
  if (sum(ATC) > avg_codes_per_visit) {
    ATC[sample(which(ATC == 1), sum(ATC) - avg_codes_per_visit)] <- 0
  }
  
  list(ICD = ICD, ATC = ATC)
}


for (t in 1:N_visits) {
  
  ICD_mat <- matrix(0, N_patients, N_ICD)
  ATC_mat <- matrix(0, N_patients, N_ATC)
  
  for (i in 1:N_patients) {
    
    prev <- if (t == 1) NULL else list(
      ICD = as.numeric(wide[i, paste0("ICD", 1:N_ICD, "_t", t - 1)]),
      ATC = as.numeric(wide[i, paste0("ATC", 1:N_ATC, "_t", t - 1)])
    )
    
    visit <- simulate_visit(F_matrix[i, ], prev)
    
    ICD_mat[i, ] <- visit$ICD
    ATC_mat[i, ] <- visit$ATC
  }
  
  colnames(ICD_mat) <- paste0("ICD", 1:N_ICD, "_t", t)
  colnames(ATC_mat) <- paste0("ATC", 1:N_ATC, "_t", t)
  
  wide <- cbind(wide, ICD_mat, ATC_mat)
}




#...............................................................................
# 7. Outcome generation ####
#...............................................................................

# Time weights: recent visits more important
time_weights <- seq(0.5, 1.5, length.out = N_visits)

# Latent factor effects on outcome (ground truth)
beta_factors <- runif(N_factors, 0.5, 1.2)

# Code burden effects
gamma_icd <- 0.15
gamma_atc <- 0.20

# Intercept (controls prevalence)
beta_0 <- -2


# Define coefficients of static variables
beta_sex <- 0.4
beta_risk <- 0.8





# Compute time-weighted code exposure:

# ICD and ATC column names
icd_cols <- grep("^ICD", names(wide), value = TRUE)
atc_cols <- grep("^ATC", names(wide), value = TRUE)

# Time-weighted sums
time_weighted_icd <- rowSums(
  as.matrix(wide[, icd_cols]) *
    rep(time_weights, each = N_patients)
)

time_weighted_atc <- rowSums(
  as.matrix(wide[, atc_cols]) *
    rep(time_weights, each = N_patients)
)


# Add latent factor contribution:
latent_effect <- as.vector(F_matrix %*% beta_factors)



# Build linear predictor and outcome:
linear_pred <- beta_0 +
  latent_effect +
  beta_sex * sex +
  beta_risk * baseline_risk +
  gamma_icd * time_weighted_icd +
  gamma_atc * time_weighted_atc


prob <- plogis(linear_pred)

wide$outcome <- rbinom(N_patients, 1, prob)



#...............................................................................
# 8. Sanity checks ####
#...............................................................................
mean(wide$outcome)              # prevalence
table(wide$outcome)

cor(wide$outcome, time_weighted_icd)
cor(wide$outcome, time_weighted_atc)




#...............................................................................
# 9. Save datasets and ground truth ####
#...............................................................................


write.csv(wide, "ehr_simulated_longitudinal.csv", row.names = FALSE)

saveRDS(
  list(
    meta = list(
      n_visits = N_visits,
      n_icd = N_ICD,
      n_atc = N_ATC,
      time_weights = time_weights
    ),
    ground_truth = list(
      beta_factors = beta_factors,
      gamma_icd = gamma_icd,
      gamma_atc = gamma_atc,
      beta_sex = beta_sex,
      beta_risk = beta_risk
    ),
    F_matrix = F_matrix
  ),
  "ehr_simulation_truth.rds"
)




