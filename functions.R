
# Simulate EHR data ####
# Simulate longitudinal EHR-like data with a binary cancer outcome

# Each simulated patients has:
#   - Baseline information (demographics, socioeconomic status)
#   - Multiple visits over time
#   - Medical codes (ICD for diagnoses, ATC for prescriptions)
#   - Lab measurements
#   - Binary cancer outcome (1 = cancer, 0 = no cancer)

source("helpers.R")


## Baseline variables ####

# patient_id: integer; unique identifier
# age: integer; normally distributed (mean = 60, sd = 12, truncated to greater or equal 18)
# male: binary (0/1); 1 = male, 0 = female; with probability 0.48
# ses: ordinal (1-3): Socioeconomic status; with probabilities 0.3, 0.5, 0.2 (low/mid/high)

# Generate baseline population 
simulate_baseline <- function(n, age_mean = 60, age_sd = 12, male_prob = 0.48, seed = NULL) {
  
  if (!is.null(seed)) set.seed(seed)
  
  id <- seq_len(n)
  age <- round(rtruncnorm(n, a = 18, mean = age_mean, sd = age_sd))
  male <- rbinom(n, 1, male_prob)
  ses <- sample(1:3, n, replace = TRUE, prob = c(0.3,0.5,0.2))
  
  baseline <- data.table(patient_id = id, age = age, male = male, ses = ses)
  baseline
  
}



## Longitudinal data ####

# For each patient: 
#   - Number of visits: Negative Binomial (mean = mean_visits_per_year * followup_years)
#   - Each visit gets a random time within the follow-up window (0-3 years by default)
#   - age_at_visit: baseline age + (time in days / 365)

simulate_visits <- function(baseline, mean_visits_per_year = 2, followup_years = 3,
                            visit_dispersion = 1.2, seed = NULL) {
  
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(baseline)
  visits_list <- vector("list", n)
  
  for (i in seq_len(n)) {
    
    # Poisson number of visits per year, allow overdispersion via NegBin: approximate
    lambda <- mean_visits_per_year * followup_years # (expected 6 visits per patient on average)
    
    # rnbinom parametrization: size chosen from dispersion
    size <- 1 / (visit_dispersion - 1e-6)
    m <- rnbinom(1, size = size, mu = lambda)
    
    if (m == 0) m <- sample(0:1, 1, prob = c(0.5,0.5))
    
    # generate visit times uniformly across followup (in days)
    times <- sort(runif(m, min = 0, max = followup_years * 365))
    
    if (length(times) == 0) {
      visits_list[[i]] <- NULL
      next
      
    }
    
    visits <- data.table(patient_id = baseline$patient_id[i],
                         visit_id = seq_len(length(times)),
                         time_days = times,
                         age_at_visit = baseline$age[i] + times/365)
    visits_list[[i]] <- visits
    
  }
  
  visits_dt <- rbindlist(visits_list, use.names = TRUE, fill = TRUE)
  visits_dt
  
}


## Generate sparse high-dimensional ICD/ATC features at visit-level ####

# ICD and ATC codes are represented as binary indicators per visit (or counts)
simulate_codes_at_visits <- function(visits_dt, n_icd = 200, n_atc = 100,
                                     icd_shape = 1.2, atc_shape = 1.1,
                                     avg_codes_per_visit = 3, seed = NULL) {
  
  if (!is.null(seed)) set.seed(seed)
  m <- nrow(visits_dt)
  icd_probs <- sample_code_probs(n_icd, shape = icd_shape)
  atc_probs <- sample_code_probs(n_atc, shape = atc_shape)
  
  
  # Now for each visit sample number of ICDs and ATCs and then sample codes
  icd_ind <- Matrix(0, nrow = m, ncol = n_icd, sparse = TRUE)
  atc_ind <- Matrix(0, nrow = m, ncol = n_atc, sparse = TRUE)
  row <- 1
  
  for (r in seq_len(m)) {
    
    # poisson number of codes with mean avg_codes_per_visit (can be 0)
    k_icd <- rpois(1, lambda = avg_codes_per_visit)
    k_atc <- rpois(1, lambda = avg_codes_per_visit/2)
    
    if (k_icd > 0) {
      
      codes <- sample.int(n_icd, size = min(k_icd, n_icd), prob = icd_probs)
      icd_ind[row, unique(codes)] <- 1
      
    }
    
    if (k_atc > 0) {
      codes2 <- sample.int(n_atc, size = min(k_atc, n_atc), prob = atc_probs)
      atc_ind[row, unique(codes2)] <- 1
      
    }
    
    row <- row + 1
    
  }
  
  colnames(icd_ind) <- paste0("ICD", sprintf("%03d", seq_len(n_icd)))
  colnames(atc_ind) <- paste0("ATC", sprintf("%03d", seq_len(n_atc)))
  list(icd = icd_ind, atc = atc_ind)
  
}




## Aggregate visit-level codes to patient-level features ####
aggr_codes_to_patient <- function(visits_dt, code_mats, agg_fun = c("sum","any")) {
  agg_fun <- match.arg(agg_fun)
  # attach patient ids to rows
  rows_patient <- visits_dt$patient_id
  icd <- code_mats$icd
  atc <- code_mats$atc
  # Use sparse aggregation
  patients <- unique(rows_patient)
  n_pat <- length(patients)
  # create mapping from visit rows to patient index
  pat_idx <- match(rows_patient, patients)
  # aggregate by patient via matrix multiplication with sparse indicator
  M <- sparseMatrix(i = seq_along(pat_idx), j = pat_idx, x = 1, dims = c(length(pat_idx), n_pat))
  if (agg_fun == "sum") {
    icd_patient <- t(icd) %*% M
    atc_patient <- t(atc) %*% M
    icd_patient <- t(icd_patient)
    atc_patient <- t(atc_patient)
    rownames(icd_patient) <- paste0("patient_", patients)
    rownames(atc_patient) <- paste0("patient_", patients)
  } else {
    # any: convert to logical presence
    icd_patient <- Matrix(0, nrow = n_pat, ncol = ncol(icd), sparse = TRUE)
    atc_patient <- Matrix(0, nrow = n_pat, ncol = ncol(atc), sparse = TRUE)
    for (i in seq_len(n_pat)) {
      vis_rows <- which(pat_idx == i)
      if (length(vis_rows) > 0) {
        icd_patient[i, ] <- as.integer(colSums(icd[vis_rows, , drop = FALSE]) > 0)
        atc_patient[i, ] <- as.integer(colSums(atc[vis_rows, , drop = FALSE]) > 0)
      }
    }
    rownames(icd_patient) <- paste0("patient_", patients)
    rownames(atc_patient) <- paste0("patient_", patients)
  }
  list(icd_patient = icd_patient, atc_patient = atc_patient, patient_id = patients)
}



## Simulate laboratory continuous values (time-varying) ####
simulate_labs <- function(visits_dt, labs = c("hb","crp"), seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  m <- nrow(visits_dt)
  lab_mat <- matrix(NA, nrow = m, ncol = length(labs))
  colnames(lab_mat) <- labs
  for (i in seq_len(m)) {
    # baseline variation by age and sex could be added; here simple random
    lab_mat[i, "hb"] <- rnorm(1, mean = 13 - (visits_dt$age_at_visit[i] - 60)/100, sd = 1.2)
    lab_mat[i, "crp"] <- abs(rnorm(1, mean = 2 + (visits_dt$age_at_visit[i] - 60)/40, sd = 1.5))
  }
  lab_dt <- as.data.table(lab_mat)
  cbind(visits_dt, lab_dt)
}