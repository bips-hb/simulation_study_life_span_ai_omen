#===============================================================================
# LONGITUDINAL EHR SIMULATION FOR XAI BENCHMARKING/EVALUATION
# Parameterized for multi-scenario comparison
#===============================================================================

library(dplyr)
library(tidyr)
library(reshape2)

# ==============================================================================
# CORE SIMULATION FUNCTION
# ==============================================================================

simulate_ehr <- function(
    N_patients        = 3000,
    N_visits          = 20,
    N_ICD             = 100,
    N_ATC             = 40,
    trigger_prop      = 0.15,   # proportion of patients with trigger phenotype
    escalation_prop   = 0.10,   # proportion of patients with escalation phenotype
    bg_noise_icd      = 0.01,   # background ICD noise probability
    bg_noise_atc      = 0.01,   # background ATC noise probability
    escalation_noise  = 0.08,   # ICD spike rate during escalation window
    target_prevalence = 0.30,
    seed              = 3105,
    output_dir        = ".",
    save_files        = TRUE,
    verbose           = TRUE
) {
  
  set.seed(seed)
  
  # ---------------------------------------------------------------------------
  # DERIVED / FIXED PARAMETERS
  # ---------------------------------------------------------------------------
  CHRONIC_ICDS      <- c(1, 2, 3)
  TRIGGER_ICD       <- 10
  TRIGGER_ATC       <- 5
  ESCALATION_WINDOW <- 3
  
  # Build a short tag for file naming
  scenario_tag <- sprintf(
    "N%d_V%d_trig%.2f_esc%.2f_bg%.3f",
    N_patients, N_visits, trigger_prop, escalation_prop, bg_noise_icd
  )
  
  if (verbose) cat("\n=== Running scenario:", scenario_tag, "===\n")
  
  # ---------------------------------------------------------------------------
  # 1. ARRAYS
  # ---------------------------------------------------------------------------
  ICD_arr <- array(0L, dim = c(N_patients, N_visits, N_ICD))
  ATC_arr <- array(0L, dim = c(N_patients, N_visits, N_ATC))
  
  # Background noise
  for (i in 1:N_patients) {
    for (t in 1:N_visits) {
      ICD_arr[i, t, ] <- rbinom(N_ICD, 1, bg_noise_icd)
      ATC_arr[i, t, ] <- rbinom(N_ATC, 1, bg_noise_atc)
    }
  }
  
  # ---------------------------------------------------------------------------
  # 2. PHENOTYPE INJECTION
  # ---------------------------------------------------------------------------
  
  # Trigger phenotype
  trigger_ids <- sample(1:N_patients, size = round(N_patients * trigger_prop))
  for (i in trigger_ids) {
    t_idx <- sample(1:min(10, N_visits - 1), 1)   # guard against short visit sequences
    ICD_arr[i, t_idx,     TRIGGER_ICD] <- 1L
    ATC_arr[i, t_idx + 1, TRIGGER_ATC] <- 1L
  }
  
  # Escalation phenotype
  escalation_ids <- sample(1:N_patients, size = round(N_patients * escalation_prop))
  late_idx <- (N_visits - ESCALATION_WINDOW + 1):N_visits
  for (i in escalation_ids) {
    for (t in late_idx) {
      ICD_arr[i, t, ] <- pmax(ICD_arr[i, t, ], rbinom(N_ICD, 1, escalation_noise))
    }
  }
  
  # ---------------------------------------------------------------------------
  # 3. STATIC & LAB DATA
  # ---------------------------------------------------------------------------
  static_bin  <- rbinom(N_patients, 1, 0.5)
  static_cont <- runif(N_patients, 18, 90) / 90
  lab_values  <- array(0, dim = c(N_patients, N_visits))
  for (i in 1:N_patients) {
    base_val   <- rnorm(1, 100, 10)
    trend      <- if (i %% 10 == 0) 1.5 else rnorm(1, 0, 0.5)
    lab_values[i, ] <- base_val + (1:N_visits * trend) + rnorm(N_visits, 0, 2)
  }
  
  # ---------------------------------------------------------------------------
  # 4. PHENOTYPE SCORES & GROUND-TRUTH ATTRIBUTION MAPS
  # ---------------------------------------------------------------------------
  phenotype_contributions <- data.frame(
    patient_id    = 1:N_patients,
    chronic_val   = 0,
    trigger_val   = 0,
    escalation_val = 0,
    static_val    = 0,
    trend_val     = 0
  )
  
  truth_map_icd <- array(0, dim = c(N_patients, N_visits, N_ICD))
  truth_map_atc <- array(0, dim = c(N_patients, N_visits, N_ATC))
  
  for (i in 1:N_patients) {
    
    # Chronic
    phenotype_contributions$chronic_val[i] <-
      sum(colSums(ICD_arr[i, , CHRONIC_ICDS, drop = FALSE])) * 0.5
    truth_map_icd[i, , CHRONIC_ICDS] <- ICD_arr[i, , CHRONIC_ICDS] * 0.25
    
    # Trigger
    t_icd <- which(ICD_arr[i, , TRIGGER_ICD] == 1)
    t_atc <- which(ATC_arr[i, , TRIGGER_ATC] == 1)
    if (length(t_icd) > 0 &&
        any(t_atc > min(t_icd) & t_atc <= (min(t_icd) + 2))) {
      phenotype_contributions$trigger_val[i] <- 10.0
      truth_map_icd[i, t_icd,                           TRIGGER_ICD] <- 1.0
      truth_map_atc[i, t_atc[t_atc > min(t_icd)],      TRIGGER_ATC] <- 1.0
    }
    
    # Escalation
    early_mean_rate <- sum(ICD_arr[i, 1:(N_visits - ESCALATION_WINDOW), ]) /
      (N_visits - ESCALATION_WINDOW)
    if (sum(ICD_arr[i, late_idx, ]) > (early_mean_rate * ESCALATION_WINDOW + 2)) {
      phenotype_contributions$escalation_val[i] <- 2.5
      truth_map_icd[i, late_idx, ] <- ICD_arr[i, late_idx, ] * 0.5
    }
    
    # Static & trend
    phenotype_contributions$static_val[i] <-
      (static_bin[i] * 0.8) + (static_cont[i] * 0.5)
    if (lab_values[i, N_visits] - lab_values[i, 1] > 10)
      phenotype_contributions$trend_val[i] <- 3.0
  }
  
  # ---------------------------------------------------------------------------
  # 5. OUTCOME GENERATION (calibrated to target prevalence)
  # ---------------------------------------------------------------------------
  total_risk_signal <- rowSums(phenotype_contributions[, -1])
  find_intercept    <- function(b) {
    mean(1 / (1 + exp(-(b + total_risk_signal)))) - target_prevalence
  }
  calibrated_intercept <- uniroot(find_intercept, interval = c(-50, 10))$root
  final_probs          <- 1 / (1 + exp(-(calibrated_intercept + total_risk_signal)))
  outcomes             <- rbinom(N_patients, 1, final_probs)
  
  if (verbose)
    cat(sprintf("  Prevalence: %.3f  (target: %.2f)\n", mean(outcomes), target_prevalence))
  
  # ---------------------------------------------------------------------------
  # 6. ASSEMBLE OUTPUT OBJECTS
  # ---------------------------------------------------------------------------
  
  # 6a. Events (long format)
  event_list <- list()
  for (i in 1:N_patients) {
    icd_pos <- which(ICD_arr[i, , ] == 1, arr.ind = TRUE)
    if (nrow(icd_pos) > 0)
      event_list[[length(event_list) + 1]] <-
        data.frame(patient_id = i, time = icd_pos[, 1],
                   token = paste0("ICD_", icd_pos[, 2]))
    
    atc_pos <- which(ATC_arr[i, , ] == 1, arr.ind = TRUE)
    if (nrow(atc_pos) > 0)
      event_list[[length(event_list) + 1]] <-
        data.frame(patient_id = i, time = atc_pos[, 1],
                   token = paste0("ATC_", atc_pos[, 2]))
  }
  events_df <- do.call(rbind, event_list)
  
  # 6b. Static
  static_df <- data.frame(
    patient_id  = 1:N_patients,
    static_bin  = static_bin,
    static_cont = static_cont,
    outcome     = outcomes
  )
  
  # 6c. Labs (long)
  lab_df <- melt(lab_values, varnames = c("patient_id", "time"))
  
  # 6d. Phenotype ground-truth flags
  pheno_gt <- data.frame(
    patient_id       = 1:N_patients,
    has_trigger_gt   = phenotype_contributions$trigger_val   > 0,
    has_escalation_gt = phenotype_contributions$escalation_val > 0,
    chronic_count    = apply(ICD_arr[, , CHRONIC_ICDS, drop = FALSE], 1, sum),
    has_chronic_gt   = phenotype_contributions$chronic_val   > 0
  )
  pheno_gt$group_gt <- with(pheno_gt,
    ifelse(has_trigger_gt & has_escalation_gt & has_chronic_gt, "all_three",
    ifelse(has_trigger_gt & has_escalation_gt,                  "trigger+escalation",
    ifelse(has_trigger_gt & has_chronic_gt,                     "trigger+chronic",
    ifelse(has_escalation_gt & has_chronic_gt,                  "escalation+chronic",
    ifelse(has_trigger_gt,                                      "trigger_only",
    ifelse(has_escalation_gt,                                   "escalation_only",
    ifelse(has_chronic_gt,                                      "chronic_only", "none")))))))
  )
  
  # ---------------------------------------------------------------------------
  # 7. SAVE FILES
  # ---------------------------------------------------------------------------
  if (save_files) {
    if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
    
    write.csv(events_df, file.path(output_dir, paste0("ehr_events_",       scenario_tag, ".csv")), row.names = FALSE)
    write.csv(static_df, file.path(output_dir, paste0("static_data_",      scenario_tag, ".csv")), row.names = FALSE)
    write.csv(lab_df,    file.path(output_dir, paste0("lab_data_",         scenario_tag, ".csv")), row.names = FALSE)
    write.csv(pheno_gt,  file.path(output_dir, paste0("phenotype_labels_gt_", scenario_tag, ".csv")), row.names = FALSE)
    saveRDS(
      list(truth_icd = truth_map_icd, truth_atc = truth_map_atc, outcomes = outcomes),
      file.path(output_dir, paste0("simulation_truth_", scenario_tag, ".rds"))
    )

    # ------------------------------------------------------------------
    # Export truth arrays as flat CSVs for Python evaluation
    # Each row = one (patient, visit) pair; columns = feature weights.
    # truth_icd_flat: N*T rows x N_ICD cols  (R column order: ICD_1..ICD_100)
    # truth_atc_flat: N*T rows x N_ATC cols  (R column order: ATC_1..ATC_40)
    # A companion index CSV carries the patient_id and visit index.
    # ------------------------------------------------------------------
    n_pv    <- N_patients * N_visits
    pv_idx  <- data.frame(
      patient_id = rep(1:N_patients, each = N_visits),
      visit      = rep(1:N_visits,   times = N_patients)
    )

    truth_icd_mat <- matrix(
      aperm(truth_map_icd, c(1, 2, 3)),   # keep N x T x F order then flatten
      nrow = n_pv, ncol = N_ICD
    )
    colnames(truth_icd_mat) <- paste0("ICD_", 1:N_ICD)

    truth_atc_mat <- matrix(
      aperm(truth_map_atc, c(1, 2, 3)),
      nrow = n_pv, ncol = N_ATC
    )
    colnames(truth_atc_mat) <- paste0("ATC_", 1:N_ATC)

    write.csv(
      cbind(pv_idx, truth_icd_mat),
      file.path(output_dir, paste0("truth_icd_flat_", scenario_tag, ".csv")),
      row.names = FALSE
    )
    write.csv(
      cbind(pv_idx, truth_atc_mat),
      file.path(output_dir, paste0("truth_atc_flat_", scenario_tag, ".csv")),
      row.names = FALSE
    )

    if (verbose) cat("  Files saved with tag:", scenario_tag, "\n")
  }
  
  # Return a named list invisibly for in-memory use
  invisible(list(
    scenario_tag            = scenario_tag,
    events                  = events_df,
    static                  = static_df,
    labs                    = lab_df,
    phenotype_labels_gt     = pheno_gt,
    phenotype_contributions = phenotype_contributions,
    truth_icd               = truth_map_icd,
    truth_atc               = truth_map_atc,
    outcomes                = outcomes,
    params = list(
      N_patients = N_patients, N_visits = N_visits,
      trigger_prop = trigger_prop, escalation_prop = escalation_prop,
      bg_noise_icd = bg_noise_icd, bg_noise_atc = bg_noise_atc
    )
  ))
}


# ==============================================================================
# SCENARIO GRID & BATCH RUNNER
# ==============================================================================

scenarios <- list(
  
  # --- Baseline (matches your original script exactly) ---
  baseline = list(
    N_patients = 3000, N_visits = 20,
    trigger_prop = 0.15, escalation_prop = 0.10,
    bg_noise_icd = 0.01, bg_noise_atc = 0.01, escalation_noise = 0.08
  ),
  
  # --- Vary N ---
  small_N = list(
    N_patients = 500,  N_visits = 20,
    trigger_prop = 0.15, escalation_prop = 0.10,
    bg_noise_icd = 0.01, bg_noise_atc = 0.01, escalation_noise = 0.08
  ),
  large_N = list(
    N_patients = 10000, N_visits = 20,
    trigger_prop = 0.15, escalation_prop = 0.10,
    bg_noise_icd = 0.01, bg_noise_atc = 0.01, escalation_noise = 0.08
  ),
  
  # --- Vary trigger prevalence ---
  rare_trigger = list(
    N_patients = 3000, N_visits = 20,
    trigger_prop = 0.05, escalation_prop = 0.10,
    bg_noise_icd = 0.01, bg_noise_atc = 0.01, escalation_noise = 0.08
  ),
  very_rare_trigger = list(
    N_patients = 3000, N_visits = 20,
    trigger_prop = 0.02, escalation_prop = 0.10,
    bg_noise_icd = 0.01, bg_noise_atc = 0.01, escalation_noise = 0.08
  ),
  
  # --- Vary visit length ---
  short_history = list(
    N_patients = 3000, N_visits = 10,
    trigger_prop = 0.15, escalation_prop = 0.10,
    bg_noise_icd = 0.01, bg_noise_atc = 0.01, escalation_noise = 0.08
  ),
  long_history = list(
    N_patients = 3000, N_visits = 50,
    trigger_prop = 0.15, escalation_prop = 0.10,
    bg_noise_icd = 0.01, bg_noise_atc = 0.01, escalation_noise = 0.08
  ),
  
  # --- Vary background noise ---
  high_noise = list(
    N_patients = 3000, N_visits = 20,
    trigger_prop = 0.15, escalation_prop = 0.10,
    bg_noise_icd = 0.05, bg_noise_atc = 0.05, escalation_noise = 0.08
  ),
  very_high_noise = list(
    N_patients = 3000, N_visits = 20,
    trigger_prop = 0.15, escalation_prop = 0.10,
    bg_noise_icd = 0.10, bg_noise_atc = 0.10, escalation_noise = 0.08
  ),
  
  # --- Combined "stress test" scenario ---
  stress_test = list(
    N_patients = 5000, N_visits = 30,
    trigger_prop = 0.05, escalation_prop = 0.10,
    bg_noise_icd = 0.05, bg_noise_atc = 0.05, escalation_noise = 0.08
  )
)


# Run all scenarios and collect results in a named list
run_all_scenarios <- function(scenarios, output_dir = "sim_outputs", ...) {
  results <- vector("list", length(scenarios))
  names(results) <- names(scenarios)
  
  for (nm in names(scenarios)) {
    cat("\n[", nm, "]\n", sep = "")
    results[[nm]] <- do.call(
      simulate_ehr,
      c(scenarios[[nm]], list(output_dir = output_dir, ...))
    )
  }
  
  invisible(results)
}

# ==============================================================================
# RUN
# ==============================================================================

all_results <- run_all_scenarios(scenarios, output_dir = "sim_outputs")


# ==============================================================================
# QUICK SUMMARY TABLE
# ==============================================================================

summary_tbl <- do.call(rbind, lapply(all_results, function(r) {
  pheno <- r$phenotype_labels_gt
  data.frame(
    scenario        = r$scenario_tag,
    N               = r$params$N_patients,
    visits          = r$params$N_visits,
    trigger_prop    = r$params$trigger_prop,
    bg_noise        = r$params$bg_noise_icd,
    prevalence      = round(mean(r$outcomes), 3),
    pct_trigger_gt  = round(mean(pheno$has_trigger_gt),    3),
    pct_escalation_gt = round(mean(pheno$has_escalation_gt), 3),
    pct_chronic_gt  = round(mean(pheno$has_chronic_gt),    3)
  )
}))

print(summary_tbl, row.names = FALSE)
