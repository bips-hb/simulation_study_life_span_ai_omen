#===============================================================================
# Longitudinal EHR Simulation (array-based) tuned for XAI benchmarking
#
# What you get:
# 1) ICD_arr: [N_patients x N_visits x N_ICD] (0/1)
# 2) ATC_arr: [N_patients x N_visits x N_ATC] (0/1)
# 3) Wide CSV for classic ML baselines (LR/RF): ehr_wide.csv
# 4) Long/token CSV for DL (Transformers/RNNs): ehr_events.csv with (patient_id,time,token)
# 5) Truth RDS with:
#    - generative params and true coefficients
#    - index sets (causal/protective/window/incident)
#    - per-patient per-time per-code contributions (code x time attribution ground truth)
#
# What is done:
# - Realistic-ish correlation: shared latent factors drive code co-occurrence
# - Temporal dependence: persistence on logit scale
# - Proxy structure: ICD->ATC linkage (confounding by indication / treatment follows diagnosis)
# - Outcome with known truth: risk + protective + time windows + recency + incident (first-ever) + trend
# - Calibrated prevalence
#===============================================================================

# Goal:
# Simulate patients who have unobserved conditions, which cause diagnosis and
# medication codes to appear over time. Then define a cancer outvcome that depends
# on whch codes appeared, when they appeared, and whether they were risk or pro-
# tective, with known ground truth.


# Structure:
# Latent patient state
# ->
# Observed longitudinal codes (ICD, ATC)
# -> 
# Temporal patterns (persistence, windows, incidents)
# -> 
# Outcome (cancer, binary)
# -> 
# Ground truth explanation map



# Set seed for reproducibility
set.seed(123)

#...............................................................................
# 1. Parameters 
#...............................................................................

# Number of patients and number of visits
N_patients <- 3000
N_visits   <- 20

# Number of diagnosis (ICD) and medication (ATC) codes
N_ICD <- 200   
N_ATC <- 60    

# Number of shared factors
N_factors <- 5
# They can for example each refer to one type of condition (e.g., 
# metabolic syndrome, chronic inflammation, cardiovascular fragility, oncologic 
# susceptibility, multimorbidity etc.)
# -> They can affect many ICD and ATC codes

# Expected number of codes per visit
lambda_codes_icd <- 6
lambda_codes_atc <- 3

# Temporal dependence strength: if code present at t-1, add delta to logit at t
# This creates persistence without probability clipping artifacts.
# NOTE: We slightly reduce persistence so "timing" signals do not leak too strongly into total counts.
delta_persist_icd <- 0.6
delta_persist_atc <- 0.5

# ICD->ATC linkage: meds more likely if linked dx present (proxy structure)
use_icd_to_atc_link <- TRUE
icd_to_atc_strength <- 1.0  # add on logit scale if any linked ICD present at that visit

# Latent factor prevalence: each factor present in ~15% patients
factor_prev <- 0.15

# Baseline prevalence heterogeneity (long-tail-ish): code intercepts on logit scale
# More negative => rarer codes
icd_intercept_mu <- -3.0
icd_intercept_sd <- 1.0
atc_intercept_mu <- -3.2
atc_intercept_sd <- 0.9

# Recency weights: later visits matter more for outcome (temporal ground truth)
time_weights <- seq(0.5, 1.5, length.out = N_visits)

# Target outcome prevalence (calibrated via intercept)
target_prev <- 0.30

# Trend effect: increasing total burden increases risk
trend_strength <- 0.10  # tune 0.05–0.20

# Output paths
out_wide   <- "ehr_wide.csv"
out_events <- "ehr_events.csv"
out_truth  <- "ehr_simulation_truth.rds"


#...............................................................................
# 2. Latent patient factors (unobserved ground truth "conditions")
#...............................................................................

F_matrix <- matrix(
  rbinom(N_patients * N_factors, 1, factor_prev),
  nrow = N_patients,
  ncol = N_factors
)
# Why:
# - Shared latent factors create correlated code patterns (more realistic than i.i.d. codes)
# - Also create unobserved confounding for outcome vs observed codes


#...............................................................................
# 3. Static covariates (partly correlated with latent factors)
#...............................................................................

# Start with base
sex <- rbinom(N_patients, 1, 0.5)               # 0/1
baseline_risk <- runif(N_patients, -1, 1)       # continuous

# Induce correlation with latent factors to create realistic confounding
# Example: factor 1 increases probability of sex=1
sex_logit <- qlogis(0.5) + 0.8 * F_matrix[, 1]
sex <- rbinom(N_patients, 1, plogis(sex_logit))

# Factors 2 and 4 increase baseline risk
baseline_risk <- baseline_risk + 0.4 * F_matrix[, 2] + 0.3 * F_matrix[, 4]


#...............................................................................
# 4. Code intercepts and factor -> code loadings (drive code probabilities)
#...............................................................................

# Each code gets its own baseline logit (heterogeneous prevalence)
b_ICD <- rnorm(N_ICD, icd_intercept_mu, icd_intercept_sd)
b_ATC <- rnorm(N_ATC, atc_intercept_mu, atc_intercept_sd)

# Factor->code weights (logit additive); positive values mean factor increases code prob
W_ICD <- matrix(runif(N_factors * N_ICD, 0.4, 1.2), nrow = N_factors, ncol = N_ICD)
W_ATC <- matrix(runif(N_factors * N_ATC, 0.4, 1.2), nrow = N_factors, ncol = N_ATC)

# ICD->ATC sparse mapping: each ATC linked to 0-2 ICD codes
# Why: creates a plausible "treatment follows diagnosis" pattern and proxy structure for XAI
icd_to_atc_map <- vector("list", N_ATC)
if (use_icd_to_atc_link) {
  for (m in 1:N_ATC) {
    k <- sample(0:2, 1, prob = c(0.35, 0.45, 0.20))
    icd_to_atc_map[[m]] <- if (k == 0) integer(0) else sample(1:N_ICD, k)
  }
}


#...............................................................................
# 5. Outcome ground truth: risk/protective subsets + time-window effects
#...............................................................................

# Latent factor effects on outcome (true but unobserved)
beta_factors <- runif(N_factors, 0.3, 0.8)

# Static effects
beta_0    <- -4.0  # placeholder; calibrated later
beta_sex  <-  0.4
beta_risk <-  0.8

# Baseline small effects for all codes (so burden has mild relevance)
beta_icd <- rep(0.01, N_ICD)
beta_atc <- rep(0.015, N_ATC)

# Inject strong causal + protective subsets (core XAI target: identify which codes, sign)
n_causal_icd <- 10
n_prot_icd   <- 6
n_causal_atc <- 6
n_prot_atc   <- 4

causal_icd_idx <- sample(1:N_ICD, n_causal_icd)
prot_icd_idx   <- sample(setdiff(1:N_ICD, causal_icd_idx), n_prot_icd)

causal_atc_idx <- sample(1:N_ATC, n_causal_atc)
prot_atc_idx   <- sample(setdiff(1:N_ATC, causal_atc_idx), n_prot_atc)

beta_icd[causal_icd_idx] <- runif(n_causal_icd, 0.10, 0.20)
beta_icd[prot_icd_idx]   <- -runif(n_prot_icd, 0.08, 0.15)

beta_atc[causal_atc_idx] <- runif(n_causal_atc, 0.12, 0.25)
beta_atc[prot_atc_idx]   <- -runif(n_prot_atc, 0.08, 0.18)

# Time-window effects:
# - some codes matter ONLY late (last 2 visits)
# - some codes matter ONLY early (first 2 visits)
late_window  <- (N_visits - 1):N_visits
early_window <- 1:2

late_icd_idx  <- sample(setdiff(1:N_ICD, c(causal_icd_idx, prot_icd_idx)), 3)
early_atc_idx <- sample(setdiff(1:N_ATC, c(causal_atc_idx, prot_atc_idx)), 3)

icd_window_mult <- matrix(1, nrow = N_visits, ncol = N_ICD)
atc_window_mult <- matrix(1, nrow = N_visits, ncol = N_ATC)

# outside window => multiplier 0, inside => 1
icd_window_mult[, late_icd_idx] <- 0
icd_window_mult[late_window, late_icd_idx] <- 1

atc_window_mult[, early_atc_idx] <- 0
atc_window_mult[early_window, early_atc_idx] <- 1

# Sequence-only "incident" effect (TRUE first-ever occurrence):
# Choose codes where first appearance increases risk more than repeated presence.
n_incident_icd <- 4
n_incident_atc <- 3

incident_icd_idx <- sample(setdiff(1:N_ICD,
                                   c(causal_icd_idx, prot_icd_idx, late_icd_idx)),
                           n_incident_icd)
incident_atc_idx <- sample(setdiff(1:N_ATC,
                                   c(causal_atc_idx, prot_atc_idx, early_atc_idx)),
                           n_incident_atc)

beta_incident_icd <- rep(0, N_ICD)
beta_incident_atc <- rep(0, N_ATC)

# NOTE:
# These incident coefficients are kept strong enough to be an XAI target, but not so large that
# count-based tabular baselines dominate via simple "ever happened" proxies.
# In addition, we will make incident codes "acute": they do NOT persist (see section 6b below),
# and we will apply incident effects only in the late window (see section 7).
beta_incident_icd[incident_icd_idx] <- runif(n_incident_icd, 0.20, 0.45)
beta_incident_atc[incident_atc_idx] <- runif(n_incident_atc, 0.20, 0.50)

# Order rule: ICD_A followed by ATC_B within a short horizon increases risk
# Why:
# - Captures progression/escalation logic ("dx then treatment").
# - Order information is not represented in simple sum-over-time tabular baselines.
n_order_pairs <- 6
order_horizon <- 2  # ATC must occur within next 1..order_horizon visits after ICD
order_pairs <- data.frame(
  icd = sample(1:N_ICD, n_order_pairs),
  atc = sample(1:N_ATC, n_order_pairs)
)

# Effect size: keep moderate to preserve story (B)
beta_order_pairs <- runif(n_order_pairs, 0.25, 0.50)

# Order strength: can be used to scale all order effects together
order_strength <- 1.0

# Escalation strength: keep small to preserve story (B)
# (We define escalation_effect later in section 7, once we have the realized sequences.)
escalation_strength <- 0.25  # tune 0.10–0.40


#...............................................................................
# 6. Core longitudinal generation into arrays (fast + clean)
#...............................................................................

# Allocate arrays: [patient x time x code]
ICD_arr <- array(0L, dim = c(N_patients, N_visits, N_ICD))
ATC_arr <- array(0L, dim = c(N_patients, N_visits, N_ATC))

# For "incident" computation we will track whether each code was ever seen before time t
ever_icd <- matrix(0L, nrow = N_patients, ncol = N_ICD)
ever_atc <- matrix(0L, nrow = N_patients, ncol = N_ATC)

# Helper notes:
# We want sparse visits: we sample K codes without replacement using weights proportional to probabilities.
# This mimics a visit with a limited number of recorded codes.
# Doing it per patient keeps it simple and interpretable.

for (t in 1:N_visits) {
  for (i in 1:N_patients) {
    
    # ---- 6a) Build base logits from intercepts + latent factors ----
    # logit = b + F_i %*% W
    logit_icd <- b_ICD + as.numeric(F_matrix[i, ] %*% W_ICD)
    logit_atc <- b_ATC + as.numeric(F_matrix[i, ] %*% W_ATC)
    
    # ---- 6b) Add persistence on logit scale using previous visit ----
    #
    # IMPORTANT CHANGE (to keep "incident" closer to sequence-only information):
    # - For most codes, persistence applies as before.
    # - For incident codes, we REMOVE persistence so they behave more like "acute one-off events".
    #
    # Why:
    # If incident codes persist, then "first-ever occurrence" becomes highly correlated with total counts.
    # Count-based tabular baselines (LR/RF on summed codes) can then pick up incident signal easily.
    # Removing persistence for incident codes reduces this leakage and makes timing information more important.
    if (t > 1) {
      prev_icd <- ICD_arr[i, t - 1, ]
      prev_atc <- ATC_arr[i, t - 1, ]
      
      # Remove persistence for incident codes (acute behavior)
      prev_icd_adj <- prev_icd
      prev_atc_adj <- prev_atc
      prev_icd_adj[incident_icd_idx] <- 0L
      prev_atc_adj[incident_atc_idx] <- 0L
      
      logit_icd <- logit_icd + delta_persist_icd * prev_icd_adj
      logit_atc <- logit_atc + delta_persist_atc * prev_atc_adj
    }
    
    # Convert logits to probabilities (for weighting selection)
    p_icd <- plogis(logit_icd)
    p_atc <- plogis(logit_atc)
    
    # ---- 6c) Decide how many codes appear this visit (sparsity control) ----
    K_icd <- rpois(1, lambda_codes_icd)
    K_atc <- rpois(1, lambda_codes_atc)
    K_icd <- max(0, min(K_icd, N_ICD))
    K_atc <- max(0, min(K_atc, N_ATC))
    
    # ---- 6d) Sample ICD codes without replacement, weighted by p_icd ----
    if (K_icd > 0) {
      idx_icd <- sample(1:N_ICD, size = K_icd, replace = FALSE, prob = p_icd + 1e-8)
      ICD_arr[i, t, idx_icd] <- 1L
    }
    
    # ---- 6e) Sample ATC codes with optional ICD->ATC linkage ----
    if (K_atc > 0) {
      if (use_icd_to_atc_link) {
        # Boost ATC logits if any linked ICD present at this visit
        boost <- rep(0, N_ATC)
        for (m in 1:N_ATC) {
          linked <- icd_to_atc_map[[m]]
          if (length(linked) > 0) {
            boost[m] <- icd_to_atc_strength * as.numeric(any(ICD_arr[i, t, linked] == 1L))
          }
        }
        p_atc2 <- plogis(qlogis(p_atc) + boost)
        idx_atc <- sample(1:N_ATC, size = K_atc, replace = FALSE, prob = p_atc2 + 1e-8)
        ATC_arr[i, t, idx_atc] <- 1L
      } else {
        idx_atc <- sample(1:N_ATC, size = K_atc, replace = FALSE, prob = p_atc + 1e-8)
        ATC_arr[i, t, idx_atc] <- 1L
      }
    }
    
    # Update "ever seen" trackers for true first-ever incident effect
    ever_icd[i, ] <- pmax(ever_icd[i, ], ICD_arr[i, t, ])
    ever_atc[i, ] <- pmax(ever_atc[i, ], ATC_arr[i, t, ])
  }
}


#...............................................................................
# 7. Compute outcome components + per-patient per-time true contributions
#...............................................................................

# Latent + static components (not time-resolved)
latent_effect <- as.numeric(F_matrix %*% beta_factors)
static_effect <- beta_sex * sex + beta_risk * baseline_risk

# We will compute:
# - total code_effect per patient
# - plus "truth maps" that allow objective XAI evaluation:
#   contrib_icd[i,t,j] = time_weights[t] * X * beta_icd[j] * window_mult[t,j]
#   contrib_atc likewise
#
# NOTE: These truth maps can be large. With N=3000, T=20, ICD=200 => 12M entries (OK).
# If you scale ICD to 5000, store truth in sparse/long form instead.

contrib_icd <- array(0, dim = c(N_patients, N_visits, N_ICD))
contrib_atc <- array(0, dim = c(N_patients, N_visits, N_ATC))

# Main (time-weighted, windowed) contributions
for (t in 1:N_visits) {
  # Vectorized over patients for speed
  # ICD_arr[,t,] is [N x ICD]
  X_icd_t <- ICD_arr[, t, , drop = FALSE]
  X_atc_t <- ATC_arr[, t, , drop = FALSE]
  
  # per-code coefficient at this time includes window multiplier
  coef_icd_t <- beta_icd * icd_window_mult[t, ]
  coef_atc_t <- beta_atc * atc_window_mult[t, ]
  
  # Broadcast: (N x codes) * (codes) => (N x codes)
  # then multiply by time weight
  contrib_icd[, t, ] <- time_weights[t] * (X_icd_t[, 1, ] * rep(coef_icd_t, each = N_patients))
  contrib_atc[, t, ] <- time_weights[t] * (X_atc_t[, 1, ] * rep(coef_atc_t, each = N_patients))
}

# Incident effect (TRUE first-ever occurrence):
# incident at time t means present at t AND absent in any previous visits
# We'll compute incident indicators using cumulative history.
incident_contrib_icd <- array(0, dim = c(N_patients, N_visits, N_ICD))
incident_contrib_atc <- array(0, dim = c(N_patients, N_visits, N_ATC))

# Track history up to t-1
seen_before_icd <- matrix(0L, nrow = N_patients, ncol = N_ICD)
seen_before_atc <- matrix(0L, nrow = N_patients, ncol = N_ATC)

for (t in 1:N_visits) {
  X_icd_t <- ICD_arr[, t, , drop = FALSE][, 1, ]
  X_atc_t <- ATC_arr[, t, , drop = FALSE][, 1, ]
  
  inc_icd <- (X_icd_t == 1L) & (seen_before_icd == 0L)
  inc_atc <- (X_atc_t == 1L) & (seen_before_atc == 0L)
  
  # contribution = time_weight[t] * incident_indicator * beta_incident
  # IMPORTANT CHANGE:
  # Incident effects are applied ONLY if the first-ever occurrence happens in the late window.
  # Why:
  # - "Ever happened" is easy for tabular baselines to exploit.
  # - "First happened late" is genuinely temporal and gives sequence models + XAI clear targets.
  in_late_window <- as.numeric(t %in% late_window)
  
  if (any(beta_incident_icd != 0)) {
    incident_contrib_icd[, t, ] <- in_late_window * time_weights[t] *
      (inc_icd * rep(beta_incident_icd, each = N_patients))
  }
  if (any(beta_incident_atc != 0)) {
    incident_contrib_atc[, t, ] <- in_late_window * time_weights[t] *
      (inc_atc * rep(beta_incident_atc, each = N_patients))
  }
  
  # update history for next time step
  seen_before_icd <- pmax(seen_before_icd, X_icd_t)
  seen_before_atc <- pmax(seen_before_atc, X_atc_t)
}

# Trend effect: increasing burden over time increases risk
trend_effect <- rep(0, N_patients)
burden_prev <- rep(0, N_patients)

for (t in 1:N_visits) {
  burden_t <- rowSums(ICD_arr[, t, , drop = FALSE][, 1, ]) +
    rowSums(ATC_arr[, t, , drop = FALSE][, 1, ])
  delta_burden <- pmax(0, burden_t - burden_prev)
  trend_effect <- trend_effect + time_weights[t] * delta_burden
  burden_prev <- burden_t
}

# Escalation effect: consecutive worsening increases risk (sequence-only pattern)
# Why:
# - Total counts do not encode consecutive increases well.
# - Sequence models can learn "worsening trajectory" more naturally.
escalation_effect <- rep(0, N_patients)

# Define escalation as: burden(t-2) < burden(t-1) < burden(t)
# We weight escalation by the time weight at t (so late escalation matters more).
if (N_visits >= 3) {
  burden <- matrix(0, nrow = N_patients, ncol = N_visits)
  for (tt in 1:N_visits) {
    burden[, tt] <- rowSums(ICD_arr[, tt, , drop = FALSE][, 1, ]) +
      rowSums(ATC_arr[, tt, , drop = FALSE][, 1, ])
  }
  for (tt in 3:N_visits) {
    esc_t <- (burden[, tt-2] < burden[, tt-1]) & (burden[, tt-1] < burden[, tt])
    escalation_effect <- escalation_effect + time_weights[tt] * as.numeric(esc_t)
  }
}

# Order effect contributions (ICD_A at t AND ATC_B occurs within next 1..H visits)
# We compute an explicit per-time contribution map to use as ground truth for temporal XAI.
order_contrib <- array(0, dim = c(N_patients, N_visits, n_order_pairs))
order_effect <- rep(0, N_patients)

if (n_order_pairs > 0) {
  for (p in 1:n_order_pairs) {
    icdA <- order_pairs$icd[p]
    atcB <- order_pairs$atc[p]
    beta_p <- beta_order_pairs[p]
    
    for (t0 in 1:N_visits) {
      # ICD_A present at time t0
      hasA <- ICD_arr[, t0, icdA] == 1L
      
      # ATC_B occurs in (t0+1 ... t0+H) (bounded by N_visits)
      t2_max <- min(N_visits, t0 + order_horizon)
      if (t2_max >= t0 + 1) {
        hasB_future <- rep(FALSE, N_patients)
        for (tt in (t0 + 1):t2_max) {
          hasB_future <- hasB_future | (ATC_arr[, tt, atcB] == 1L)
        }
      } else {
        hasB_future <- rep(FALSE, N_patients)
      }
      
      rule_t <- hasA & hasB_future
      
      # Attribute this effect to the time t0 where the triggering ICD_A occurs
      # (so XAI should highlight visit t0 for ICD_A).
      order_contrib[, t0, p] <- time_weights[t0] * beta_p * as.numeric(rule_t)
    }
  }
  
  # Total order effect per patient
  order_effect <- rowSums(matrix(order_contrib, nrow = N_patients))
}

order_effect <- order_strength * order_effect


# Total code-related effects
code_effect_main <- rowSums(matrix(contrib_icd, nrow = N_patients)) +
  rowSums(matrix(contrib_atc, nrow = N_patients))
code_effect_inc  <- rowSums(matrix(incident_contrib_icd, nrow = N_patients)) +
  rowSums(matrix(incident_contrib_atc, nrow = N_patients))

code_effect_trend <- trend_strength * trend_effect
code_effect_escalation <- escalation_strength * escalation_effect
code_effect_order <- order_effect

code_effect_total <- code_effect_main + code_effect_inc + code_effect_trend + code_effect_escalation + code_effect_order


#...............................................................................
# 8. Calibrate intercept to hit target prevalence; sample outcome
#...............................................................................

base_lp <- latent_effect + static_effect + code_effect_total

# Solve beta_0 so mean(sigmoid(beta_0 + base_lp)) = target_prev
beta_0 <- uniroot(
  function(b0) mean(plogis(b0 + base_lp)) - target_prev,
  interval = c(-30, 10)
)$root

linear_pred <- beta_0 + base_lp
prob <- plogis(linear_pred)
outcome <- rbinom(N_patients, 1, prob)

cat("Calibrated beta_0:", beta_0,
    "| Target prev:", target_prev,
    "| Realized prev:", mean(outcome), "\n")


#...............................................................................
# 9. Build outputs: wide + token/long format
#...............................................................................

# ---- 9a) Wide dataset (classic ML friendly)
wide <- data.frame(
  patient_id = 1:N_patients,
  sex = sex,
  baseline_risk = baseline_risk,
  outcome = outcome
)

# Add code columns visit by visit
# Naming convention matches your earlier approach: ICD{j}_t{t}, ATC{m}_t{t}
for (t in 1:N_visits) {
  ICD_mat <- ICD_arr[, t, , drop = FALSE][, 1, ]
  ATC_mat <- ATC_arr[, t, , drop = FALSE][, 1, ]
  
  colnames(ICD_mat) <- paste0("ICD", 1:N_ICD, "_t", t)
  colnames(ATC_mat) <- paste0("ATC", 1:N_ATC, "_t", t)
  
  wide <- cbind(wide, ICD_mat, ATC_mat)
}

write.csv(wide, out_wide, row.names = FALSE)
cat("Wrote wide CSV:", out_wide, "\n")


# ---- 9b) Long/token events (DL friendly)
# Output columns: patient_id, time, token
# Token format: "ICD_17" or "ATC_3"
#
# Why this format:
# - compact even when codes are high-dimensional
# - easy to build sequences + masks in Python
# - easy to add SEP tokens later if you want explicit visit boundaries

event_pid <- integer(0)
event_time <- integer(0)
event_token <- character(0)

for (t in 1:N_visits) {
  # ICD events
  idx_icd <- which(ICD_arr[, t, , drop = FALSE][, 1, ] == 1L, arr.ind = TRUE)
  # idx_icd columns: row = patient, col = code
  if (nrow(idx_icd) > 0) {
    event_pid   <- c(event_pid, idx_icd[, 1])
    event_time  <- c(event_time, rep.int(t, nrow(idx_icd)))
    event_token <- c(event_token, paste0("ICD_", idx_icd[, 2]))
  }
  
  # ATC events
  idx_atc <- which(ATC_arr[, t, , drop = FALSE][, 1, ] == 1L, arr.ind = TRUE)
  if (nrow(idx_atc) > 0) {
    event_pid   <- c(event_pid, idx_atc[, 1])
    event_time  <- c(event_time, rep.int(t, nrow(idx_atc)))
    event_token <- c(event_token, paste0("ATC_", idx_atc[, 2]))
  }
}

events <- data.frame(
  patient_id = event_pid,
  time = event_time,
  token = event_token,
  stringsAsFactors = FALSE
)

write.csv(events, out_events, row.names = FALSE)
cat("Wrote events CSV:", out_events, "\n")


#...............................................................................
# 10. Sanity checks (quick)
#...............................................................................

cat("Outcome prevalence:", mean(outcome), "\n")
cat("Outcome counts:\n"); print(table(outcome))

# Simple association with total burden (should be >0, but not perfect)
total_icd <- rowSums(ICD_arr[, , , drop = FALSE])
total_atc <- rowSums(ATC_arr[, , , drop = FALSE])
cat("Cor(outcome, total ICD):", cor(outcome, total_icd), "\n")
cat("Cor(outcome, total ATC):", cor(outcome, total_atc), "\n")

# Check that causal contributions correlate positively and protective negatively
# (using true contribution maps)
causal_score <- rep(0, N_patients)
prot_score <- rep(0, N_patients)
for (t in 1:N_visits) {
  X_icd_t <- ICD_arr[, t, , drop = FALSE][, 1, ]
  X_atc_t <- ATC_arr[, t, , drop = FALSE][, 1, ]
  causal_score <- causal_score +
    time_weights[t] * (
      rowSums(X_icd_t[, causal_icd_idx, drop=FALSE]) +
        rowSums(X_atc_t[, causal_atc_idx, drop=FALSE])
    )
  prot_score <- prot_score +
    time_weights[t] * (
      rowSums(X_icd_t[, prot_icd_idx, drop=FALSE]) +
        rowSums(X_atc_t[, prot_atc_idx, drop=FALSE])
    )
}
cat("Cor(outcome, causal_score):", cor(outcome, causal_score), "\n")
cat("Cor(outcome, prot_score):",  cor(outcome, prot_score),  "\n")


#...............................................................................
# 11. Save truth object (parameters + realized attribution ground truth)
#...............................................................................

truth <- list(
  meta = list(
    n_patients = N_patients,
    n_visits   = N_visits,
    n_icd      = N_ICD,
    n_atc      = N_ATC,
    n_factors  = N_factors,
    time_weights = time_weights,
    lambda_codes_icd = lambda_codes_icd,
    lambda_codes_atc = lambda_codes_atc,
    delta_persist_icd = delta_persist_icd,
    delta_persist_atc = delta_persist_atc,
    use_icd_to_atc_link = use_icd_to_atc_link,
    icd_to_atc_strength = icd_to_atc_strength,
    target_prev = target_prev,
    trend_strength = trend_strength,
    escalation_strength = escalation_strength,
    order_strength = order_strength,
    order_horizon = order_horizon,
    n_order_pairs = n_order_pairs
  ),
  ground_truth = list(
    beta_0 = beta_0,
    beta_factors = beta_factors,
    beta_sex = beta_sex,
    beta_risk = beta_risk,
    beta_icd = beta_icd,
    beta_atc = beta_atc,
    causal_icd_idx = causal_icd_idx,
    prot_icd_idx = prot_icd_idx,
    causal_atc_idx = causal_atc_idx,
    prot_atc_idx = prot_atc_idx,
    late_icd_idx = late_icd_idx,
    early_atc_idx = early_atc_idx,
    icd_window_mult = icd_window_mult,
    atc_window_mult = atc_window_mult,
    incident_icd_idx = incident_icd_idx,
    incident_atc_idx = incident_atc_idx,
    beta_incident_icd = beta_incident_icd,
    beta_incident_atc = beta_incident_atc,
    order_pairs = order_pairs,
    beta_order_pairs = beta_order_pairs
  ),
  generative_params = list(
    F_matrix = F_matrix,
    b_ICD = b_ICD,
    b_ATC = b_ATC,
    W_ICD = W_ICD,
    W_ATC = W_ATC,
    icd_to_atc_map = if (use_icd_to_atc_link) icd_to_atc_map else NULL
  ),
  realized_truth_maps = list(
    # These are the *exact additive contributions* used in the data generating model.
    # For XAI scoring you can compare model attributions to these tensors
    # (aggregate per code, per time, signed, etc.).
    contrib_icd = contrib_icd,
    contrib_atc = contrib_atc,
    incident_contrib_icd = incident_contrib_icd,
    incident_contrib_atc = incident_contrib_atc,
    # components for completeness
    latent_effect = latent_effect,
    static_effect = static_effect,
    trend_effect = trend_effect,
    escalation_effect = escalation_effect,
    order_contrib = order_contrib,
    order_effect = order_effect
  )
)

saveRDS(truth, out_truth)
cat("Saved truth RDS:", out_truth, "\n")
