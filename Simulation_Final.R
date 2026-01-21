# ==========================================
# Simulate longitudinal sparse EHR tokens + export
# Outputs: events.csv, labels.csv, ground_truth.json
# ==========================================

set.seed(42)

# ---- Settings ----
out_dir <- "sim_out"
dir.create(out_dir, showWarnings = FALSE)

n <- 4000          # patients
T <- 20            # visits
D <- 3000          # diagnosis codes
M <- 1200          # medication codes

k_diag <- 6        # avg diag tokens per visit (approx)
k_med  <- 2        # avg med tokens per visit (approx)

p_persist <- 0.6   # chronic persistence
noise_diag <- 1    # irrelevant diag per visit
noise_med  <- 0    # irrelevant med per visit

# Outcome model knobs
n_risk_diag <- 20
n_prot_diag <- 15
n_risk_med  <- 10
n_prot_med  <- 8

# Recency weighting: later visits matter more
rec_w <- seq(0.2, 1.0, length.out = T)

# Temporal motif: A then B within window -> increases risk
window <- 2
theta_motif <- 1.5

# Optional protective motif: med P present >= 3 consecutive visits -> reduces risk
theta_prot_run <- -1.0
prot_run_len <- 3

# Control prevalence with intercept (tune later if needed)
intercept <- -2.3

mislabel <- 0.02   # flip fraction of positives to 0 (set 0 to disable)

# ---- Packages ----
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
library(jsonlite)

# ---- Choose ground-truth codes (small subset truly matters) ----
risk_diag <- sample.int(D, n_risk_diag)
prot_diag <- sample(setdiff(1:D, risk_diag), n_prot_diag)

risk_med  <- sample.int(M, n_risk_med)
prot_med  <- sample(setdiff(1:M, risk_med), n_prot_med)

# weights
w_risk_diag <- runif(n_risk_diag, 0.3, 1.0)
w_prot_diag <- -runif(n_prot_diag, 0.3, 1.0)
w_risk_med  <- runif(n_risk_med, 0.2, 0.8)
w_prot_med  <- -runif(n_prot_med, 0.2, 0.8)

# ---- Temporal motif codes ----
A_diag <- sample(setdiff(1:D, c(risk_diag, prot_diag)), 1)
B_med  <- sample(setdiff(1:M, c(risk_med, prot_med)), 1)

# Protective run code (med)
P_med <- sample(setdiff(1:M, c(risk_med, prot_med, B_med)), 1)

cat("Ground truth:\n")
cat("  Motif risk: D_", A_diag, " then M_", B_med, " within ", window, " visits\n", sep="")
cat("  Motif protective: M_", P_med, " for >= ", prot_run_len, " consecutive visits\n\n", sep="")

# ---- Storage: long events ----
events_patient <- integer(0)
events_time <- integer(0)
events_token <- character(0)

# For ground-truth evaluation: per-patient trigger times
motif_trigger_time <- rep(NA_integer_, n)  # time when A occurs that later leads to B in window
prot_run_start_time <- rep(NA_integer_, n) # start of first protective run

# Track per-patient chronic pool (diagnoses)
chronic_pool <- vector("list", n)
for (i in 1:n) chronic_pool[[i]] <- integer(0)

# We also simulate a simple diagnosis->med “policy” to create confounding:
# If a patient has A_diag at time t, B_med becomes more likely at t+1..t+window
p_policy <- 0.5

# Keep a per-patient "policy window" for B_med
policy_until <- rep(0L, n)

# Also record per-patient med presence for run detection
P_present <- matrix(FALSE, nrow = n, ncol = T)

# ---- Generate sequences ----
# We keep boolean matrices only for the few ground-truth codes to score risk fast
has_risk_diag <- matrix(FALSE, nrow = n, ncol = T)
has_prot_diag <- matrix(FALSE, nrow = n, ncol = T)
has_risk_med  <- matrix(FALSE, nrow = n, ncol = T)
has_prot_med  <- matrix(FALSE, nrow = n, ncol = T)

has_A <- matrix(FALSE, nrow = n, ncol = T)
has_B <- matrix(FALSE, nrow = n, ncol = T)

for (t in 1:T) {
  # approximate Poisson counts via rpois around k_diag/k_med
  for (i in 1:n) {
    
    # --- diagnoses ---
    kD <- rpois(1, k_diag)
    if (kD < 1) kD <- 1
    
    base_diag <- sample.int(D, size = min(kD, D), replace = FALSE)
    
    # update chronic pool with a subset
    num_new_chronic <- max(1, floor(length(base_diag) * 0.2))
    new_chronic <- base_diag[seq_len(min(num_new_chronic, length(base_diag)))]
    chronic_pool[[i]] <- unique(c(chronic_pool[[i]], new_chronic))
    
    # carryover
    carry <- chronic_pool[[i]]
    if (length(carry) > 0) {
      keep <- runif(length(carry)) < p_persist
      carry <- carry[keep]
    }
    
    noiseD <- if (noise_diag > 0) sample.int(D, noise_diag, replace = FALSE) else integer(0)
    diag_codes <- unique(c(base_diag, carry, noiseD))
    
    # track ground-truth diag code presence
    if (any(diag_codes %in% risk_diag)) has_risk_diag[i, t] <- TRUE
    if (any(diag_codes %in% prot_diag)) has_prot_diag[i, t] <- TRUE
    if (A_diag %in% diag_codes) {
      has_A[i, t] <- TRUE
      # open policy window for B_med
      policy_until[i] <- max(policy_until[i], min(T, t + window))
    }
    
    # emit diag tokens
    if (length(diag_codes) > 0) {
      events_patient <- c(events_patient, rep.int(i, length(diag_codes)))
      events_time    <- c(events_time, rep.int(t, length(diag_codes)))
      events_token   <- c(events_token, paste0("D_", diag_codes))
    }
    
    # --- medications ---
    kM <- rpois(1, k_med)
    if (kM < 0) kM <- 0
    med_codes <- if (kM > 0) sample.int(M, size = min(kM, M), replace = FALSE) else integer(0)
    
    # policy: encourage B_med if in window
    if (t <= policy_until[i] && runif(1) < p_policy) {
      med_codes <- unique(c(med_codes, B_med))
    }
    
    # add protective run med occasionally (background)
    if (runif(1) < 0.08) {
      med_codes <- unique(c(med_codes, P_med))
    }
    
    noiseM <- if (noise_med > 0) sample.int(M, noise_med, replace = FALSE) else integer(0)
    med_codes <- unique(c(med_codes, noiseM))
    
    # track ground-truth med code presence
    if (any(med_codes %in% risk_med)) has_risk_med[i, t] <- TRUE
    if (any(med_codes %in% prot_med)) has_prot_med[i, t] <- TRUE
    if (B_med %in% med_codes) has_B[i, t] <- TRUE
    if (P_med %in% med_codes) P_present[i, t] <- TRUE
    
    # emit med tokens
    if (length(med_codes) > 0) {
      events_patient <- c(events_patient, rep.int(i, length(med_codes)))
      events_time    <- c(events_time, rep.int(t, length(med_codes)))
      events_token   <- c(events_token, paste0("M_", med_codes))
    }
  }
}

events <- data.frame(
  patient_id = events_patient,
  time = events_time,
  token = events_token,
  stringsAsFactors = FALSE
)

# ---- Build motif indicators per patient (ground truth) ----
motif_hit <- rep(FALSE, n)
for (i in 1:n) {
  tA <- which(has_A[i, ])
  if (length(tA) == 0) next
  # A then B within window (B after A)
  found <- FALSE
  for (t0 in tA) {
    t1 <- min(T, t0 + window)
    if (t0 < t1 && any(has_B[i, (t0 + 1):t1])) {
      motif_hit[i] <- TRUE
      motif_trigger_time[i] <- t0
      found <- TRUE
      break
    }
  }
}

# Protective run: P_med for >= prot_run_len consecutive visits
prot_run_hit <- rep(FALSE, n)
for (i in 1:n) {
  r <- rle(P_present[i, ])
  ends <- cumsum(r$lengths)
  starts <- ends - r$lengths + 1
  idx <- which(r$values == TRUE & r$lengths >= prot_run_len)
  if (length(idx) > 0) {
    prot_run_hit[i] <- TRUE
    prot_run_start_time[i] <- starts[idx[1]]
  }
}

# ---- Risk score (known truth): main effects + motifs + recency ----
# We keep main effects lightweight: just "any risk code present at visit" type signals.
# (You can make it richer by using counts if you want.)
score <- rep(0, n)
for (t in 1:T) {
  score <- score + rec_w[t] * (
    0.8 * has_risk_diag[, t] - 0.7 * has_prot_diag[, t] +
      0.6 * has_risk_med[, t]  - 0.5 * has_prot_med[, t]
  )
}
score <- score + theta_motif * motif_hit + theta_prot_run * prot_run_hit

p <- 1 / (1 + exp(-(intercept + score)))
y <- rbinom(n, 1, p)

# Optional mislabel: flip some positives to 0
if (!is.null(mislabel) && mislabel > 0) {
  pos <- which(y == 1L)
  if (length(pos) > 0) {
    flip <- runif(length(pos)) < mislabel
    y[pos[flip]] <- 0L
  }
}

cat(sprintf("Cancer prevalence: %.2f%%\n", 100 * mean(y)))

labels <- data.frame(patient_id = 1:n, y = y)

# ---- Export ----
write.csv(events, file.path(out_dir, "events.csv"), row.names = FALSE)
write.csv(labels, file.path(out_dir, "labels.csv"), row.names = FALSE)

ground_truth <- list(
  D = D, M = M, T = T, n = n,
  risk_diag = risk_diag,
  prot_diag = prot_diag,
  risk_med  = risk_med,
  prot_med  = prot_med,
  motif = list(type = "A_then_B_within_window",
               A_diag = paste0("D_", A_diag),
               B_med  = paste0("M_", B_med),
               window = window,
               theta = theta_motif),
  protective_motif = list(type = "P_consecutive_run",
                          P_med = paste0("M_", P_med),
                          run_len = prot_run_len,
                          theta = theta_prot_run),
  recency_weights = rec_w,
  intercept = intercept,
  per_patient = list(
    motif_hit = motif_hit,
    motif_trigger_time = motif_trigger_time,
    prot_run_hit = prot_run_hit,
    prot_run_start_time = prot_run_start_time
  )
)

write_json(ground_truth, file.path(out_dir, "ground_truth.json"),
           auto_unbox = TRUE, pretty = TRUE)

cat("Wrote:\n")
cat(" ", file.path(out_dir, "events.csv"), "\n")
cat(" ", file.path(out_dir, "labels.csv"), "\n")
cat(" ", file.path(out_dir, "ground_truth.json"), "\n")
