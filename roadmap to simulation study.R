#...............................................................................
# Big-picture ............................................................. ####
#...............................................................................

# 1. Simulate simple, controlled data
#    -> Ground-truth-feature-outcome-relationships are known (easy effects)

# 2. Fit simple baselines
#    -> Logistic regression and/or XGBoost on aggregated features

# 3. Train sequential DL models (LSTM, GRU, TCN, transformer-based methods) on 
#    the 3D-array
#    -> Get stable training, reasonable metrics

# 4. Apply IML methods
#    -> Do they recover the known ground-truth signals?

# 5. Inrease simulation complexity
#    -> Persistence, resolution, heterogeneity, noise, missingness, confounding

# 6. Re-run models + IML

# 7. Report
#    -> predictive performance, explanation quality, ...



## Load packages ####

library(reticulate)
library(keras3)   
library(dplyr)
library(pROC)




## Data Simulation ####

### Toy (Level 0) - Minimal temporal structure ####

# - Binary ICD/ATC that appear randomly per timepoint (no persistence)
# - Outcome depends only on a small subset of features (e.g. ICD1 & ATC1) in a 
#   known linear way.

tf <- import("tensorflow", delay_load = TRUE)

# Set seed
set.seed(3105)

# Set parameters
n_patients <- 2000
n_timepoints <- 10
n_icd <- 5
n_atc <- 5
causal_icd <- 1
causal_atc <- 1
effect_icd <- 1.0
effect_atc <- -1.0

# Simple static features
age <- round(runif(n_patients, 18, 90))
sex <- rbinom(n_patients, 1, 0.5)
ses <- sample(c("low","mid","high"), 
              n_patients, 
              replace = TRUE, 
              prob = c(0.3, 0.5, 0.2))
ses_dummy <- model.matrix(~ ses - 1)

wide <- data.frame(patient_id = 1:n_patients,
                   age = age, sex = sex, ses = ses,
                   stringsAsFactors = FALSE)

# Simple random ICDs and ATCs across time, but causal_icd/codal_atc will be injected
for (j in 1:n_icd) {
  
  prev <- rbinom(n_patients, 1, 0.05) # baseline prevalence
  
  for (t in 1:n_timepoints) {
    
    p <- 0.02 + 0.002 * (age - 50) + 0.02 * sex
    p <- pmin(pmax(p, 0), 1)
    new <- rbinom(n_patients, 1, p)

    current <- pmax(prev, new)
    
    # inject a stronger signal for the causal ICD across all times for some fraction
    if (j == causal_icd) {
      # make causal ICD more frequent
      flip <- rbinom(n_patients, 1, 0.15)
      current <- pmax(current, flip)
    }
    
    wide[[paste0("ICD", j, "_t", t)]] <- current
    prev <- current
    
  }
  
}

for (k in 1:n_atc) {
  
  prev <- rbinom(n_patients, 1, 0.02)
  
  for (t in 1:n_timepoints) {
    
    # ATC triggered by ICD burden at same t (simple)
    icd_cols_t <- grep(paste0("^ICD[0-9]+_t", t, "$"), names(wide), value = TRUE)
    
    # Make sure we have exactly n_icd matches
    stopifnot(length(icd_cols_t) == n_icd)
    
    
    icd_burden <- rowSums(wide[, icd_cols_t, drop=FALSE])
    
    prob <- plogis(-4 + 0.3*icd_burden)
    new <- rbinom(n_patients, 1, prob)
    current <- pmax(prev, new)
    
    # if this is causal_atc, give it protective pattern: more frequent for some subset
    if (k == causal_atc) {
      flip <- rbinom(n_patients, 1, 0.08)
      current <- pmax(current, flip)
      
    }
    
    wide[[paste0("ATC", k, "_t", t)]] <- current
    prev <- current
    
  }
  
}



# Outcome: logistic of age/sex + effect from causal ICD + ATC
icd_cols_all <- grep("^ICD", names(wide), value = TRUE)
atc_cols_all <- grep("^ATC", names(wide), value = TRUE)
causal_icd_cols <- grep(paste0("^ICD", causal_icd, "_t"), names(wide), value = TRUE)
causal_atc_cols <- grep(paste0("^ATC", causal_atc, "_t"), names(wide), value = TRUE)
sum_icd_causal <- rowSums(wide[, causal_icd_cols, drop=FALSE])
sum_atc_causal <- rowSums(wide[, causal_atc_cols, drop=FALSE])

linear_pred <- -3 + 0.02*age + 0.3*sex + effect_icd * sum_icd_causal + effect_atc * sum_atc_causal
prob <- 1/(1+exp(-linear_pred))
y <- rbinom(n_patients, 1, prob)
wide$outcome <- y
wide$prob <- prob




## Convert into 3D array (patients, timesteps, features) ####
make_rnn_array_from_wide <- function(df, n_timepoints, n_icd, n_atc) {
  n_patients <- nrow(df)
  # static features: age, sex, ses dummies
  static_mat <- model.matrix(~ df$ses - 1)
  static_names <- c("age","sex", colnames(static_mat))
  n_static <- length(static_names)
  n_features <- n_static + n_icd + n_atc
  arr <- array(0, dim = c(n_patients, n_timepoints, n_features))
  for (i in 1:n_patients) {
    for (t in 1:n_timepoints) {
      icd_names_t <- paste0("ICD", 1:n_icd, "_t", t)
      atc_names_t <- paste0("ATC", 1:n_atc, "_t", t)
      rowvals <- c(df$age[i], df$sex[i], as.numeric(static_mat[i,]),
                   as.numeric(df[i, icd_names_t]),
                   as.numeric(df[i, atc_names_t]))
      arr[i, t, ] <- rowvals
    }
  }
  feature_names <- c(static_names, paste0("ICD",1:n_icd), paste0("ATC",1:n_atc))
  dimnames(arr) <- list(paste0("pat",1:n_patients), paste0("t",1:n_timepoints), feature_names)
  return(arr)
}

 
wide0 <- wide


x_array0 <- make_rnn_array_from_wide(wide0, n_timepoints, n_icd, n_atc)
dim(x_array0)  # check




## Train/Test split by patient ####
set.seed(42)
n <- dim(x_array0)[1]
train_idx <- sample(seq_len(n), size = round(0.8*n))
x_train <- x_array0[train_idx,,,drop=FALSE]
y_train <- wide0$outcome[train_idx]
x_val <- x_array0[-train_idx,,,drop=FALSE]
y_val <- wide0$outcome[-train_idx]

# Normalize static continuous (age) globally using train mean/sd
age_mean <- mean(x_train[,, "age"])
age_sd   <- sd(x_train[,, "age"])
x_train[,, "age"] <- (x_train[,, "age"] - age_mean)/age_sd
x_val[,, "age"]   <- (x_val[,, "age"] - age_mean)/age_sd





## Baseline logistic on aggregated features ####
# aggregate: total causal ICD & ATC exposures and recent exposures (last 3 timepoints)
agg_features <- function(x_array) {
  
  n_pat <- dim(x_array)[1]
  n_time <- dim(x_array)[2]
  
  # columns names from dimnames:
  fnames <- dimnames(x_array)[[3]]
  icd_idx <- which(grepl("^ICD", fnames))
  atc_idx <- which(grepl("^ATC", fnames))
  
  # total exposure
  total_icd <- apply(x_array[,, icd_idx, drop=FALSE], 1, sum)
  total_atc <- apply(x_array[,, atc_idx, drop=FALSE], 1, sum)
  
  # recent (last 3 timepoints)
  last3_icd <- apply(x_array[, (n_time-2):n_time, icd_idx, drop=FALSE], 1, sum)
  last3_atc <- apply(x_array[, (n_time-2):n_time, atc_idx, drop=FALSE], 1, sum)
  
  # static
  age_vec <- x_array[,1,"age"]
  sex_vec <- x_array[,1,"sex"]
  cbind(age = age_vec, sex = sex_vec, total_icd = total_icd, total_atc = total_atc,
        last3_icd = last3_icd, last3_atc = last3_atc)
}

X_train_agg <- agg_features(x_train)
X_val_agg   <- agg_features(x_val)
glm_base <- glm(y_train ~ ., data = as.data.frame(X_train_agg), family = binomial)
summary(glm_base)
pred_glm <- predict(glm_base, as.data.frame(X_val_agg), type = "response")
roc_glm <- roc(y_val, pred_glm); auc_glm <- as.numeric(roc_glm$auc)
cat("Baseline logistic AUC:", round(auc_glm,3), "\n")







## Small LSTM model ####
# Build model: input shape = (timesteps, features)
timesteps <- dim(x_train)[2]
features <- dim(x_train)[3]

build_lstm_model <- function(timesteps, features, units = 32, lr = 1e-3) {
  model <- keras_model_sequential() %>%
    layer_lstm(units, input_shape = c(timesteps, features), return_sequences = FALSE) %>%
    layer_dropout(0.2) %>%
    layer_dense(1, activation = "sigmoid")
  model %>% compile(optimizer = optimizer_adam(learning_rate = lr),
                    loss = "binary_crossentropy",
                    metrics = list(metric_binary_accuracy))
  return(model)
}

model <- build_lstm_model(timesteps, features, units = 32, lr = 1e-3)
print(model)
# Fit (small epochs for demo)
history <- model %>% fit(x = x_train, y = y_train,
                         validation_data = list(x_val, y_val),
                         epochs = 12, batch_size = 64)

# Evaluate
pred_lstm_val <- as.numeric(model %>% predict(x_val))
auc_lstm <- as.numeric(roc(y_val, pred_lstm_val)$auc)
cat("LSTM val AUC:", round(auc_lstm,3), "\n")





## Integrated Gradients (simple approximation) ####
# Implementation notes:
# - IG: approximate integral of gradients from baseline to input
# - We'll compute IG for one patient at a time (feature x time grid)
# - Baseline: zero array (or mean baseline)
# - Uses tf$GradientTape via reticulate; works with keras3 model (TF backend)

integrated_gradients <- function(model, instance, baseline = NULL, m_steps = 50, target_index = NULL) {
  # instance: array shape (1, timesteps, features)
  if (is.null(baseline)) baseline <- array(0, dim = dim(instance))
  alphas <- seq(0, 1, length.out = m_steps)
  total_grad <- array(0, dim = dim(instance))
  for (a in alphas) {
    input_a <- baseline + a * (instance - baseline)
    # convert to tf tensor
    x_tf <- tf$convert_to_tensor(input_a, dtype = "float32")
    with (tf$GradientTape() %as% tape, {
      tape$watch(x_tf)
      preds <- model(x_tf)
      # if binary, preds is shape (1,1)
      # if target_index specified (for multi-class) pick that logit
      loss_tf <- preds[,1]
    })
    grads <- tape$gradient(loss_tf, x_tf)  # shape (1,timesteps,features)
    grads_np <- as.array(grads)
    total_grad <- total_grad + grads_np
  }
  avg_grad <- total_grad / m_steps
  ig <- (instance - baseline) * avg_grad
  return(as.numeric(ig)) # flattened; reshape by dim(instance)
}

# Test IG on a small subset: 10 validation patients
n_test_ig <- 20
ig_results <- array(0, dim = c(n_test_ig, timesteps, features))
for (i in 1:n_test_ig) {
  idx <- i
  instance <- x_val[idx,,,drop=FALSE]
  ig_vec <- integrated_gradients(model, instance, baseline = array(0, dim = dim(instance)), m_steps = 30)
  ig_results[i,,] <- array(ig_vec, dim = dim(instance))
}
# Example: aggregate IG across time for feature importance
ig_feat_importance <- apply(abs(ig_results), c(3), mean)
feature_names <- dimnames(x_train)[[3]]
names(ig_feat_importance) <- feature_names
sort(ig_feat_importance, decreasing = TRUE)[1:10]




## Temporal occlusion ####
# For each feature (column index), mask it across all timesteps (set to 0) and measure
# change in average prediction (on validation set)
occlusion_scores <- function(model, x_val, baseline = 0, feature_idx = NULL) {
  n_pat <- dim(x_val)[1]
  base_preds <- as.numeric(model %>% predict(x_val))
  if (is.null(feature_idx)) feature_idx <- 1:dim(x_val)[3]
  deltas <- numeric(length(feature_idx))
  for (j in seq_along(feature_idx)) {
    fi <- feature_idx[j]
    x_pert <- x_val
    x_pert[,,fi] <- baseline
    pred_pert <- as.numeric(model %>% predict(x_pert))
    deltas[j] <- mean(base_preds - pred_pert)  # positive => removing feature reduces prediction
  }
  names(deltas) <- dimnames(x_val)[[3]][feature_idx]
  return(deltas)
}

occl <- occlusion_scores(model, x_val, baseline = 0)
sort(occl, decreasing = TRUE)[1:12]




## Quick evaluation: do top attributed features contain the true causal ones? ####
# Ground truth: causal ICD1 and causal ATC1 (their feature names)

true1 <- c("ICD1","ATC1")

# Map to feature indices
feat_names <- dimnames(x_val)[[3]]
idx_true <- which(feat_names %in% true1)

# Using occlusion ranking:
top_occl <- names(sort(occl, decreasing = TRUE))[1:10]
cat("Top occlusion features:", paste(top_occl, collapse=", "), "\n")
cat("True features present in top10 (occlusion):", sum(true1 %in% top_occl), "/2\n")

# Using IG aggregated ranking:
ig_rank <- names(sort(ig_feat_importance, decreasing = TRUE))[1:10]
cat("Top IG features:", paste(ig_rank, collapse=", "), "\n")
cat("True features present in top10 (IG):", sum(true1 %in% ig_rank), "/2\n")





## Summary prints ####
cat("Baseline logistic AUC:", round(auc_glm,3), "\n")
cat("LSTM val AUC:", round(auc_lstm,3), "\n")
cat("Top occlusion features (first 10):", paste(names(sort(occl, decreasing = TRUE))[1:10], collapse=", "), "\n")
cat("Top IG features (first 10):", paste(names(sort(ig_feat_importance, decreasing = TRUE))[1:10], collapse=", "), "\n")

# Save results if you want
res <- list(model = model, glm = glm_base, occlusion = occl, IG_feat = ig_feat_importance,
            X_train_agg = X_train_agg, X_val_agg = X_val_agg)
saveRDS(res, "sim_experiment_results.rds")

