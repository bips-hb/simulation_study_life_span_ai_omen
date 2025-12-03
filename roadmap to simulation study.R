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

# 5. Increase simulation complexity
#    -> Persistence, resolution, heterogeneity, noise, missingness, confounding

# 6. Re-run models + IML

# 7. Report
#    -> predictive performance, explanation quality, ...



## Load packages ####

library(reticulate)
library(keras3)   
library(dplyr)
library(pROC)
library(tensorflow)
library(ggplot2)
library(reshape2)




## Data Simulation ####

### Toy (Level 0) - Minimal temporal structure ####

# - Binary ICD/ATC that appear randomly per timepoint (no persistence)
# - Outcome depends only on a small subset of features (e.g. ICD1 & ATC1) in a 
#   known linear way.

tf <- import("tensorflow", delay_load = TRUE)

# Set seed
set.seed(3105)

# Set parameters
# PARAMETERS ####
n_patients <- 100
n_timepoints <- 10
n_icd <- 10
n_atc <- 10


# STATIC FEATURES ####
age <- round(runif(n_patients, 18, 90))
sex <- rbinom(n_patients, 1, 0.5)
ses <- sample(c("low", "mid", "high"), 
              n_patients, 
              replace = TRUE, 
              prob = c(0.3, 0.5, 0.2))
ses_dummy <- model.matrix(~ ses - 1)


# CHRONICITY DESIGN ####

# 25% of patients are chronically ill
chronic_flag <- rbinom(n_patients, 1, 0.25) 

# Define ICD and ATC codes that are "typically chronic"
icd_chronic <- c(rep(1, 4), rep(0, n_icd - 4))   # first 4 ICDs chronic
atc_chronic <- c(rep(1, 4), rep(0, n_atc - 4))   # first 4 ATCs chronic


# WIDE DATAFRAME INIT ####
wide <- data.frame(
  patient_id = 1:n_patients,
  age = age,
  sex = sex,
  ses = ses,
  chronic_flag = chronic_flag,
  stringsAsFactors = FALSE
)


# SIMULATE ICDs (wide) ####

# Simulate each ICD code across time, with per-patient persistence/resolution
for (j in 1:n_icd) {
  
  # Baseline prevalence (per patient)
  # -> chronic patients have a slightly higher baseline
  prev <- rbinom(n_patients, 1, 0.08 + 0.05 * chronic_flag)  
  
  for (t in 1:n_timepoints) {
    
    #........................ New diagnosis probability ........................
    
    # Chronic patients are more likely to get any ICD code
    new_prob <- plogis(-4 + 0.03*(age - 50) + 0.25*sex + 0.12*t + 0.9*chronic_flag)
    
    # Make chronic ICD codes more common by themselves
    new_prob <- new_prob * (0.6 + 0.8 * icd_chronic[j])
    
    # Probability of new diagnosis
    new_diag <- rbinom(n_patients, 1, pmin(new_prob, 0.99))
    
    
    #......................... Resolution probability ..........................
    
    # Resolution probability depends on whether the code is chronic and patient chronic flag
    resolve_prob <- case_when(
      icd_chronic[j] == 1 & chronic_flag == 1 ~ 0.01, # chronic ICD code, chronic patient
      icd_chronic[j] == 1 & chronic_flag == 0 ~ 0.05, # chronic ICD code, non-chronic patient
      icd_chronic[j] == 0 & chronic_flag == 1 ~ 0.10, # non-chronic ICD code, chronic patient
      icd_chronic[j] == 0 & chronic_flag == 0 ~ 0.35 # non-chronic ICD code, non-chronic patient
    )
    
    resolved <- as.integer(prev == 1 & runif(n_patients) < resolve_prob)
    
    current <- pmax(prev * (1 - resolved), new_diag)
    
    wide[[paste0("ICD", j, "_t", t)]] <- current
    prev <- current
  }
}


# SIMULATE ATCs (wide) ####

# ATCs depend on ICD burden per timepoint, have courses and different persistence for chronic vs acute ATCs
for (k in 1:n_atc) {
  
  # Baseline probability of being on the medication
  prev <- rbinom(n_patients, 1, 0.03 + 0.03 * chronic_flag)
  
  for (t in 1:n_timepoints) {
    
    #............... ICD occurrence at this time t (across all ICDs) ...........
    
    # Pick all ICD columns at the current timepoint
    icd_cols_t <- grep(paste0("_t", t, "$"), names(wide), value = TRUE)[1:n_icd] 
    
    # Sum ICDs (0/1) for each patient at current timepoint
    icd_occurrence <- rowSums(wide[, icd_cols_t, drop = FALSE])
    
    
    #............................. New ATC probability .........................
    
    # Start probability: base + ICD effect + chronic patient effect)
    # plogis(-4) = 0.018 -> 1.8% baseline prob. pf starting the ATC if patient 
    #                       has no diagnoses and is not chronic
    start_prob <- plogis(-4 + 0.25 * icd_occurrence + 0.4 * chronic_flag)
    
    # Boost for chronic ATCs
    start_prob <- start_prob + 0.3 * atc_chronic[k]
    start_prob <- pmin(start_prob, 0.99)
    
    # Sample new prescriptions
    new_presc <- rbinom(n_patients, 1, start_prob)
    
    
    #................................. Persistence .............................
    
    # Persistence: patients already on med may continue
    
    # Chronic ATCs are less likely to stop
    stop_prob <- ifelse(atc_chronic[k] == 1, 0.02, 0.35)
    still_on <- as.integer(prev == 1 & runif(n_patients) >= stop_prob)
    
    # Current status: new OR continued
    current_atc <- pmax(new_presc, still_on)
    
    # .............................. Save and update ...........................
    
    wide[[paste0("ATC", k, "_t", t)]] <- current_atc
    prev <- current_atc
    
  }
}



# OUTCOME  ####
# Outcome depends on demographics, SES, chronic_flag and recent + total exposure

#................................. Total exposure ..............................

# Columns with ICD codes
icd_cols_all  <- grep("^ICD", names(wide), value = TRUE)

# Columns with ATC codes
atc_cols_all  <- grep("^ATC", names(wide), value = TRUE)

# Count total occurrence of ICD and ATC codes per patient
total_exposure <- rowSums(wide[, c(icd_cols_all, atc_cols_all)])



#............................ Socioeconomic status .............................

ses_coefs <- c(0.3, -0.2, 0.1)
ses_effect <- as.vector(ses_dummy %*% ses_coefs)


#..................................... Outcome .................................

linear_pred <- -5 + 0.02 * age + 0.3 * sex + 0.8 * chronic_flag +
  0.02 * total_exposure + ses_effect
prob <- plogis(linear_pred)
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

# aggregate: total causal ICD & ATC exposures 
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
  
  # static
  age_vec <- x_array[,1,"age"]
  sex_vec <- x_array[,1,"sex"]
  cbind(age = age_vec, sex = sex_vec, total_icd = total_icd, total_atc = total_atc)
}

X_train_agg <- agg_features(x_train)
X_val_agg   <- agg_features(x_val)
glm_base <- glm(y_train ~ ., data = as.data.frame(X_train_agg), family = binomial)
summary(glm_base)
pred_glm <- predict(glm_base, as.data.frame(X_val_agg), type = "response")
roc_glm <- roc(y_val, pred_glm); auc_glm <- as.numeric(roc_glm$auc)
cat("Baseline logistic AUC:", round(auc_glm,3), "\n")




n_timesteps <- dim(x_train)[2]
n_features <- dim(x_train)[3]



## --- LSTM model ---
model_lstm <- keras_model_sequential() %>%
  layer_lstm(units = 16, input_shape = c(n_timepoints, n_features), activation = "tanh") %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model_lstm %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

history_lstm <- model_lstm %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 16,
  validation_split = 0.2,
  verbose = 1
)

## --- GRU model ---
model_gru <- keras_model_sequential() %>%
  layer_gru(units = 16, input_shape = c(n_timepoints, n_features), activation = "tanh") %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model_gru %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

history_gru <- model_gru %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 16,
  validation_split = 0.2,
  verbose = 1
)

# Evaluate both
cat("\nLSTM Performance:\n")
model_lstm %>% evaluate(x_val, y_val)

cat("\nGRU Performance:\n")
model_gru %>% evaluate(x_val, y_val)



# Conmpare Validation Accuracy of LSTM RNN and GRU RNN
par(mfrow = c(1, 2))

# LSTM validation accuracy
plot(history_lstm$metrics$val_accuracy, type = "l", col = "blue", lwd = 2,
     xlab = "Epoch", ylab = "Validation Accuracy", ylim = c(0, 1),
     main = "LSTM Validation Accuracy")

# GRU validation accuracy
plot(history_gru$metrics$val_accuracy, type = "l", col = "red", lwd = 2,
     xlab = "Epoch", ylab = "Validation Accuracy", ylim = c(0, 1),
     main = "GRU Validation Accuracy")

# Reset plotting layout back to default
par(mfrow = c(1, 1))






# IML techniques ####

## Agnostic ####

### Saliency Maps ####

#.................... Step 1: Compute saliency for all patients ................

N <- dim(x_val)[1]
timesteps <- dim(x_val)[2]
features <- dim(x_val)[3]

# Storage array: average saliency per patient
saliency_array <- array(0, dim = c(N, timesteps, features))

for (i in 1:N) {
  
  x_tf <- tf$constant(x_val[i,,, drop = FALSE])
  
  with(tf$GradientTape() %as% tape, {
    tape$watch(x_tf)
    pred <- model_lstm(x_tf)
  })
  
  grads <- tape$gradient(pred, x_tf)
  
  saliency_array[i,,] <- abs(as.array(grads)[1,,])
  
  if (i %% 100 == 0) cat("Processed patient:", i, "\n")
}

cat("Done computing saliency for all patients.\n")


#.................. Step 2: Aggregate saliency across all patients .............

# Mean absolute saliency over patients
saliency_avg <- apply(saliency_array, c(2, 3), mean)


#.................... Step 3: Convert to long format for ggplot ................

feature_names <- c(
  "age",
  "sex",
  "seshigh",
  "seslow",
  "sesmid",
  paste0("ICD", 1:10),
  paste0("ATC", 1:10)
)

df <- melt(saliency_avg)
names(df) <- c("Time", "FeatureIndex", "Value")

df$Feature <- factor(df$FeatureIndex,
                     levels = 1:features,
                     labels = feature_names)


#.................... Step 4: Plot aggregated saliency heatmap .................

ggplot(df, aes(x = Time, y = Feature, fill = Value)) +
  geom_tile() +
  scale_fill_gradient(low = "yellow", high = "red") +
  scale_x_continuous(
    breaks = seq(1, timesteps, 1),
    labels = seq(1, timesteps, 1)
  ) +
  scale_y_discrete(limits = feature_names) +
  labs(
    title = "Aggregated Saliency Map (Mean Absolute Gradient)",
    x = "Timepoint",
    y = "Feature"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    panel.grid = element_blank(),
    axis.ticks = element_line()
  )


#............................... What can be seen ..............................

# - Later timepoints matter more, so the model relies more on recent history 
#   than early timepoints. Expected for clinical trajectories as recent medication
#   or diagnoses affect risk more strongly.

# - Some ATC features dominate.

# - Most ICD features have a low or moderate importance, so they contribute
#   but less strongly than certain ATC features.

# - Static features have a small influence, so the model relies more on longi-
#   tudinal coding patterns than static demographics


#........................... What the map does not tell us .....................

# - Whether a feature increases or decreases the risk
# - Whether a feature is causal
# - Whether the features's effect is positive or negative


### Integrated Gradients ####

integrated_gradients <- function(model, input, baseline = NULL, steps = 50) {
  
  if (is.null(baseline)) {
    baseline <- array(0, dim = dim(input))
  }
  
  input_tf <- tf$constant(input)
  baseline_tf <- tf$constant(baseline)
  
  # Create scaled interpolated inputs
  alphas <- tf$linspace(0.0, 1.0, steps)
  
  total_gradients <- 0
  
  for (a in as.array(alphas)) {
    interpolated <- baseline_tf + a * (input_tf - baseline_tf)
    
    with(tf$GradientTape() %as% tape, {
      tape$watch(interpolated)
      pred <- model(interpolated)
    })
    
    grads <- tape$gradient(pred, interpolated)
    total_gradients <- total_gradients + grads
  }
  
  avg_grads <- total_gradients / steps
  integrated <- (input_tf - baseline_tf) * avg_grads
  
  return(as.array(integrated)[1,,])   # time × features
}



## Specific ####





















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

