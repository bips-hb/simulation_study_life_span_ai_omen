#...............................................................................
# Big-picture ..................................................................
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


#...............................................................................
# Load packages ####
#...............................................................................

library(reticulate)
library(keras3)   
library(dplyr)
library(pROC)
library(tensorflow)
library(ggplot2)
library(reshape2)


# Set seed
set.seed(3105)

#...............................................................................
# PARAMETERS ####
#...............................................................................
n_patients <- 1000
n_timepoints <- 10
n_icd <- 10
n_atc <- 15


#...............................................................................
# STATIC FEATURES ####
#...............................................................................
age <- round(runif(n_patients, 18, 90))
sex <- rbinom(n_patients, 1, 0.5)
ses <- sample(c("low", "mid", "high"), 
              n_patients, 
              replace = TRUE, 
              prob = c(0.3, 0.5, 0.2))
ses_dummy <- model.matrix(~ ses - 1)


#...............................................................................
# CHRONICITY DESIGN ####
#...............................................................................

# 25% of patients are chronically ill
chronic_flag <- rbinom(n_patients, 1, 0.25) 


# Certain ICD / ATC codes are "typically chronic" and
# tend to persist once they appear.
n_icd_chronic <- ceiling(0.1 * n_icd)
n_atc_chronic <- ceiling(0.15 * n_atc)

icd_chronic <- c(
  rep(1, n_icd_chronic),           # chronic ICD codes
  rep(0, n_icd - n_icd_chronic)
)

atc_chronic <- c(
  rep(1, n_atc_chronic),           # chronic ATC codes
  rep(0, n_atc - n_atc_chronic)
)


#...............................................................................
# WIDE DATAFRAME INIT ####
#...............................................................................
wide <- data.frame(
  patient_id = 1:n_patients,
  age = age,
  sex = sex,
  ses = ses,
  chronic_flag = chronic_flag,
  stringsAsFactors = FALSE
)


#...............................................................................
# SIMULATE ICDs (wide) ####
#...............................................................................

# Simulate each ICD code across time, with per-patient persistence/resolution
for (j in 1:n_icd) {
  
  # Baseline prevalence (per patient)
  # -> chronic patients have a slightly higher baseline
  prev <- rbinom(n_patients, 1, 0.02 + 0.03 * chronic_flag)  
  
  for (t in 1:n_timepoints) {
    
    #........................ New diagnosis probability ........................
    
    # Chronic patients are more likely to get any ICD code
    new_prob <- plogis(-5 + 0.02*(age - 50) + 0.2*sex + 0.1*t + 0.8*chronic_flag)
    
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


#...............................................................................
# SIMULATE ATCs (wide) ####
#...............................................................................

for (k in 1:n_atc) {
  
  # Baseline probability of being on the medication
  prev <- rbinom(n_patients, 1, 0.02 + 0.02 * chronic_flag)
  related_icds <- sample(1:n_icd,2) # each ATC depends on 2 ICDs
  
  for(t in 1:n_timepoints){
    icd_cols_t <- paste0("ICD",related_icds,"_t",t)
    icd_effect <- rowSums(wide[, icd_cols_t, drop=FALSE])
    
    start_prob <- plogis(-5 + 0.4*icd_effect + 0.3*chronic_flag + 0.3*atc_chronic[k])
    start_prob <- pmin(start_prob,0.99)
    new_presc <- rbinom(n_patients,1,start_prob)
    
    stop_prob <- ifelse(atc_chronic[k]==1,0.02,0.35)
    still_on <- as.integer(prev==1 & runif(n_patients)>=stop_prob)
    current <- pmax(new_presc, still_on)
    
    wide[[paste0("ATC",k,"_t",t)]] <- current
    prev <- current
  }
}


#...............................................................................
# OUTCOME  ####
#...............................................................................
# Outcome depends on time-weighted ICD & ATC exposures

icd_cols_all <- grep("^ICD", names(wide), value=TRUE)
atc_cols_all <- grep("^ATC", names(wide), value=TRUE)

time_weights <- seq(0.5,1.5,length.out = n_timepoints)

time_weighted_icd <- rowSums(as.matrix(wide[, icd_cols_all]) * rep(time_weights, each=n_patients))
time_weighted_atc <- rowSums(as.matrix(wide[, atc_cols_all]) * rep(time_weights, each=n_patients))

ses_effect <- as.vector(ses_dummy %*% c(0.3,-0.2,0.1))

linear_pred <- -2 + 0.02*age + 0.3*sex + 0.8*chronic_flag +
  2*time_weighted_icd + 2.5*time_weighted_atc + ses_effect
prob <- plogis(linear_pred)
wide$outcome <- rbinom(n_patients,1,prob)






#...............................................................................
# Convert from wide format into 3D array (patients, timesteps, features) ####
#...............................................................................

make_rnn_array_from_wide <- function(df, n_timepoints, n_icd, n_atc) {
  
  # static matrix for SES
  static_mat   <- model.matrix(~ df$ses - 1)
  
  # static features: age, sex, chronic_flag + SES dummies
  static_names <- c("age", "sex", "chronic_flag", colnames(static_mat))
  n_static     <- length(static_names)
  
  n_features   <- n_static + n_icd + n_atc
  n_patients   <- nrow(df)
  
  # initialize array
  arr <- array(0, dim = c(n_patients, n_timepoints, n_features))
  
  for (i in 1:n_patients) {
    for (t in 1:n_timepoints) {
      
      icd_vec <- as.numeric(df[i, paste0("ICD", 1:n_icd, "_t", t)])
      atc_vec <- as.numeric(df[i, paste0("ATC", 1:n_atc, "_t", t)])
      static_vec <- c(df$age[i], df$sex[i], df$chronic_flag[i], as.numeric(static_mat[i,]))
      
      arr[i, t, ] <- c(static_vec, icd_vec, atc_vec)
    }
  }
  
  dimnames(arr)[[3]] <- c(static_names, paste0("ICD",1:n_icd), paste0("ATC",1:n_atc))
  
  return(arr)
}


x_array <- make_rnn_array_from_wide(wide, n_timepoints, n_icd, n_atc)


#...............................................................................
## Train/Validation split ####
#...............................................................................

set.seed(42)
idx <- sample(seq_len(n_patients), 0.8*n_patients)

x_train <- x_array[idx,,,drop=FALSE]
x_val   <- x_array[-idx,,,drop=FALSE]
y_train <- wide$outcome[idx]
y_val   <- wide$outcome[-idx]


n_features <- dim(x_train)[3]





#...............................................................................
# Sanity checks ####
#...............................................................................


## Sparsity ####
total_elements <- prod(dim(x_array))
non_zero_elements <- sum(x_array != 0)
sparsity <- 1 - non_zero_elements / total_elements
cat("Overall sparsity:", round(sparsity*100,2), "%\n")


## Check outcome prevalence ####
mean(wide$outcome)
summary(prob)


## Basic arry dimensions ####

dim(x_array)  # should be: n_patients x n_timepoints x n_features
dimnames(x_array)[[3]]  # feature names


## Check a few single patients ####

patient1 <- x_array[1,,]  # all timesteps for patient 1
patient1_df <- as.data.frame(patient1)
patient1_df$time <- 1:n_timepoints
patient1_df

patient5 <- x_array[5,,]  # all timesteps for patient 5
patient5_df <- as.data.frame(patient5)
patient5_df$time <- 1:n_timepoints
patient5_df

patient13 <- x_array[13,,]  # all timesteps for patient 13
patient13_df <- as.data.frame(patient13)
patient13_df$time <- 1:n_timepoints
patient13_df

# Q: Is it plausible to have no ICD codes but ATC codes???


## Check feature ranges ####
apply(x_array, 3, function(x) range(x))


## Check aggregated sums of ICD and ATC codes ####

# Check that the simulated persistent/chronic patterns behave as intended
rowSums(x_array[1, , grep("^ICD", dimnames(x_array)[[3]])])
rowSums(x_array[1, , grep("^ATC", dimnames(x_array)[[3]])])


## Compare array with wide data frame ####
all(x_array[1,1,"ICD1"] == wide$ICD1_t1[1])  # should be TRUE
all(x_array[1,2,"ATC3"] == wide$ATC3_t2[1])  # should be TRUE



## Visual sanity check ####
# Plot a small heatmap for a few patients
sample_pat <- x_array[1:5,, grep("^ICD", dimnames(x_array)[[3]])]
df_plot <- melt(sample_pat)
colnames(df_plot) <- c("Patient", "Time", "ICD", "Value")

ggplot(df_plot, aes(x=Time, y=ICD, fill=Value)) +
  geom_tile() +
  facet_wrap(~Patient) +
  scale_fill_gradient(low="white", high="blue") +
  theme_minimal()




#...............................................................................
# Models ####
#...............................................................................


#...............................................................................
## Baseline logistic on aggregated features ####
#...............................................................................

# Aggregate ICD & ATC codes
wide$sum_icd <- rowSums(wide[, grep("^ICD", names(wide))])
wide$sum_atc <- rowSums(wide[, grep("^ATC", names(wide))])

# Filter variables
wide_glm <- wide %>% 
  select(age, sex, ses, outcome, sum_icd, sum_atc)


# Split into train/validation
set.seed(42)
train_indices <- sample(1:nrow(wide_glm), size = round(0.8 * nrow(wide_glm)))
wide_glm_train <- wide_glm[train_indices, ]
wide_glm_val <- wide_glm[-train_indices, ]


# Fit baseline logistic regression
glm_base <- glm(outcome ~ ., data = wide_glm_train, family = binomial)
summary(glm_base)


# Predict on training set
pred_train <- predict(glm_base, newdata = wide_glm_train, type = "response")

# Predict on validation set
pred_val <- predict(glm_base, newdata = wide_glm_val, type = "response")

# Compute AUC for training set
roc_train <- roc(wide_glm_train$outcome, pred_train)
auc_train <- as.numeric(roc_train$auc)
cat("Training AUC:", round(auc_train, 3), "\n")

# Compute AUC for validation set
roc_val <- roc(wide_glm_val$outcome, pred_val)
auc_val <- as.numeric(roc_val$auc)
cat("Validation AUC:", round(auc_val, 3), "\n")



# Visualize ROC curves
plot(roc_val, col="blue", main="ROC Curve - Validation")
plot(roc_train, col="red", add=TRUE)
legend("bottomright", legend=c("Train", "Validation"), col=c("red","blue"), lwd=2)





#...............................................................................
## LSTM model ####
#...............................................................................

## Normalize age
mu <- mean(x_train[,,"age"])
sdv <- sd(x_train[,,"age"])
x_train[,,"age"] <- (x_train[,,"age"] - mu)/sdv
x_val[,,"age"]   <- (x_val[,,"age"] - mu)/sdv



# Build model architecture
model_lstm <- keras_model_sequential() %>%
  layer_lstm(
    units = 30,
    input_shape = c(n_timepoints, n_features),
    dropout = 0.2,
    recurrent_dropout = 0.2
  ) %>%
  layer_dense(1, activation = "sigmoid")

# Compile model
model_lstm %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list(metric_auc(name="auc"), "accuracy")
)

# Train model
history_lstm <- model_lstm %>% fit(
  x_train, y_train,
  validation_data = list(x_val, y_val),
  epochs = 30,
  batch_size = 64,
  callbacks = list(
    callback_early_stopping(
      monitor = "val_auc",
      mode = "max",
      patience = 5,
      restore_best_weights = TRUE
    )
  ),
  verbose = 1
)

history_lstm$metrics$auc
history_lstm$metrics$val_auc

history_lstm$metrics$accuracy
history_lstm$metrics$val_accuracy



#...............................................................................
## GRU model 
#...............................................................................

# Build model architecture
model_gru <- keras_model_sequential() %>%
  layer_gru(
    units = 30,
    input_shape = c(n_timepoints, n_features),
    dropout = 0.2,
    recurrent_dropout = 0.2
  ) %>%
  layer_dense(1, activation = "sigmoid")

# Compile model
model_gru %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list(metric_auc(name="auc"), "accuracy")
)

history_gru <- model_gru %>% fit(
  x_train, y_train,
  validation_data = list(x_val, y_val),
  epochs = 30,
  batch_size = 64,
  callbacks = list(
    callback_early_stopping(
      monitor = "val_auc",
      mode = "max",
      patience = 5,
      restore_best_weights = TRUE
    )
  ),
  verbose = 1
)

history_gru$metrics$auc
history_gru$metrics$val_auc

history_gru$metrics$accuracy
history_gru$metrics$val_accuracy









# Compare Validation Accuracy of LSTM RNN and GRU RNN
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

tfexplain <- reticulate::import("tf_explain")



## Integrated gradients ####

explainer <- tfexplain$core$integrated_gradients$IntegratedGradients()

# Explain a batch of patients
ig <- explainer$explain(
  validation_data = list(x_val, y_val),
  model = model_lstm,
  n_steps = 50
)







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

