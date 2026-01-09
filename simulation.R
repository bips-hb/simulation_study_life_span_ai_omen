
# Realistic (hopefully lol) Longitudinal Health Insurance Data Simulation ####

# Structure of sequence data:
# -> RNNs are designed to handle sequences, so data where order matters and past
#    information influences the future
# -> Structure of sequence data: 3D tensor
#    (number of patients/batch size, number of time steps, number of features)


## Load packages ####
library(keras3)

set.seed(123)

## PARAMETERS ####
n_patients <- 1000
n_timepoints <- 20
n_icd <- 20
n_atc <- 20
n_features <- n_icd + n_atc + 5  # age + sex + 3 SES dummies


## STATIC FEATURES ####
age <- round(runif(n_patients, 18, 90))
sex <- rbinom(n_patients, 1, 0.5) # 0=female, 1=male
ses <- sample(c("low", "mid", "high"), n_patients, replace = TRUE, prob = c(0.3, 0.5, 0.2))
ses_dummy <- model.matrix(~ ses - 1)  # dummy-encoded columns: seshigh, seslow, sesmid


## CHRONIC PATIENTS ####
prop_chronic <- 0.30   # fraction of patients with chronic disease behavior
is_chronic <- rbinom(n_patients, 1, prop_chronic)  # 1 = chronic patient
table(is_chronic)


## DICD/ATC TYPES ####
# First few ICDs and ATCs are chronic-prone
chronic_icd_idx <- 1:6    # ICD codes that are likely chronic
acute_icd_idx   <- 7:n_icd
chronic_atc_idx <- 1:6
acute_atc_idx   <- 7:n_atc


## Stop probabilities (per-step) depending on chronic vs acute and patient chronic flag
# ICD stop prob: chronic ICDs low stop prob if patient is chronic, otherwise higher
icd_stop_prob_chronic_patient <- 0.02  # chronic patients rarely resolve chronic ICD
icd_stop_prob_nonchronic_patient <- 0.2
icd_stop_prob_acute <- 0.4  # acute ICDs likely to resolve

# ATC stop probs
atc_stop_prob_chronic <- 0.02
atc_stop_prob_acute <- 0.4


## ARRAYS (3D) ####
x_array <- array(0, dim = c(n_patients, n_timepoints, n_features))


## LONGITUDINAL ICD + ATC CODES ####

for(i in 1:n_patients){
  
  # baseline prev for ICD and ATC depends on chronic status
  if(is_chronic[i]==1){
    
    # chronic patients: higher baseline for chronic-prone ICDs
    icd_prev <- rep(0, n_icd)
    icd_prev[chronic_icd_idx] <- rbinom(length(chronic_icd_idx), 1,
                                        plogis(-3.5 + 0.03*(age[i]-50) + 0.15*sex[i]))
    icd_prev[acute_icd_idx] <- rbinom(length(acute_icd_idx), 1,
                                      plogis(-4 + 0.03*(age[i]-50) + 0.15*sex[i]))
    
    atc_prev <- rbinom(n_atc, 1, plogis(-4 + 0.4 + 0.02*(age[i]-50) + 0.15*sex[i]))
    
  } else {
    
    icd_prev <- rbinom(n_icd, 1, plogis(-4 + 0.03*(age[i]-50) + 0.15*sex[i]))
    atc_prev <- rbinom(n_atc, 1, plogis(-4 + 0.02*(age[i]-50) + 0.1*sex[i]))
    
  }
  
  for(t in 1:n_timepoints){
    
    # new ICDs (risk increases with time)
    new_icd <- rbinom(n_icd, 1,
                      plogis(-4 + 0.03*(age[i]-50) + 0.15*sex[i] + 0.08*t))
    
    # compute stop probabilities per ICD based on type and patient chronicness
    icd_stop_probs <- rep(NA, n_icd)
    icd_stop_probs[chronic_icd_idx] <- ifelse(is_chronic[i]==1,
                                              icd_stop_prob_chronic_patient,
                                              icd_stop_prob_nonchronic_patient)
    icd_stop_probs[acute_icd_idx] <- icd_stop_prob_acute
    icd_stop_probs <- pmin(pmax(icd_stop_probs * icd_stop_factor, 0.01), 0.99)
    
    # some existing ICDs can resolve with their stop probability
    icd_resolve <- ifelse(icd_prev == 1 & runif(n_icd) < icd_stop_probs, 1, 0)
    # combine persistence, new, and resolution
    icd <- ifelse(icd_resolve==1, 0, pmax(icd_prev, new_icd))
    icd_prev <- icd
    
    # ATC start prob increases if corresponding ICD present (map ICD->ATC crudely 1-to-1)
    # and chronic patients have lower stop probabilities for chronic ATCs
    atc_start_prob <- plogis(atc_base_logit + 0.25 * icd[1:n_atc])  # boost when ICD present
    # ensure some baseline variability
    new_atc <- rbinom(n_atc, 1, atc_start_prob)
    
    # ATC stop probs
    atc_stop_probs <- rep(NA, n_atc)
    atc_stop_probs[chronic_atc_idx] <- atc_stop_prob_chronic
    atc_stop_probs[acute_atc_idx] <- atc_stop_prob_acute
    # if patient is chronic, reduce stop prob for chronic ATCs
    atc_stop_probs[chronic_atc_idx] <- atc_stop_probs[chronic_atc_idx] * ifelse(is_chronic[i]==1, 0.3, 1)
    atc_stop_probs <- pmin(pmax(atc_stop_probs * atc_stop_factor, 0.01), 0.99)
    
    # Some ATCs stop
    atc_stopped <- ifelse(atc_prev == 1 & runif(n_atc) < atc_stop_probs, 1, 0)
    atc <- pmax(new_atc, atc_prev * (1 - atc_stopped))  # start or persist unless stopped
    atc_prev <- atc
    
    # Save features: age, sex, ses dummies, ICDs, ATCs
    x_array[i, t, ] <- c(age[i], sex[i], ses_dummy[i, ], icd, atc)
  }
}

## Rare ICD and outcome
rare_icd <- rbinom(n_patients, 1, 0.03)
rare_effect <- 2
total_exposure <- apply(x_array[, , 6:n_features], 1, sum)
ses_coefs <- c(0.3, -0.2, 0.1)

prob <- plogis(-5 + 0.02*age + 0.25*sex + as.vector(ses_dummy %*% ses_coefs) +
                 0.05 * total_exposure + rare_effect * rare_icd)
y <- rbinom(n_patients, 1, prob)

## Quick checks
cat("Chronic fraction:", mean(is_chronic), "\n")
table(is_chronic)
summary(prob)
table(y)

# Inspect a few patient trajectories
# patient 1: show ICDs over time
print(x_array[1, , 6:(5+n_icd)])  # ICD1..ICD10 for patient 1 over time
print(x_array[1, , (6+n_icd):n_features]) # ATC1..ATC10 for patient 1 over time



# Split into train/test data
set.seed(42)
train_idx <- sample(1:n_patients, size = 0.8 * n_patients)
x_train <- x_array[train_idx, , ]
y_train <- y[train_idx]
x_test <- x_array[-train_idx, , ]
y_test <- y[-train_idx]



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
model_lstm %>% evaluate(x_test, y_test)

cat("\nGRU Performance:\n")
model_gru %>% evaluate(x_test, y_test)



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





# TODO
# - Decide which methods to look at for prediction modeling
# - Decide which IML techniques to use
# - Implement it and get a feeling for the methods and the simulated data
