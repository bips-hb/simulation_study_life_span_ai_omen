

set.seed(123)

# Set parameters 
n_patients <- 100 
n_timepoints <- 10 
n_icd <- 10 
n_atc <- 10 
n_features <- n_icd + n_atc + 5 # age + sex + 3 SES dummies 

# Simulate static features 
age <- round(runif(n_patients, 18, 90)) 
sex <- rbinom(n_patients, 1, 0.5) 
ses <- sample(c("low", "mid", "high"), n_patients, replace = TRUE, prob = c(0.3, 0.5, 0.2)) 

# One-hot encode SES 
ses_dummy <- model.matrix(~ ses - 1) # columns: seshigh, seslow, sesmid 

# Simulate time-dependent features (ICD + ATC codes) 
x_array <- array(0, dim = c(n_patients, n_timepoints, n_features)) 

for (i in 1:n_patients) { 
  for (t in 1:n_timepoints) { 
  icd <- rbinom(n_icd, 1, 0.1) 
  atc <- rbinom(n_atc, 1, 0.1) 
  
  # Combine features: age, sex, SES dummies, ICDs, ATCs 
  x_array[i, t, ] <- c(age[i], sex[i], ses_dummy[i, ], icd, atc) 
  } 
} 

# Assign SES coefficients (matching order of SES dummies) 
ses_coefs <- c(ses_dummy = c(0.3, -0.2, 0.1)) # e.g., seshigh=0.3, seslow=-0.2, sesmid=0.1 

# Simulate binary outcome 
linear_pred <- 0.03*age + 0.4*sex + x_array[, n_timepoints, 3:5] %*% ses_coefs + 0.2*apply(x_array[, n_timepoints, 6:n_features], 1, sum) 
prob <- 1 / (1 + exp(-linear_pred)) 
y <- rbinom(n_patients, 1, prob)


x_array[1,,]




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
n_patients <- 100
n_timepoints <- 10
n_icd <- 10
n_atc <- 10
n_features <- n_icd + n_atc + 5  # age + sex + 3 SES dummies


## STATIC FEATURES ####
age <- round(runif(n_patients, 18, 90))
sex <- rbinom(n_patients, 1, 0.5) # 0=female, 1=male
ses <- sample(c("low", "mid", "high"), n_patients, replace = TRUE, prob = c(0.3, 0.5, 0.2))
ses_dummy <- model.matrix(~ ses - 1)  # dummy-encoded columns: seshigh, seslow, sesmid


## ARRAYS (3D) ####
x_array <- array(0, dim = c(n_patients, n_timepoints, n_features))


## LONGITUDINAL ICD + ATC CODES ####
for (i in 1:n_patients) {
  
  # Baseline persistence and probabilities depend on age/sex
  icd_prev <- rbinom(n_icd, 1, plogis(-3 + 0.03*(age[i]-50) + 0.3*sex[i]))
  atc_prev <- rbinom(n_atc, 1, plogis(-3 + 0.02*(age[i]-50) + 0.2*sex[i]))
  # -> baseline intercept of -3 gives a low base probability
  # -> age effect pf 0.03 means that each year above 50 increases the log-odds by 0.03
  # -> sex effect of 0.3, so male have a slightly higher risk
  
  for (t in 1:n_timepoints) {
    
    # Probability of new diagnoses increases slightly over time
    new_icd <- rbinom(n_icd, 1, plogis(-4 + 0.04*(age[i]-50) + 0.3*sex[i] + 0.1*t))
    # -> base log-odds of -4, very low prob. at the start
    # -> the older the higher the odds
    # -> males have higher odds
    # -> slightly higher risk over time (diseases accumulate)
    
    # Chronic persistence: once you have a diagnosis, you tend to keep it
    icd <- pmax(icd_prev, new_icd)
    icd_prev <- icd
    
    # Medications depend on ICDs and also persist
    new_atc <- rbinom(n_atc, 1, plogis(-4 + 0.3*icd + 0.1*t))
    # -> each ATC code has a higher probability if the related ICD is present
    # -> so if an ICD is active, medication probability increases
    
    atc <- pmax(atc_prev, new_atc)
    atc_prev <- atc
    # -> medications often continue for multiple timepoints once prescribed
    # -> take element-wise maximum
    
    x_array[i, t, ] <- c(age[i], sex[i], ses_dummy[i, ], icd, atc)
    
  }
  
}

## RARE EVENTS (e.g., rare ICD strongly linked to cancer) ####
rare_icd <- rbinom(n_patients, 1, 0.03)
rare_effect <- 2  # strong impact

## OUTCOME: CANCER ####
# Outcome depends on demographics + cumulative ICD/ATC exposure + rare event
cum_exposure <- apply(x_array[, , 6:n_features], c(1,3), sum)   # sum over time
total_exposure <- apply(cum_exposure, 1, sum)

# SES effects (example coefficients)
ses_coefs <- c(0.3, -0.2, 0.1)

linear_pred <- 0.03*age + 0.4*sex +
  as.vector(ses_dummy %*% ses_coefs) +
  0.05*total_exposure +
  rare_effect*rare_icd

prob <- 1 / (1 + exp(-linear_pred))
y <- rbinom(n_patients, 1, prob)

x_array


dim(x_array) # n_patients × n_timepoints × n_features
length(y) # n_patients





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
