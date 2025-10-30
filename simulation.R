
# Structure of sequence data

# RNNs are designed to handle sequences, so data where order matters and past
# information influences the future

# Structure of sequence data: 3D tensor
# (number of patients/batch size, number of time steps, number of features)


## Load packages ####
library(keras3)


set.seed(123)

# Set parameters
n_patients <- 100 # number of patients
n_timepoints <- 5 # sequence length
n_icd <- 5 # number of ICD codes
n_atc <- 5 # number of ATC codes
n_features <- n_icd + n_atc + 2 # add 2 for age and sex

# Simulate static features
age <- round(runif(n_patients, 18, 90)) # age between 18 and 90
sex <- rbinom(n_patients, 1, 0.5) # 0 = female, 1 = male

# Simulate time-dependent features (ICD + ATC codes)
x_array <- array(0, dim = c(n_patients, n_timepoints, n_features)) 

for (i in 1:n_patients) {
  
  for (t in 1:n_timepoints) {
    
    icd <- rbinom(n_icd, 1, 0.1)
    atc <- rbinom(n_atc, 1, 0.1)
    
    # Combine all features
    x_array[i, t, ] <- c(age[i], sex[i], icd, atc)

  }
  
}

x_array



# Simulate binary outcome
linear_pred <- 0.03*age + 0.4*sex + 0.2*apply(x_array[,n_timepoints, 3:n_features], 1, sum)
prob <- 1 / (1 + exp(-linear_pred))   # logistic transformation
y <- rbinom(n_patients, 1, prob)     # binary outcome 0/1

dim(x_array) # n_patients × n_timepoints × n_features
length(y) # n_patients





# Split into train/test data
set.seed(42)
train_idx <- sample(1:n_patients, size = 0.8 * n_patients)
x_train <- x_array[train_idx, , ]
y_train <- y[train_idx]
x_test <- x_array[-train_idx, , ]
y_test <- y[-train_idx]



# Build RNN model
model <- keras_model_sequential() %>%
  layer_simple_rnn(units = 16, input_shape = c(n_timepoints, n_features), activation = "tanh") %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

summary(model)



# Train model
history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 16,
  validation_split = 0.2,
  verbose = 1
)

# Evaluate
model %>% evaluate(x_test, y_test)




# TODO
# - Decide which methods to look at for prediction modeling
# - Decide which IML techniques to use
# - Implement it and get a feeling for the methods and the simulated data
