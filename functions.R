
# Simulate EHR data ####
# Simulate longitudinal EHR-like data with a binary cancer outcome

# Each simulated patients has:
#   - Baseline information (demographics, socioeconomic status)
#   - Multiple visits over time
#   - Medical codes (ICD for diagnoses, ATC for prescriptions)
#   - Lab measurements
#   - Binary cancer outcome (1 = cancer, 0 = no cancer)



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

