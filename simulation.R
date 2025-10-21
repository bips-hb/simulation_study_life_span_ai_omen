
# Simulate EHR data ####
# Simulate longitudinal EHR-like data with a binary cancer outcome

## Load packages ####
library(Matrix)
library(data.table)
library(mvtnorm)
library(truncnorm)
library(tidyverse)

## Load R script with functions ####
source("functions.R")


## Simulate baseline data ####
data_baseline <- simulate_baseline(n = 1000, seed = 26)


ggplot(data = data_baseline, aes(x=age)) + geom_density()
summary(data_baseline$age)



## Simulate visits ####
data_visits <- simulate_visits(baseline = data_baseline, seed = 26)


## Simulate ICD and ATC codes ####
data_ics_atc_codes <- simulate_codes_at_visits(data_visits, seed = 26)


## Aggregate visit-level codes to patient-level features ####
data_aggr <- aggr_codes_to_patient(data_visits, data_ics_atc_codes, agg_fun = "any")


## Simulate laboratory continuous values (time-varying) ####
data_labs <- simulate_labs(data_visits, seed = 26)


## Generate binary cancer outcome at patient-level ####


