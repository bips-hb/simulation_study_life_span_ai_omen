
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


