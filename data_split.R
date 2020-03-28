library(tidyverse)
library(caret)
library(data.table)
library(mltools)
library(readr)

# Load raw data sets
load("ao_training.rda")
load("ao_test_unlabelled.rda")

# Add winner indicator 
ao_tr$winner<-((ao_tr$rally %% 2 == 0)*(ao_tr$serverwon == FALSE)) + 
  ((ao_tr$rally %% 2 != 0)*(ao_tr$serverwon == TRUE))
ao_ts_unlabelled$winner<-((ao_ts_unlabelled$rally %% 2 == 0)*(ao_ts_unlabelled$serverwon == FALSE)) + 
  ((ao_ts_unlabelled$rally %% 2 != 0)*(ao_ts_unlabelled$serverwon == TRUE))

# Filter out winners - Training 
X_tr_full <- ao_tr %>% filter(ao_tr$winner != 1) %>% select(-event, -year, -pointid, -outcome, -matchid)
y_tr_full <- ao_tr %>% filter(ao_tr$winner != 1) %>% select(pointid,outcome)
X_tr_W <- ao_tr %>% filter(winner == 1) %>% select(pointid)
y_tr_full$outcome <- as.numeric(y_tr_full$outcome) - 1

# Filter out winners - Test 
X_te_error <- ao_ts_unlabelled %>% filter(winner != 1) %>% select(-event, -year, -pointid, -matchid) 
y_te_error <- ao_ts_unlabelled %>% filter(winner != 1) %>% select(pointid)
X_te_W <- ao_ts_unlabelled %>% filter(winner == 1) %>% select(pointid)
  
# Combine all variables from training and test set 
full_X <- rbind(X_tr_full, X_te_error)

# Convert variables to factors
full_X$serve <- as.factor(full_X$serve)
full_X$hitpoint <- as.factor(full_X$hitpoint)
full_X$outside.sideline <- as.factor(full_X$outside.sideline)
full_X$outside.baseline <- as.factor(full_X$outside.baseline)
full_X$previous.hitpoint <- as.factor(full_X$previous.hitpoint)

# Add manual feature variables based on distribution analysis
full_X$depthf1 <- ifelse(2 < full_X$depth && full_X$depth < 6, 1, 0) # 2 < depth < 6
full_X$spd_lessf2 <- ifelse(full_X$speed < 25, 1, 0) # speed < 25
full_X$spd_bigf3 <- ifelse(full_X$speed > 42, 1, 0) # speed > 42
full_X$opdist_f4 <- ifelse(full_X$opponent.distance.from.center > 2.8, 1, 0) # opponent.distance.from.center > 2.8
full_X$rall_f5 <- ifelse(full_X$rally > 1, 1, 0) # rally > 1
full_X$serveFlt_f6 <- (full_X$serverwon==FALSE)*(full_X$shotinrally ==1)*(full_X$rally == 1) # service fault 

# One-hot encoding for categorical features
full_X <- full_X %>% as.data.table() %>% one_hot() %>% as.matrix()

# Split back into original training and test sets
X_tr_full <- full_X[1:nrow(X_tr_full),]
X_te_error <- full_X[(nrow(X_tr_full)+1):nrow(full_X),]

# Split full training set into training and test
set.seed(1)
tr_idx <- createDataPartition(y_tr_full$outcome, p=0.20)
X_tr <- X_tr_full[-tr_idx$Resample1,]
X_va <- X_tr_full[tr_idx$Resample1,]

# Create y dataframes containing pointid and outcome 
y_tr <- y_tr_full[-tr_idx$Resample1,]
y_va <- y_tr_full[tr_idx$Resample1,]

# Export files
write_csv(as.data.frame(X_tr), path="X_tr.csv")
write_csv(as.data.frame(X_va), path="X_va.csv")
write_csv(as.data.frame(y_tr), path="y_tr.csv")
write_csv(as.data.frame(y_va), path="y_va.csv")
write_csv(as.data.frame(X_te_error), path="X_te_error.csv")
write_csv(as.data.frame(y_te_error), path="y_te_error.csv")
write_csv(as.data.frame(X_te_W), path="X_te_winner.csv")
write_csv(as.data.frame(X_te_W), path="X_tr_winner.csv")