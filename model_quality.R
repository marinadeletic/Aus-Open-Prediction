library(tidyverse)
library(reshape2)
library(MLmetrics)
library(pROC)

# Load labels
y_tr <- read_csv("y_tr.csv")
y_tr_labels <- y_tr$outcome %>% as.matrix()

y_va <- read_csv("y_va.csv")
y_va_labels <- y_va$outcome %>% as.matrix()

# Define function to calculate metrics
get_quality <- function(preds, labels, is_training) {
  probs <- preds$V1
  model_name <- preds$model[1]
  
  pred_labels <- round(probs)
  acc <- mean(pred_labels == labels)
  f1 <- F1_Score(labels, pred_labels)
  roc_auc <- auc(labels, probs)
  
  if (is_training) {
    data.frame(model = model_name, train_acc = acc, train_f1 = f1, train_roc_auc = roc_auc)
  } else {
    data.frame(model = model_name, valid_acc = acc, valid_f1 = f1, valid_roc_auc = roc_auc)
  }
}

# Model names to be assessed
model_names <- c("model_a", "model_b", "model_c", "model_d", "model_e", "model_f", "model_g")

# Calcualate metrics
training_preds <- lapply(model_names, function(n) { read_csv(paste("models_predictions/", n, "/training_training_preds.csv", sep = "")) %>% mutate(model = n) })
training_metrics <- lapply(training_preds, function(preds) { get_quality(preds, y_tr_labels, is_training = TRUE) })
training_metrics <- do.call(rbind, training_metrics)

validation_preds <- lapply(model_names, function(n) { read_csv(paste("models_predictions/", n, "/training_validation_preds.csv", sep = "")) %>% mutate(model = n) })
validation_metrics <- lapply(validation_preds, function(preds) { get_quality(preds, y_va_labels, is_training = FALSE) })
validation_metrics <- do.call(rbind, validation_metrics)

# Save metrics to file
metrics <- left_join(training_metrics, validation_metrics, by = "model")
metrics

write_csv(metrics, "ensemble_metrics.csv")
