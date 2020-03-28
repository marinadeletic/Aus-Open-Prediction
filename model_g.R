library(keras)
library(tidyverse)

# Model G
model_name <- "model_g"

# Import data
X_tr <- read_csv("X_tr.csv") %>% as.matrix()
y_tr <- read_csv("y_tr.csv")
y_tr_labels <- y_tr$outcome %>% as.matrix()

X_va <- read_csv("X_va.csv") %>% as.matrix()
y_va <- read_csv("y_va.csv")
y_va_labels <- y_va$outcome %>% as.matrix()

X_te_error <- read_csv("X_te_error.csv") %>% as.matrix()
y_te_error <- read_csv("y_te_error.csv")
X_te_W <- read_csv("X_te_error.csv")

# Model architecture
model <- keras_model_sequential()
model %>% 
  layer_batch_normalization() %>%
  layer_dense(units = 96,
              activation = 'relu',
              input_shape = c(ncol(X_tr)),
              kernel_regularizer = keras::regularizer_l2(0.0005)) %>%
  layer_dense(units = 48,
              activation = 'relu',
              kernel_regularizer = keras::regularizer_l2(0.0001)) %>%
  layer_dense(units = 1, activation = "sigmoid")

num_ensembles <- 5
tr_probs <- matrix(nrow = nrow(X_tr), ncol=num_ensembles)
va_probs <- matrix(nrow = nrow(X_va), ncol=num_ensembles)
te_probs <- matrix(nrow = nrow(X_te_error), ncol=num_ensembles)

tr_preds <- matrix(nrow = nrow(X_tr), ncol=num_ensembles)
va_preds <- matrix(nrow = nrow(X_va), ncol=num_ensembles)
te_preds <- matrix(nrow = nrow(X_te_error), ncol=num_ensembles)

intermediate_models <- c()

# Initial training
model %>% compile(loss = 'binary_crossentropy', optimizer = optimizer_adam(lr = 0.0001), metrics = c('accuracy'))
history <- model %>% fit(X_tr, y_tr_labels, epochs = 25, batch_size = 64, validation_data = list(X_va, y_va_labels))

for (i in 1:num_ensembles) {
  print(paste("Starting round", i))
  
  model %>% compile(loss = 'binary_crossentropy', optimizer = optimizer_adam(lr = 0.01), metrics = c('accuracy'))
  history <- model %>% fit(X_tr, y_tr_labels, epochs = 5, batch_size = 64, validation_data = list(X_va, y_va_labels))
  model %>% compile(loss = 'binary_crossentropy', optimizer = optimizer_adam(lr = 0.005), metrics = c('accuracy'))
  history <- model %>% fit(X_tr, y_tr_labels, epochs = 10, batch_size = 64, validation_data = list(X_va, y_va_labels))
  model %>% compile(loss = 'binary_crossentropy', optimizer = optimizer_adam(lr = 0.001), metrics = c('accuracy'))
  history <- model %>% fit(X_tr, y_tr_labels, epochs = 10, batch_size = 64, validation_data = list(X_va, y_va_labels))
  model %>% compile(loss = 'binary_crossentropy', optimizer = optimizer_adam(lr = 0.0005), metrics = c('accuracy'))
  history <- model %>% fit(X_tr, y_tr_labels, epochs = 15, batch_size = 64, validation_data = list(X_va, y_va_labels))
  
  tr_probs[,i] <- predict(model, X_tr)
  va_probs[,i] <- predict(model, X_va)
  te_probs[,i] <- predict(model, X_te_error)
  tr_preds[,i] <- round(tr_probs[,i])
  va_preds[,i] <- round(va_probs[,i])
  te_preds[,i] <- round(te_probs[,i])
  
  intermediate_models <- c(intermediate_models, model)
  
  train_acc <- mean(y_tr_labels == tr_preds[,i])
  valid_acc <- mean(y_va_labels == va_preds[,i])
  cat(paste("Sub-model", i, "results:"), "\n")
  cat(paste("Training accuracy  :", round(100*train_acc, 2), "%"), "\n")
  cat(paste("Validation accuracy:", round(100*valid_acc, 2), "%"), "\n")
  cat("\n")
}

for (i in 1:num_ensembles) {
  train_acc <- mean(y_tr_labels == tr_preds[,i])
  valid_acc <- mean(y_va_labels == va_preds[,i])
  cat(paste("Sub-model", i, "results:"), "\n")
  cat(paste("Training accuracy  :", round(100*train_acc, 2), "%"), "\n")
  cat(paste("Validation accuracy:", round(100*valid_acc, 2), "%"), "\n")
  cat("\n")
}

cat("Training correlation matrix:\n")
print(cor(tr_probs))

cat("Validation correlation matrix:\n")
print(cor(va_probs))

# Mean ensembling
tr_mean_ensemble_probs <- rowMeans(tr_probs)
va_mean_ensemble_probs <- rowMeans(va_probs)
te_mean_ensemble_probs <- rowMeans(te_probs)

tr_mean_ensemble_preds <- round(tr_mean_ensemble_probs)
va_mean_ensemble_preds <- round(va_mean_ensemble_probs)
te_mean_ensemble_preds <- round(te_mean_ensemble_probs)

mean_ensemble_train_acc <- mean(tr_mean_ensemble_preds == y_tr_labels)
mean_ensemble_valid_acc <- mean(va_mean_ensemble_preds == y_va_labels)

cat(paste("Mean ensemble results:"), "\n")
cat(paste("Training accuracy  :", round(100*mean_ensemble_train_acc, 2), "%"), "\n")
cat(paste("Validation accuracy:", round(100*mean_ensemble_valid_acc, 2), "%"), "\n")

# Navigate to predictions folder
root_predictions_path <- "./models_predictions/"
model_predictions_path <- paste(root_predictions_path, model_name, sep = "")

if (!file.exists(root_predictions_path)) {
  dir.create(root_predictions_path)
}

if (file.exists(model_predictions_path)) {
  setwd(model_predictions_path)
} else {
  dir.create(model_predictions_path)
  setwd(model_predictions_path)
}

# Make predictions on validation data and test data
training_training_preds <- tr_mean_ensemble_probs %>% as_tibble() %>% rename(V1 = value) %>% mutate(pointid = y_tr$pointid)
training_validation_preds <- va_mean_ensemble_probs %>% as_tibble() %>% rename(V1 = value) %>% mutate(pointid = y_va$pointid)
training_test_preds <- te_mean_ensemble_probs %>% as_tibble() %>% rename(V1 = value) %>% mutate(pointid = y_te_error$pointid)

write_csv(training_training_preds, "training_training_preds.csv")
write_csv(training_validation_preds, "training_validation_preds.csv")
write_csv(training_test_preds, "training_test_preds.csv")

# Save trained model and summary
model_summary_filename <- paste(model_name, "_model_summary.txt", sep = "")
sink(file = model_summary_filename, append = TRUE, type = "output")
model %>% summary()

for (i in 1:num_ensembles) {
  intermediate_models[[i]] %>% save_model_hdf5(paste(model_name, "_", i, ".h5", sep = ""))
}
