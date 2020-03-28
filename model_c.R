library(keras)
library(tidyverse)

# Model C
model_name <- "model_c"

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
  layer_dense(units = 96,
              activation = 'relu',
              input_shape = c(ncol(X_tr)),
              kernel_regularizer = keras::regularizer_l1_l2(l1 = 0.0001, l2 = 0.0001)) %>%
  layer_dense(units = 64,
              activation = 'relu',
              kernel_regularizer = keras::regularizer_l1_l2(l1 = 0.0001, l2 = 0.0001)) %>%
  layer_dense(units = 32,
              activation = 'relu',
              kernel_regularizer = keras::regularizer_l1_l2(l1 = 0.0001, l2 = 0.0001)) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(lr = 0.001),
  metrics = c('accuracy')
)

# Fit the model on training data
history <- model %>% fit(
  X_tr, y_tr_labels, 
  epochs = 15, batch_size = 32,
  validation_data = list(X_va, y_va_labels)
)

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
training_training_preds <- predict(model, X_tr) %>% as_tibble() %>% mutate(pointid = y_tr$pointid)
training_validation_preds <- predict(model, X_va) %>% as_tibble() %>% mutate(pointid = y_va$pointid)
training_test_preds <- predict(model, X_te_error) %>% as_tibble() %>% mutate(pointid = y_te_error$pointid)

write_csv(training_training_preds, "training_training_preds.csv")
write_csv(training_validation_preds, "training_validation_preds.csv")
write_csv(training_test_preds, "training_test_preds.csv")

# Save trained model and summary
model %>% save_model_hdf5(paste(model_name, ".h5", sep = ""))

model_summary_filename <- paste(model_name, "_model_summary.txt", sep = "")
sink(file = model_summary_filename, append = TRUE, type = "output")
model %>% summary()
