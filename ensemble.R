library(tidyverse)
library(reshape2)

# Read in the pointids that were identified to be winners
X_te_W <- read_csv("X_te_W.csv") %>%
  distinct() %>%
  mutate(outcome = "W")

model_names <- c("model_a", "model_b", "model_c", "model_d", "model_e", "model_f", "model_g")

test_preds <- lapply(model_names, function(n) { read_csv(paste("models_predictions/", n, "/training_test_preds.csv", sep = "")) %>% mutate(model = n) })
test_preds <- do.call(rbind, test_preds)

# Search for marginal points
test_preds %>%
  group_by(pointid, model) %>%
  summarise(prob = mean(V1)) %>%
  dcast(pointid ~ model) %>%
  arrange(abs(model_a-0.5))

# Final predictions
test_preds %>%
  group_by(pointid, model) %>%
  summarise(prob = mean(V1)) %>%
  summarise(outcome = round(mean(prob))) %>%
  mutate(outcome = ifelse(outcome == 0, "F", "U")) %>%
  rbind(X_te_W) -> out_preds

write_csv(out_preds, "ensemble_predictions.csv")
