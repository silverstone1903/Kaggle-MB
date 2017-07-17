#### xgboost grid search function ####
setwd("../Desktop/kaggle_mercedes/")
library(xgboost)
library(Matrix)

train_df <- read.csv("train.csv")
test_df <- read.csv("test.csv")

target <- train_df$y
train_df$y <- NULL
data <- rbind(train_df, test_df)
data$ID <- NULL 
gc(verbose = FALSE)

data_sparse <- sparse.model.matrix(~.-1, data = as.data.frame(data))
cat("Data size: ", data_sparse@Dim[1], " x ", data_sparse@Dim[2], "  \n", sep = "")


dtrain <- xgb.DMatrix(data = data_sparse[1:nrow(train_df), ], label = target) 
dtest <- xgb.DMatrix(data = data_sparse[(nrow(train_df)+1):nrow(data), ]) 




searchGridSubCol <- expand.grid(subsample = seq(0.5, 0.9, 0.1), 
                                colsample_bytree = seq(0.5, 0.9, 0.1),
                                max_depth = c(3,4,5),
                                min_child = seq(1, 5, 1), eta = seq(0.01, 0.05, 0.01)
)

ntrees <- 1000

system.time(
rmseErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
  
  #Extract Parameters to test
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]
  currentDepth <- parameterList[["max_depth"]]
  currentEta <- parameterList[["eta"]]
  currentMinChild <- parameterList[["min_child"]]
  
  
  
  
  xgboostModelCV <- xgb.cv(data =  dtrain, nrounds = ntrees, nfold = 10, showsd = TRUE, 
                       metrics = "rmse", verbose = TRUE, "eval_metric" = "rmse",
                     "objective" = "reg:linear", "max.depth" = currentDepth, "eta" = currentEta,                               
                     "subsample" = currentSubsampleRate, "colsample_bytree" = currentColsampleRate
                      , print_every_n = 10, "min_child_weight" = currentMinChild, booster = "gbtree",
                     early_stopping_rounds = 10)
  
   xvalidationScores <- as.data.frame(xgboostModelCV$evaluation_log)
  rmse <- tail(xvalidationScores$test_rmse_mean, 1)
  trmse <- tail(xvalidationScores$train_rmse_mean,1)
  output <- return(c(rmse, trmse, currentSubsampleRate, currentColsampleRate, currentDepth, currentEta, currentMinChild))
  
}))

output <- as.data.frame(t(rmseErrorsHyperparameters))
varnames <- c("TestRMSE", "TrainRMSE", "SubSampRate", "ColSampRate", "Depth", "eta", "currentMinChild")
names(output) <- varnames
head(output)
write.csv(output, "xgb_gridsearch_results.csv")
