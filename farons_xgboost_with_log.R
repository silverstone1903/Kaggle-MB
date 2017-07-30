# Kaggle Allstate Severity Claims competition - Farons Python code ported to R. So not my code!
# Here is the best model which gave me to the best LB score. (0.55555)
# Used log transformation for the target (y) variable.
# Dropped the constant variables. (X11, X93, X107, X233, X235, X268, X289, X290, X293, X297, X330, X347)
# Used rmse for watching results. Also added R^2 function.


setwd("../Desktop/kaggle_mercedes/")
library(data.table)
library(Matrix)
library(xgboost)
library(Metrics)
library(MLmetrics)

ID = ID
TARGET = y
SEED = 2017

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "sample_submission.csv"


train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)

y_train = log(train[,TARGET, with = FALSE])[[TARGET]]

train[, c(ID, TARGET) := NULL]
test[, c(ID) := NULL]

ntrain = nrow(train)
train_test = rbind(train, test)
a <- c(X11, X93, X107, X233, X235, X268, X289, X290, X293, X297, X330, X347)
train_test[,c(a) := NULL]

features = names(train)

for (f in features) {
  if (class(train_test[[f]])=="character") {
    levels <- unique(train_test[[f]])
    train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
  }
}


x_train = train_test[1:ntrain,]
x_test = train_test[(ntrain+1):nrow(train_test),]


x_train[] <- lapply(x_train, as.numeric)
x_test[] <- lapply(x_test, as.numeric)

dtrain = xgb.DMatrix(as.matrix(x_train), label = y_train)
dtest = xgb.DMatrix(as.matrix(x_test))


xg_eval_rmse <- function (yhat, dtrain) {
  y <- getinfo(dtrain, "label")
  err <- rmse(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

xg_R_squared <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= R2_Score(yhat, y)
  return (list(metric = "error", value = err))
}

xgb_params = list(
  seed = 2017,
  colsample_bytree = 0.5,
  subsample = 0.9,
  eta = 0.01,
  objective = reg:linear,
  max_depth = 4,
  num_parallel_tree = 1,
  min_child_weight = 1
  )

res <- xgb.cv(xgb_params,
             dtrain,
             nrounds = 5000,
             nfold = 10,
             early_stopping_rounds = 25,
             print_every_n = 10,
             verbose= 1,
             feval = xg_R_squared,
             maximize = T, nthread = 4)

best_nrounds <- res$best_iteration
cv_mean = res$evaluation_log$test_error_mean[best_nrounds]
cv_std = res$evaluation_log$test_error_std[best_nrounds]
cat(paste0(CV-Mean: ,cv_mean, , cv_std))

gbdt <- xgb.train(xgb_params, dtrain, best_nrounds, nthread = 4)

submission = fread(SUBMISSION_FILE, colClasses = c("integer", "numeric"))
submission$y = exp(predict(gbdt,dtest))
write.csv(submission,xgb_starter_v3.csv,row.names = FALSE)


xgb.ggplot.deepness(gbdt)
xgb.ggplot.deepness(gbdt, which=max.depth)
xgb.ggplot.deepness(gbdt, which=med.depth)
xgb.ggplot.deepness(gbdt, which=med.weight)

importance <- xgb.importance(feature_names = colnames(train_test), 
                             model = gbdt)
xgb.ggplot.importance(top_n = 10, importance_matrix = importance)
DT::datatable(importance)
