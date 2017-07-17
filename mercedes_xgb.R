#### kaggle mercedes benz comp. ####
# I tried sparse matrix transformation amd one hot encoding. Sparse matrix result's was better than numerical transformation.

setwd("Desktop/kaggle_mercedes/")

train_df <- read.csv("train.csv")
test_df <- read.csv("test.csv")

str(train_df)
head(train_df)
summary(train_df)


numeric_var <- names(train_df)[which(sapply(train_df, is.numeric))]
cat_var <- names(train_df)[which(sapply(train_df, is.factor))]
colSums(sapply(train_df[, cat_var], is.na))
colSums(sapply(train_df[, numeric_var], is.na))
summary(train_df[, numeric_var])
summary(train_df[, cat_var])

colSums(sapply(train_df, is.na))
std <- sapply(train_df, sd)
which(std == 0)


library(ggplot2)
ggplot(train_df, aes(train_df$y)) + geom_histogram(aes(fill = ..count..)) 

#### xgboost numeric transformation ####

train_df_model <- train_df
y_train <- train_df_model$y
train_df_model$y <- NULL

train_test <- rbind(train_df_model, test_df)
ntrain <- nrow(train_df_model)

features <- names(train_df)

#convert character into integer
for (f in features) {
  if (is.character(train_test[[f]])) {
    levels = sort(unique(train_test[[f]]))
    train_test[[f]] = as.integer(factor(train_test[[f]],levels = levels))
  }
}

#splitting whole data back again
train_x <- train_test[1:ntrain,]
test_x <- train_test[(ntrain + 1):nrow(train_test),]

#convert into numeric for XGBoost implementation
train_x[] <- purrr::map(train_x, as.numeric)
test_x[] <- purrr::map(test_x, as.numeric)

dtrain_int <- xgb.DMatrix(as.matrix(train_x),label = y_train)
dtest_int <- xgb.DMatrix(as.matrix(test_x))

system.time(
  cvmodel_int <- xgb.cv(data = dtrain_int, booster = "gblinear",
                    nfold = 10,
                    nrounds = 5000,
                    max_depth = 4,
                    eta = 0.05,
                    subsample = 0.5,
                    colsample_bytree = 0.5,
                    metrics = "rmse",
                    maximize = FALSE,
                    early_stopping_rounds = 25,
                    objective = "reg:linear",
                    print_every_n = 10,
                    verbose = TRUE)
)



system.time(
  temp_model_int <- xgb.train(data = dtrain_int, 
                          nrounds = cvmodel_int$best_iteration,
                          max_depth = 4,
                          subsample = 0.5,
                          colsample_bytree = 0.5,
                          eta = 0.05,  
                          objective = "reg:linear",
                          print_every_n = 10,
                          ))

predictedValues_int <- predict(temp_model_int, dtest_int)
head(predictedValues_int)


#### xgboost sparse matrix ####
library(Matrix)
target <- train_df$y
train_df$y <- NULL
data <- rbind(train_df, test_df)
data$ID <- NULL 
gc(verbose = FALSE)

data_sparse <- sparse.model.matrix(~.-1, data = as.data.frame(data))
cat("Data size: ", data_sparse@Dim[1], " x ", data_sparse@Dim[2], "  \n", sep = "")
rm(data)

library(xgboost)


dtrain <- xgb.DMatrix(data = data_sparse[1:nrow(train_df), ], label = target) 
dtest <- xgb.DMatrix(data = data_sparse[(nrow(train_df)+1):nrow(data), ]) 

system.time(
  cvmodel <- xgb.cv(data = dtrain, booster = "gblinear",
                       nfold = 10,
                       nrounds = 5000,
                       max_depth = 4,
                       eta = 0.05,
                       subsample = 0.5,
                       colsample_bytree = 0.5,
                       metrics = "rmse",
                       maximize = FALSE,
                       early_stopping_rounds = 25,
                       objective = "reg:linear",
                       print_every_n = 10,
                       verbose = TRUE)
)



system.time(
  temp_model <- xgb.train(data = dtrain, 
                          nrounds = cvmodel$best_iteration,
                          max_depth = 4, subsample = 0.5, colsample_bytree = 0.5, 
                          eta = 0.05,  
                          objective = "reg:linear",
                          print_every_n = 10, 
                          verbose = TRUE, booster = "gbtree",
                          watchlist = list(train = dtrain)))


predictedValues <- predict(temp_model, dtest)
head(predictedValues)
ggplot() + aes(predictedValues) + geom_histogram(aes(fill = ..count..))


submission <- data.table::fread("sample_submission.csv", header = TRUE, showProgress = FALSE, data.table = FALSE)
submission$y <- predictedValues
write.csv(submission, "xgboost_cv_v2.csv", row.names = FALSE)

importance <- xgb.importance(feature_names = data_sparse@Dimnames[[2]], 
                             model = temp_model)

xgb.ggplot.deepness(temp_model)
xgb.ggplot.deepness(temp_model, which='max.depth')
xgb.ggplot.deepness(temp_model, which='med.depth')
xgb.ggplot.deepness(temp_model, which='med.weight')


xgb.ggplot.importance(top_n = 10, importance_matrix = importance)
xgb.model.dt.tree(feature_names = data_sparse@Dimnames[[2]], model = temp_model, 
                  n_first_tree = 1)
xgb.plot.tree(feature_names = data_sparse@Dimnames[[2]], model = temp_model, n_first_tree = 2)
DT::datatable(importance)



