#### h2o algorithms ####
# I also used h2o package but it didn't perform well. xgboost rulezz

library(h2o)
setwd("../Desktop/kaggle_mercedes/")

train_df <- read.csv("train.csv")
test_df <- read.csv("test.csv")
h2o.init(nthreads = -1)


train_h2o <- as.h2o(train_df)
test_h2o <- as.h2o(test_df)
rm(train_df)
rm(test_df)

splits <- h2o.splitFrame(
  data = train_h2o, 
  ratios = c(0.8),   ## only need to specify 2 fractions, the 3rd is implied
  destination_frames = c("train.hex", "valid.hex"), seed = 2017
)
train <- splits[[1]]
valid <- splits[[2]]



a <- colnames(train_h2o)
x <- a[3:378]
target <- "y"

system.time(
  gbm <- h2o.gbm(x, y = target,
                 distribution = "gaussian",
                 training_frame = train,
                 validation_frame = valid,
                 stopping_metric = "RMSE",
                 ntrees = 1000,
                 max_depth = 5,
                 learn_rate = 0.05,
                 stopping_rounds = 5,
                 stopping_tolerance = 0.01,
                 sample_rate = 0.7,   
                 col_sample_rate = 0.7,                                                   
                 seed = 2017,    
                 score_tree_interval = 50,
                 nfolds = 10, score_each_iteration = T))

gbm
summary(gbm)
gbm@model$validation_metrics
gbm@model$training_metrics
gbm@model$scoring_history
gbm@allparameters
h2o.rmse(h2o.performance(gbm, valid = TRUE))


h2o.performance(gbm)
h2o.performance(gbm, valid = T)
plot(gbm)

pred <- h2o.predict(gbm, test_h2o)
head(pred)
class(pred)
submissionFrame <- h2o.cbind(test_h2o$id, pred)
submissionFrame <- as.data.frame(submissionFrame)
colnames(submissionFrame) <- c("ID","y")
head(submissionFrame)
h2o.exportFile(submissionFrame, path = "h2o_mb_gbm.csv")


submission <- data.table::fread("sample_submission.csv", header = TRUE, showProgress = FALSE, data.table = FALSE)
submission$y <- as.vector(pred)
write.csv(submission, "h2o_gbm.csv", row.names = FALSE)
data.table::fwrite(submission, file = "h2o_gbm_2.csv")


#### h2o deep learning ####
system.time(
  dl <- h2o.deeplearning(x, y = target,
                         training_frame = train,
                         validation_frame = valid,
                         rate = 0.01,
                         distribution = "gaussian",
                         stopping_rounds = 10,
                         stopping_metric = "RMSE",
                         epoch = 100,
                         hidden = c(40,50,60),
                         activation = "Rectifier",
                         seed = 1903
  ) )

h2o.mse(h2o.performance(dl, valid = TRUE))
h2o.performance(dl)
system.time(predict.dl <- (h2o.predict(dl,test)))

head(predict.dl)



#### h2o random forest ####
system.time(
  rf <- h2o.randomForest(x, 
                         target, 
                         training_frame = train,
                         validation_frame = valid,
                         ntrees = 3000, 
                         max_depth = 5, 
                         seed = 2017, 
                         sample_rate = 0.7, 
                         nfolds = 10, 
                         stopping_metric = "RMSE",
                         col_sample_rate_per_tree = 0.7,
                         stopping_rounds = 10, score_each_iteration = T,stopping_tolerance = 0.01
                         ))


rf
summary(rf)
rf@model$training_metrics
rf@model$validation_metrics
rf@model$scoring_history
h2o.rmse(h2o.performance(rf, valid = T))
h2o.performance(rf)
rfvimp <- h2o.varimp(rf)
predict.rforest <- h2o.predict(rf, test)
head(predict.rforest)

#submission
class(predict.rforest)
submissionFrame<-h2o.cbind(test$id,predict.rforest)
colnames(submissionFrame)<-c("id","loss")
head(submissionFrame)
h2o.exportFile(submissionFrame,path="h2o_mercedes_rf_3105.csv")
