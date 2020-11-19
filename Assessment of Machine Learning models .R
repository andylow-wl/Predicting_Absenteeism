data <- read.csv(file.choose(), header =TRUE) #THIS IS FOR OVERALL DATA
mydata <-read.csv(file.choose(), header =TRUE) #THIS IS FOR TRAIN DATA
test<-read.csv(file.choose(), header =TRUE) #THIS IS FOR TEST DATA

linearModel<-glm(mydata$Absenteeism.time.in.hours~., data = mydata)
summary(linearModel)

##############################################################################

## 4.2.2 Feature Selection using Hypothesis Testing(T-test and chisquare) ##

set.seed(123)

#t-test(for continuous variables)

cont=c(1,6,7,8,9,10,11,14,17,18,19)

for(i in cont){
  print(colnames(mydata[i]))
  print(t.test(mydata[,i],mydata$Absenteeism.time.in.hours)$p.value)
}

#chi-square test(for categorical variables)

install.packages("MASS")
library(MASS)
cat= c(2,3,4,5,12,13,15,16)
for(i in cat){
  tbl=table(mydata$Absenteeism.time.in.hours,mydata[,i])
  print(colnames(mydata[i]))
  print(chisq.test(tbl)$p.value)
}

##############################################################################

## 4.2.3 Feature Selection using Information Gain ##

install.packages("FSelector")
library(FSelector)

set.seed(123)

information.gain(Absenteeism.time.in.hours~.,data=mydata)


##############################################################################

## 4.3.1 Feature Selection using Stepwise Forward and Backward Selection ##

set.seed(123)

base.mod<-lm(Absenteeism.time.in.hours~1,data=mydata)

all.mod<-lm(Absenteeism.time.in.hours~.,data=mydata)

stepMod <- step(base.mod, scope=list(lower=base.mod,
                                     upper=all.mod), direction="both", trace=0, steps=1000)
forwardMod <- step(base.mod, scope=list(lower=base.mod,
                                        upper=all.mod), direction="forward", trace=0, steps=1000)
backwardMod <- step(all.mod, scope=list(lower=base.mod,
                                        upper=all.mod), direction="backward", trace=0, steps=1000)

vars_step <- names(unlist(stepMod[[1]]))
vars_step <- shortlistedVars_step[!shortlistedVars_step %in%
                                    "(Intercept)"] 
vars_forward <- names(unlist(forwardMod[[1]]))
vars_forward <- shortlistedVars_forward[!shortlistedVars_forward
                                        %in% "(Intercept)"] 
vars_backward <- names(unlist(backwardMod[[1]]))
vars_backward <-
  shortlistedVars_backward[!shortlistedVars_backward %in%
                             "(Intercept)"] 
print(vars_step)
print(vars_forward)
print(vars_backward)

##############################################################################

## 4.3.2 Feature Selection using RFE ##

library(caret)
set.seed(123)
control <- rfeControl(functions = rfFuncs, method = 'cv', number = 10, allowParallel = TRUE, verbose = TRUE)
results <- rfe(mydata[,1:19], mydata$Absenteeism.time.in.hours, sizes = c(1:1), rfeControl = control)

results

##############################################################################

## 4.4.1 Feature Selection using LASSO ##

library(glmnet)
set.seed(123)
feat_mod_select <- cv.glmnet(as.matrix(mydata[,1:19]) ,
                             mydata[, 20], standardize = TRUE, alpha =1)
as.matrix(round(coef(feat_mod_select,
                     feat_mod_select$lambda.min),5))

plot(feat_mod_select)

##############################################################################

## 4.4.2 Feature Selection using Boruta ##

library(Boruta)
set.seed(123)

boruta_output <- Boruta(Absenteeism.time.in.hours~.,data=na.omit(mydata),doTrace=0)

names(boruta_output)

boruta_signif<- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)

roughFixMod<- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

imps<-attStats(roughFixMod)
imps2= imps[imps$decision!='Rejected', c('meanImp','decision')]
head(imps2[order(-imps2$meanImp),])

plot(boruta_output, cex.axis=7 , las=2, xlab="", main = "Variable Importance")

##############################################################################

## Feature Selection using Random Forest ##

library(caret)

control<-trainControl(method="cv", number=10, verboseIter= TRUE , allowParallel = TRUE)

rfMod <-train(y=mydata[,20],x=mydata[,1:19], method="rf", preProcess="scale", do.trace= TRUE, importance=T, ntrees=5000, trControl=control)

rfMod

postResample(predict(rfMod, newdata= test[,1:19]),test[,20])

rfImp <- varImp(rfMod, scale = FALSE)

rfImp

##############################################################################

## 5.2.1 SVM Model Kernel Selection ##

library(e1071)
library(Metrics)
set.seed(123)

model_svm_linear<-svm(Absenteeism.time.in.hours~.,data=mydata,type="C-classification",kernel="linear")
pred_linear<-predict(model_svm_linear,test)
table(pred=pred_linear,actual=test$Absenteeism.time.in.hours)
auc(pred_linear,test$Absenteeism.time.in.hours)
mean(test$Absenteeism.time.in.hours==pred_linear)

model_svm_polynomial <-svm(Absenteeism.time.in.hours~.,data=mydata,type="C-classification",kernel= "polynomial")
pred_polynomial<-predict(model_svm_polynomial,test)
table(pred=pred_polynomial,actual=test$Absenteeism.time.in.hours)
auc(pred_polynomial,test$Absenteeism.time.in.hours)
mean(test$Absenteeism.time.in.hours==pred_polynomial)

model_svm_radial <-svm(Absenteeism.time.in.hours~.,data=mydata,type="C-classification",kernel= "radial")
pred_radial<-predict(model_svm_radial,test)
table(pred=pred_radial,actual=test$Absenteeism.time.in.hours)
auc(pred_radial,test$Absenteeism.time.in.hours)
mean(test$Absenteeism.time.in.hours==pred_radial)

model_svm_sigmoid <-svm(Absenteeism.time.in.hours~.,data=mydata,type="C-classification",kernel= "sigmoid")
pred_sigmoid<-predict(model_svm_sigmoid,test)
table(pred=pred_sigmoid,actual=test$Absenteeism.time.in.hours)
auc(pred_sigmoid,test$Absenteeism.time.in.hours)
mean(test$Absenteeism.time.in.hours==pred_sigmoid)

##############################################################################

## 5.3.1 Random Forest Model Evaluation ##

library(randomForest)
set.seed(123)

# Feature Selection from Hypothesis Testing
rf_hypotest <- randomForest(y=mydata[,20], x =mydata[,c(1,2,3,5,6,7,8,9,10,11,12,14,15,17,18,19)], ytest = test[,20], xtest = test[,c(1,2,3,5,6,7,8,9,10,11,12,14,15,17,18,19)], importance = TRUE)
rf_hypotest

# Feature Selection from Information Gain
rf_infogain <- randomForest(y=mydata[,20], x =mydata[,c(2,6,12)], ytest = test[,20], xtest = test[,c(2,6,12)], importance = TRUE)
rf_infogain

# Feature Selection from Stepwise
rf_stepwise <- randomForest(y=mydata[,20], x =mydata[,c(2,4,5,12,14,15,19)], ytest = test[,20], xtest = test[,c(2,4,5,12,14,15,19)], importance = TRUE)
rf_stepwise

# Feature Selection from RFE
rf_rfe <- randomForest(y=mydata[,20], x =mydata[,c(2,5,8,12,19)], ytest = test[,20], xtest = test[,c(2,5,8,12,19)], importance = TRUE)
rf_rfe

# Feature Selection from LASSO
rf_lasso <- randomForest(y=mydata[,20], x =mydata[,c(1,2,3,4,5,6,7,9,10,11,12,13,14,15,17,19)], ytest = test[,20], xtest = test[,c(1,2,3,4,5,6,7,9,10,11,12,13,14,15,17,19)], importance = TRUE)
rf_lasso

# Feature Selection from Boruta Algorithm
rf_boruta <- randomForest(y=mydata[,20], x =mydata[,c(2,6,8,9,12,19)], ytest = test[,20], xtest = test[,c(2,6,8,9,12,19)], importance = TRUE)
rf_boruta

# Feature Selection from RF
rf_rf <- randomForest(y=mydata[,20], x =mydata[,c(2,12)], ytest = test[,20], xtest = test[,c(2,12)], importance = TRUE)
rf_rf

##############################################################################

## 5.4.1 Random Forest Model Evaluation ##

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

maxmindf <- as.data.frame(lapply(mydata, normalize))
library(neuralnet)
nn <- neuralnet(Absenteeism.time.in.hours ~ ., data=mydata, hidden=c(2,1), linear.output=FALSE, threshold=0.01)
plot(nn)
nn$result.matrix

#Test the resulting output
temp_test <- subset(test, select = c(1:20))
head(temp_test)
nn.results <- compute(nn, temp_test)
results <- data.frame(actual = test$Absenteeism.time.in.hours, prediction = nn.results$net.result)

# Model Accuracy (Not working)
predicted=results$prediction * abs(diff(range(mydata$Absenteeism.time.in.hours))) + min(mydata$Absenteeism.time.in.hours)
actual=results$actual * abs(diff(range(mydata$Absenteeism.time.in.hours))) + min(mydata$Absenteeism.time.in.hours)
comparison=data.frame(predicted,actual)
deviation=((actual-predicted)/actual)
comparison=data.frame(predicted,actual,deviation)
accuracy=1-abs(mean(deviation))
accuracy

##############################################################################

samplesize = 0.80 * nrow(data)
data <- as.data.frame(sapply(data,as.numeric))
set.seed(80)
index = sample( seq_len ( nrow ( data ) ), size = samplesize )

datatrain = data[ index, ]
datatest = data[ -index, ]

max = apply(data , 2 , max)
min = apply(data, 2 , min)
scaled = as.data.frame(scale(data, center = min, scale = max - min))
library(neuralnet)
trainNN = scaled[index , ]
testNN = scaled[-index , ]
set.seed(123)
NN = neuralnet(Absenteeism.time.in.hours ~ ., trainNN, hidden = 3 , linear.output = TRUE )
plot(NN)

predict_testNN = compute(NN, testNN[,c(1:19)])
predict_testNN = (predict_testNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)
predict_trainNN = compute(NN, trainNN[,c(1:19)])
predict_trainNN = (predict_trainNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)

plot(datatest$Absenteeism.time.in.hours, predict_testNN, col='blue', pch=16, ylab = "predicted Absenteeism.time.in.hours", xlab = "real Absenteeism.time.in.hours")
abline(0,1)
RMSE.NN = (sum((datatest$Absenteeism.time.in.hour - predict_testNN)^2) / nrow(datatest)) ^ 0.5
RMSE.NN
RMSE.train = (sum((datatrain$Absenteeism.time.in.hour - predict_trainNN)^2) / nrow(datatrain)) ^ 0.5
RMSE.train

##################################################################################

## test ##

samplesize = 0.80 * nrow(data)
data <- as.data.frame(sapply(data,as.numeric))
set.seed(80)
index = sample( seq_len ( nrow ( data ) ), size = samplesize )

datatrain = data[ index, ]
datatest = data[ -index, ]

max = apply(data , 2 , max)
min = apply(data, 2 , min)
scaled = as.data.frame(scale(data, center = min, scale = max - min))
library(neuralnet)
trainNN = scaled[index , ]
testNN = scaled[-index , ]
set.seed(123)

nodes <- c(1,2,3,4,5)

# No Feature Selection #

#Hidden Layer = 1
for (i in nodes) {
  NN = neuralnet(Absenteeism.time.in.hours ~ ., trainNN, hidden = i , linear.output = TRUE )
  predict_testNN = compute(NN, testNN[,c(1:19)])
  predict_testNN = (predict_testNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)
  predict_trainNN = compute(NN, trainNN[,c(1:19)])
  predict_trainNN = (predict_trainNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)
  
  abline(0,1)
  RMSE.NN = (sum((datatest$Absenteeism.time.in.hour - predict_testNN)^2) / nrow(datatest)) ^ 0.5
  RMSE.NN
  RMSE.train = (sum((datatrain$Absenteeism.time.in.hour - predict_trainNN)^2) / nrow(datatrain)) ^ 0.5
  RMSE.train
  print(paste("Hidden Layers = 1","node =", i,"train:",RMSE.train,"test:",RMSE.NN,sep = " ", collapse=NULL))
}
#Hidden Layer = 2
for (i in nodes) {
  for (o in nodes) {
    NN = neuralnet(Absenteeism.time.in.hours ~ ., trainNN, hidden = c(i,o) , linear.output = TRUE )
    predict_testNN = compute(NN, testNN[,c(1:19)])
    predict_testNN = (predict_testNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)
    predict_trainNN = compute(NN, trainNN[,c(1:19)])
    predict_trainNN = (predict_trainNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)
    
    abline(0,1)
    RMSE.NN = (sum((datatest$Absenteeism.time.in.hour - predict_testNN)^2) / nrow(datatest)) ^ 0.5
    RMSE.NN
    RMSE.train = (sum((datatrain$Absenteeism.time.in.hour - predict_trainNN)^2) / nrow(datatrain)) ^ 0.5
    RMSE.train
    print(paste("Hidden Layers = 2","node =", i+o,"config:",i,",",o,"train:",RMSE.train,"test:",RMSE.NN,sep = " ", collapse=NULL))
  }
}
#Hidden Layer = 3
for (i in nodes) {
  for (o in nodes) {
    for (r in nodes) {
      NN = neuralnet(Absenteeism.time.in.hours ~ ., trainNN, hidden = c(i,o,r) , linear.output = TRUE )
      predict_testNN = compute(NN, testNN[,c(1:19)])
      predict_testNN = (predict_testNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)
      predict_trainNN = compute(NN, trainNN[,c(1:19)])
      predict_trainNN = (predict_trainNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)
      
      abline(0,1)
      RMSE.NN = (sum((datatest$Absenteeism.time.in.hour - predict_testNN)^2) / nrow(datatest)) ^ 0.5
      RMSE.NN
      RMSE.train = (sum((datatrain$Absenteeism.time.in.hour - predict_trainNN)^2) / nrow(datatrain)) ^ 0.5
      RMSE.train
      print(paste("Hidden Layers = 3","node =", i+o+r,"config:",i,",",o,",",r,"train:",RMSE.train,"test:",RMSE.NN,sep = " ", collapse=NULL))
    }
  }
}

# Feature Selection using Hypothesis Testing #
myvars <- c(1,2,3,5,6,7,8,9,10,11,12,14,15,17,18,19,20)

# Feature Selection using Information Gain #
myvars <- c(2,6,12,20)

# Feature Selection using Stepwise #
myvars <- c(2,4,5,12,14,15,19,20)

datatrain2 <- datatrain[,myvars]
datatest2 <-datatest[,myvars]
trainNN2 = scaled[index , ]
testNN2 = scaled[-index , ]



#Hidden Layer = 1
for (i in nodes) {
  NN = neuralnet(Absenteeism.time.in.hours ~ ., trainNN2, hidden = i , linear.output = TRUE )
  predict_testNN = compute(NN, testNN2[,c(1:19)])
  predict_testNN = (predict_testNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)
  predict_trainNN = compute(NN, trainNN2[,c(1:19)])
  predict_trainNN = (predict_trainNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)
  
  RMSE.NN = (sum((datatest2$Absenteeism.time.in.hour - predict_testNN)^2) / nrow(datatest)) ^ 0.5
  RMSE.NN
  RMSE.train = (sum((datatrain2$Absenteeism.time.in.hour - predict_trainNN)^2) / nrow(datatrain)) ^ 0.5
  RMSE.train
  print(paste("Hidden Layers = 1","node =", i,"train:",RMSE.train,"test:",RMSE.NN,sep = " ", collapse=NULL))
}
#Hidden Layer = 2
for (i in nodes) {
  for (o in nodes) {
    NN = neuralnet(Absenteeism.time.in.hours ~ ., trainNN2, hidden = c(i,o) , linear.output = TRUE )
    predict_testNN = compute(NN, testNN2[,c(1:19)])
    predict_testNN = (predict_testNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)
    predict_trainNN = compute(NN, trainNN2[,c(1:19)])
    predict_trainNN = (predict_trainNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)
    
    abline(0,1)
    RMSE.NN = (sum((datatest2$Absenteeism.time.in.hour - predict_testNN)^2) / nrow(datatest)) ^ 0.5
    RMSE.NN
    RMSE.train = (sum((datatrain2$Absenteeism.time.in.hour - predict_trainNN)^2) / nrow(datatrain)) ^ 0.5
    RMSE.train
    print(paste("Hidden Layers = 2","node =", i+o,"config:",i,",",o,"train:",RMSE.train,"test:",RMSE.NN,sep = " ", collapse=NULL))
  }
}
#Hidden Layer 3
for (i in nodes) {
  for (o in nodes) {
    for (r in nodes) {
      NN = neuralnet(Absenteeism.time.in.hours ~ ., trainNN2, hidden = c(i,o,r) , linear.output = TRUE )
      predict_testNN = compute(NN, testNN2[,c(1:19)])
      predict_testNN = (predict_testNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)
      predict_trainNN = compute(NN, trainNN2[,c(1:19)])
      predict_trainNN = (predict_trainNN$net.result * (max(data$Absenteeism.time.in.hours) - min(data$Absenteeism.time.in.hours))) + min(data$Absenteeism.time.in.hours)
      
      RMSE.NN = (sum((datatest2$Absenteeism.time.in.hour - predict_testNN)^2) / nrow(datatest)) ^ 0.5
      RMSE.NN
      RMSE.train = (sum((datatrain2$Absenteeism.time.in.hour - predict_trainNN)^2) / nrow(datatrain)) ^ 0.5
      RMSE.train
      print(paste("Hidden Layers = 3","node =", i+o+r,"config:",i,",",o,",",r,"train:",RMSE.train,"test:",RMSE.NN,sep = " ", collapse=NULL))
    }
  }
}
