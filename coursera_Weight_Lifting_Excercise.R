library(caret)
library(dplyr)
library(corrplot)
library(devtools)
library(rpart)
#library(rattle)
#library(rpart.plot)
library(ggplot2)
#library(gam)
#library(genefilter)
set.seed(1)
rm(list=ls())


url_training<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_testing<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if(!file.exists("pml-training.csv")){
  download.file(url_training, destfile = paste0(getwd(), '/pml-training.csv'))
  }
training<-read.csv("pml-training.csv")


if(!file.exists("pml-testing.csv")){
  download.file(url_testing, destfile = paste0(getwd(), '/pml-testing.csv'))
  }
testing<-read.csv("pml-testing.csv")

#Creat data partition
training_index <- createDataPartition(training$classe, p=0.7, list=FALSE)

training_data <- training %>% slice(training_index)

test_data <- training %>% slice(-training_index)

dim(training_data)
dim(test_data)

#First I remove the first 5 identification variables
training_data <- training_data[, -(1:5)]
dim(training_data)

# Remove variables with low variance
NrZrVr <- nearZeroVar(training_data)
training_data <- training_data[, -NrZrVr]
dim(training_data)

# Remove variables with large number of missing values
missing_ratio<-sapply(names(training_data), function(var) mean(is.na(training_data[ ,var])))
range(missing_ratio)
unique(missing_ratio)
missing<-missing_ratio>0.97
sum(missing)

training_data <- training_data[, !missing]
dim(training_data)

CorrMatrix<-cor(training_data[ , 1:(ncol(training_data)-1)])
range(CorrMatrix)

corr_cutoff<-0.98
high_corr<-findCorrelation(CorrMatrix, cutoff = corr_cutoff, names = FALSE)
training_data<-training_data[ , -high_corr]
dim(training_data)

CorrMatrix <- cor(training_data[, -ncol(training_data)])
sum(abs(CorrMatrix)>corr_cutoff & CorrMatrix!=1)
range(CorrMatrix)
corrplot(CorrMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
test_data <- test_data[ , names(training_data)]


#Classification Tree Model
fit_DecTree <-train(classe ~ ., method = "rpart",
              tuneGrid=data.frame(cp=seq(0, 0.05, len=25)),
              data = training_data)
predict_DecTree <- predict(fit_DecTree, newdata=test_data)
Acc_DecTree <- confusionMatrix(predict_DecTree, test_data$classe)$overall[[1]]
Acc_DecTree

#Random forest
fit_RandFor <-train(classe ~ .,
              method="rf",
              data=training_data, 
              trControl=trainControl(method="cv", number=3, verboseIter=FALSE))
predict_RandFor <- predict(fit_RandFor, newdata=test_data)
Acc_RandFor <- confusionMatrix(predict_RandFor, test_data$classe)$overall[[1]]
Acc_RandFor


#Generalized Boosted Model
fit_GBM  <- train(classe ~ ., data=training_data,
              method = "gbm",
              trControl = trainControl(method = "repeatedcv", number = 5, repeats = 1),
              verbose = FALSE)
predict_GBM <- predict(fit_GBM, newdata=test_data)
Acc_GBM <- confusionMatrix(predict_GBM, test_data$classe)$overall[[1]]
Acc_GBM


#Ensembling
predDF<-data.frame(pred1=predict_DecTree, pred2=predict_RandFor, pred3=predict_GBM, classe=test_data$classe)
CombModFit<-train(classe ~ ., method="ctree", data=predDF)
predictEns<-predict(CombModFit, predDF)
Acc_Ens <- confusionMatrix(predictEns, predDF$classe)$overall[[1]]
Acc_Ens

#Predicted outcomes using the prefered model Random Forest
