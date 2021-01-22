rm(list=ls(all=T))
setwd("C:/Users/HP/.jupyter")
#LOADING LIBRARIES
x = c("ggplot2","corrgram","DMwR","caret","randomForest","unbalanced","c50","dummies","e1071","MASS",
      "rpart","gbm","ROSE","mlbench","mlr","caTools","rBayesianOptimization","pROC"
      ,"pdp","Matrix")
#INSTALL PACKAGES
lapply(x, require, character.only = TRUE)
rm(x)
train_df <- read.csv("train.csv", header = T, na.strings = c(" ", "", "NA"))
head(train_df)

#Dimension of the train data
dim(train_df)

#summary of the dataset
str(train_df)

#Typecasting the target variable(convert to Target Variable)
train_df$target<-as.factor(train_df$target)
#Target classes count in the train data
require(gridExtra)
table(train_df$target)
#percentage count of target classes
table(train_df$target)/length(train_df$target)*100
#barplot
table(train_df$target)
barplot(table(train_df$target),ylab = 'frequency', main = 'barplot taget variable')
#Distribution of the train attributes from 3 to 10
for (var in names(train_df)[c(3:10)]) {
  target<-train_df$target
  plot<-ggplot(train_df,aes(x=train_df[[var]], fill= target))+
    geom_density(kernel='gaussian')+ggtitle(var)+theme_classic()
  print(plot)
}
#importing test data
test_df <- read.csv("test.csv", header = T, na.strings = c(" ", "", "NA"))
head(test_df)
#dimension of test data
dim(test_df)
#distribution of mean values per row and column in train and test dataset
train_mean<-apply(train_df[-c(1,2)], MARGIN = 1, FUN = mean)
test_mean<-apply(test_df[,-c(1)], MARGIN = 1, FUN = mean)
ggplot()+
  geom_density(data = train_df[,-c(1,2)],aes(x=train_mean),kernel='gaussian',
               show.legend = TRUE, color='blue')+theme_classic()+
  geom_density(data=test_df[,-c(1)], aes(x=test_mean),kernel='gaussian',show.legend = TRUE,
               color='green')+
  labs(x='mean values per row', title="Distribution of mean values per row in train and test dataset")
#distribution of standard daviation values per row and column in train and test data
train_sd<-apply(train_df[,-c(1,2)],MARGIN = 1,FUN = sd)
test_sd<-apply(test_df[,-c(1)],MARGIN=1, FUN=sd)
ggplot()+
  geom_density(data=train_df[,-c(1,2)],aes(x=train_sd),kernel='gaussian',show.legend = TRUE,
               color='red')+theme_classic()+
  geom_density(data = test_df[,-c(1)], aes(x=test_sd), kernel='gaussian', show.legend = TRUE,color='blue')+
  labs(x='sd values per row', title="Distribution of Standard Deviation")
#applying the functions to find the sd values per column in train and test dataset
train_sd<- apply(train_df[,-c(1,2)], MARGIN = 2, FUN = sd)
test_sd<- apply(test_df[,-c(1)], MARGIN = 2, FUN = sd)
ggplot()+
  geom_density(aes(x=train_sd), kernel='gaussian', show.legend = TRUE, color='red')+theme_classic()+
  geom_density(aes(x=test_sd), kernel='gaussian', show.legend = TRUE, color='blue')+
  labs(x='sd values per column', title = "Distribution of standard daviation per column")
#Missing Value Analysis

missing_val<-data.frame(missing_val=apply(train_df,2,function(x){sum(is.na(x))}))
missing_val<-sum(missing_val)
missing_val

#finding the missing value in test data

missing_val<-data.frame(missing_val=apply(test_df,2,function(x){sum(is.na(x))}))
missing_val<-sum(missing_val)
missing_val

#Correlation in train data

train_df$target<-as.numeric(train_df$target)
train_correlation<-cor(train_df[,c(2:202)])
train_correlation

#Correlation in test data

test_correlation<-cor(test_df[,c(3:201)])
test_correlation

#split the training data using random sampling

train_index<-sample(1:nrow(train_df),0.75*nrow(train_df))
train_data<-train_df[-train_index,]

#validation data

valid_data<-train_df[-train_index]

#dimension of the train and validation data

dim(train_data)
dim(valid_data)

#Random Forest Classifier

set.seed(2732)
train_data$target<-as.factor(train_data$target)
mtry<-floor(sqrt(200))
tunegrid<-expand.grid(.mtry=mtry)

#fitting the random forest

rf<-randomForest(target~.,train_data[,-c(1)], mtry=mtry, ntree=10, importance=TRUE)

#feature importance by random Forest

VarImp<-importance(rf, type=2)
VarImp

#LOGISTIC REGRESSION MODEL
#training and validation dataset

x_t<-as.matrix(train_data[,-c(1,2)])
y_t<-as.matrix(train_data$target)
x_v<-as.matrix(valid_data[,-c(1,2)])
y_v<-as.matrix(valid_data$target)
test<-as.matrix(test_df[,-c(1)])

#logistic regression model

set.seed(667)
lr_model<-glmnet(X_t,Y_t, Family="binomial")
summary(lr_model)

#cross validation prediction

set.seed(8909)
cv_lr<-cv.glmnet(X_t,y_t, family="binomial", type.measure="class")
cv_lr

#model performance on validation dataset

set.seed(5363)
cv_predict.lr<-predict(cv_lr, X_v,s="lambda.min", type="class")
cv_predict.lr

#confusion matrix

set.seed(689)
target<-valid.data$target
target<-as.factor(target)
cv_predict.lr<-as.factor(cv_predict.lr)
confusionMatrix(data=cv_predict.lr, reference=target)

#Random OverSampling Examples(ROSE)

set.seed(699)
train.rose<-ROSE(target~., data=train.data[,-c(1)], seed=32)$data
table(train.rose$target)
valid.rose<-ROSE(target~., data=valid_data[,-c(1)], seed=42)$data
table(valid.rose$target)
