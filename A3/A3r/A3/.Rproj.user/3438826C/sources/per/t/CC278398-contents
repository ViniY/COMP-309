library(foreach)
library(iterators)
library(itertools)
library(ggplot2)
library(gridExtra)
library(corrplot)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(cowplot)
library(missForest)
library(dplyr)
data<-read.csv("correct.csv",header = T)
str(data)
convert<-c(2,3,4,5,6,7,8,10,11,12,13,14,15,16,18,19,20,21,22,23,24,27,28,30,31,32,33,34,35,36,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,58,59)
data[,convert]<-data.frame(apply(data[convert],2,as.factor))
str(data)
View(data)
data1.imp<-missForest(data)
data.imp_saver<-data1.imp$ximp
write.csv(data.imp_saver,"finalcorrect.csv")
data1.imp$OOBerror
fixed<-read.csv("finalcorrect.csv")

train<-read.csv("lalalala.csv",header = T)
View(train)
names(train)[1]<-"ID"
View(train)
test<-read.csv("testfinal.csv",header = T)
test[test=="?"]<-NA
names(test)[1]<-"ID"
View(test)
convert<-c(2,3,4,5,6,7,8,10,11,12,13,14,15,16,18,19,20,21,22,23,24,27,28,30,31,32,33,34,35,36,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,58,59)
train[,convert]<-data.frame(apply(train[convert],2,as.factor))
str(train)
convert<-c(2,3,4,5,6,7,8,10,11,12,13,14,15,16,18,19,20,21,22,23,24,27,28,30,31,32,33,34,35,36,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,58,59)
test[,convert]<-data.frame(apply(test[convert],2,as.factor))
set.seed(111)
rf <-randomForest(total_income_hhld_code~.,data=train)
print(rf)
attributes(rf)
library(caret)
p1 <-predict(rf,train)
head(p1)
library(e1071)
confusionMatrix(p1,train$total_income_hhld_code)
f<-factor(c(1,2,3,4,5,6,7,8,9,10,11,12,13,14),exclude = NULL)
length(levels(f))
levels(f)
train1<-train
train1$total_income_hhld_code<-factor(as.character(c(1:14,NA)),exclude = NULL)
str(train1)
test1<-test
View(test)
str(test1)
#test1$total_income_hhld_code<-factor(c(1:14,NA),exclude = NULL)
p2<-predict(rf,test)
levels(test1$total_income_hhld_code)<-c(levels(test1$total_income_hhld_code),"1","2","3","4","5","6","7","8","9","10","11","12","13","14",NA)
library(VIM)
dataknn<-read.csv("knnneedfix.csv")


str(knnneedfix1)
fixing<-knnneedfix1
str(fixing)

library(Hmisc)
library(DMwR)
library(VIM)
da00<-read.csv("knnneedfix.csv",header=T)
da00[da00=="?"]<-NA

View(da00)
KnnOut<-knnImputation(da00,k=10,scal=T,meth="weighAvg" )
impute()
View(KnnOut)
write.csv(knnOut,"knnfixed.csv")


data100<-read.csv("fixed.csv")
dataxxx<-data100[data100$m_age<bench]
dataxx<-data100(data100$m_age)
