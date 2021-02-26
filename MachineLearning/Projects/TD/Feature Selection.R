
library(leaps) # this contains the function regsubsets - This only works for Linear Transformations
library(MASS)
library(ISLR)
library(caret)
library(tidyverse)
library(PerformanceAnalytics)
library(dplyr)
library(splines)
library(gam)
library(nlcor)
library(randomForest)

# load in the data

df <- read.csv("Desktop/Extra-curricular/TD/Data Science Project - Excel.csv")
attach(df)

# Fit the linear regression to all predictors
# remove certain columns 
linearModel <- lm(rental ~ ., data=df)
# Summarize the results
summary(linearModel) # We can see that some are significant, in regards to other variables which are present.
# Thus, do best subset selection to get better predictors

# note: R has automatically turned qualitative predictors into dummies
# this inly works if originally predictors are coded as strings or factors
#-----------------------------------------------------------------------
# now look at different ways of selecting models
# first: best subset selection
# Non-linearity
regfit_best = regsubsets(rental ~.,data = df, really.big = T, nvmax = 5)
# performs exhaustive search
x<- summary(regfit_best)

# note: by default only search up to 8 predictors
regfit_best = regsubsets(Balance~.,data = Credit,nvmax = 11)
summary(regfit_best)

# to extract coefficients from one particular model, 
# for example model with 4 predictors
coef(regfit.best,4)

# =======================================================================
# look at forward stepwise
regfit.forward = regsubsets(data$y~.,data = data,method='forward', nvmax = 50)
# performs forward selection
summary(regfit.forward)

# note: for example for k = 4 we get a different model compared
# to the result from best subset selection.

coef(regfit.forward,50)

# look at forward stepwise
regfit.forward = regsubsets(data$rental ~., data = df,method='forward', nvmax = 50)
# performs forward selection
summary(regfit.forward)

# note: for example for k = 4 we get a different model compared
# to the result from best subset selection.

coef(regfit.forward,10)

library(leaps) # this contains the function regsubsets - This only works for Linear Transformations
library(MASS)
library(ISLR)
library(caret)
library(tidyverse)
library(PerformanceAnalytics)
library(dplyr)
library(splines)
library(gam)
library(nlcor)
library(randomForest)
