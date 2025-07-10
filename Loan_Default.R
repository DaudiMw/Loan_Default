# installed the package to access MySql from R

# Loading the package
library(RMySQL)

# importing library for regression
library(tidyverse)

# importing library for useful list arithmetic
library(purrr)

# loading packages for Naive Bayes Prediction
library(caret) #Short for "Classification and Regression Training" used for Bayes
library(caretEnsemble) #makes ensembles of caret models
library(psych) #used by social scientists
library(Amelia) #used to replace missing data
library(mice) #used to replace missing data
library(GGally) #extension for ggplot2
library(rpart) #used to split the data recursively
library(randomForest) #random forest algorithm forclassification and regression
library(klaR) #Evaluates the performance of a classification method
#Functions for Naive Bayes (provided by R)
library(e1071)

# Connect to MySQL
connect <- dbConnect(MySQL(),
                     user = 'root',
                     host = 'localhost',
                     dbname = 'proj3')

# Check the description of the connection
summary(connect)
dbListTables(connect)

# Import the data into R
data <- dbReadTable(connect, "lending")

# Doing statistical Analysis of loan_amnt, annual_inc, month_since_first_credit

data_list <- list(
  loan_amount = data$loan_amnt,
  annual_income = data$adjusted_annual_inc,
  months_since_firstcred = data$months_since_first_credit
)

# Min
min <- sapply(data_list, min)

# Max
max <- sapply(data_list, max)

# Mean
mean <- sapply(data_list, mean)
mean <- sapply(mean, round, digits = 2) # Round it to make it look a little more readable.

# Median
median <- sapply(data_list, median)

# 25th percentile
percentile_25th <- sapply(data_list, quantile, probs = 0.25)

# 75th percentile
percentile_75th <- sapply(data_list, quantile, probs = 0.75)

# Standard deviation
std <- sapply(data_list, sd)
std <- sapply(std, round, digits = 2) # Round it.

# Variance
var <- sapply(data_list, var)
var <- sapply(var, round, digits = 2) # Round it.

# Analysis report in the form of a dataframe
analysis <- data.frame(min, max, mean, median, percentile_25th, percentile_75th, std, var)


# Check the validity of my analysis using r function describe().
print("Validity check using describe()")
describe(data)

# Write my analysis to my database.
dbWriteTable(connect, "Statistical_Analysis", analysis, overwrite=TRUE)

# Multiple Regression (By hand) using the same columns but this time the 
# variable that we want to predict is loan amount because I want
# to simulate a scenario where we have data on customers with different loan lengths and income
# and we want to figure how much money we should lend them based on prior loans

# y <- data$loan_amnt
# x1 <- data$adjusted_annual_inc
# x2 <- data$months_since_first_credit
# 
# x1y <- y * x1
# x2y <- y * x2
# x1x2 <- x1 * x2
# n <- length(y)
# 
# sum_x1 <- as.numeric(sum(x1))
# sum_x2 <- as.numeric(sum(x2))
# sum_y <- as.numeric(sum(y))
# sum_X1Y <- as.numeric(sum(x1y))
# sum_X2Y <- as.numeric(sum(x2y))
# sum_X1X2 <- as.numeric(sum(x1x2))

# Some values needed to be wrapped in the as.numeric() function because the integer was too big initially and there were warnings of NA values.
# sum_x1_squared <- as.numeric(sum_x1) * as.numeric(sum_x1)
# sum_x2_squared <- as.numeric(sum_x2) * as.numeric(sum_x2)
# sum_y_squared <- as.numeric(sum_y) * as.numeric(sum_y)
# 
# x1_resid <- sum(x1 - mean(x1))
# x2_resid <- sum(x2 - mean(x2))
# y_resid <- sum(y - mean(y))
# sum_squared_x1 <- sqrt((x1_resid * x1_resid)/ (n-1)) - (sum_x1_squared / n)
# sum_squared_x2 <- sqrt((x2_resid * x2_resid)/ (n-1)) - (sum_x2_squared / n)
# sum_squared_y <- sqrt((y_resid * y_resid)/ (n-1)) - (sum_y_squared / n)
# 
# sum_x1y <- sum_X1Y - ((as.numeric(sum_x1) * as.numeric(sum_y)) / n)
# sum_x2y <- sum_X2Y - ((as.numeric(sum_x2) * as.numeric(sum_y)) / n)
# sum_x1x2 <- sum_X1X2 - ((as.numeric(sum_x1) * as.numeric(sum_x2)) / n)
# 
# b1_numer <- (sum_squared_x2 * sum_x1y) - (sum_x1x2 * sum_x2y)
# b1_denom <- (sum_squared_x1 * sum_squared_x2) - (sum_x1x2 * sum_x1x2)
# b1 <- b1_numer / b1_denom
# 
# b2_numer <- (sum_squared_x1 * sum_x2y) - (sum_x1x2 * sum_x1y)
# b2_denom <- (sum_squared_x1 * sum_squared_x2) - (sum_x1x2 * sum_x1x2)
# b2 <- b2_numer / b2_denom
# 
# cat("a =", mean(y) - b1*mean(x1) - b2*mean(x2))

# Using lm function for easy readability and to double check
model <- lm(loan_amnt ~ adjusted_annual_inc + months_since_first_credit, data = data)
summary(model)
coef(model)

# For the naive bayes prediction we will choose our predictor variable as the loan_default
# so we are trying to be able to predict whether or not a customer will not pay their loan
# after 90 days based on the same factors used in our multiple regression model
# which is loan_amount, annual_income, and months_since_first_cred.

# Change the column from binary to True and False
data$Outcome <- factor(data$loan_default, levels=c(0,1), labels = c("False", "True"))
dataStripped <- data[c("loan_amnt", "adjusted_annual_inc", "months_since_first_credit", "Outcome")]

# Plotting the relationships between all features
ggpairs(dataStripped)

# The upcoming plots are similar but I changed around x and y limits and scale to make more readable

# Plot showing the relationship between a persons adjusted annual income and the loan given to them by the bank.
ggplot(dataStripped, aes(loan_amnt, adjusted_annual_inc)) + geom_point() + geom_smooth(method="lm") + scale_y_continuous(trans='log2') + ggtitle("Annual Income vs Loan Amount") + xlab("Loan Amount (USD)") + ylab("Adjusted Annual Income (USD)")

# Plot showing the relationship between a persons months since first credit and the loan given to them by the bank.
ggplot(dataStripped, aes(loan_amnt, months_since_first_credit)) + geom_point() + geom_smooth(method="lm") + ggtitle("Months since credit established vs Loan Amount") + xlab("Loan Amount (USD)") + ylab("Months")

# Plot showing the relationship between a persons income and whether or not they defaulted on their loan.
ggplot(dataStripped, aes(adjusted_annual_inc, Outcome)) + geom_boxplot() + xlim(-1000, 100000) + ggtitle("Loan Defaults vs. Adjusted Annual Income") + ylab("Loan Defaulted") + xlab("Adjusted Annual Income (USD)")

# Plot showing the relationship between a persons months since credit was established and whether or not they defaulted on their loan.
ggplot(dataStripped, aes(months_since_first_credit, Outcome)) + geom_boxplot() + xlim(0, 400) + ggtitle("Loan Defaults vs. Months since Credit established") + ylab("Loan Defaulted") + xlab("Months")


# Splitting data into training and testing sets.
indxTrain <- createDataPartition(y = dataStripped$Outcome,p = 0.75,list = FALSE)
training <- dataStripped[indxTrain,]
testing <- dataStripped[-indxTrain,] #Check dimensions of the split >
prop.table(table(dataStripped$Outcome)) * 100

# Seperating the data and the labels
x = training[,-4]
y = training$Outcome

# Sets up a grid of tuning parameters for a number of classification and regression routines
# From Carat
model = train(x,y,'nb',trControl=trainControl(method='cv',number=10))

Predict <- predict(model,newdata = testing ) #Get the confusion matrix to see accuracy value and other parameter values

confusionMatrix(Predict, testing$Outcome)
#Confusion Matrix and Statistics
#Plot Variable performance
#Tracks the changes in model statistics from Carat
X <- varImp(model)
plot(X)