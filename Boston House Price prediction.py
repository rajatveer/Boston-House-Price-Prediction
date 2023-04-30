#!/usr/bin/env python
# coding: utf-8

# # **Regression Project: Boston House Price Prediction**
# 
# # **Marks: 60**

# Welcome to the project on regression. We will use the **Boston house price dataset** for this project.
# 
# -------------------------------
# ## **Objective**
# -------------------------------
# 
# The problem at hand is to **predict the housing prices of a town or a suburb based on the features of the locality provided to us**. In the process, we need to **identify the most important features affecting the price of the house**. We need to employ techniques of data preprocessing and build a linear regression model that predicts the prices for the unseen data.
# 
# ----------------------------
# ## **Dataset**
# ---------------------------
# 
# Each record in the database describes a house in Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. Detailed attribute information can be found below:
# 
# Attribute Information:
# 
# - **CRIM:** Per capita crime rate by town
# - **ZN:** Proportion of residential land zoned for lots over 25,000 sq.ft.
# - **INDUS:** Proportion of non-retail business acres per town
# - **CHAS:** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# - **NOX:** Nitric Oxide concentration (parts per 10 million)
# - **RM:** The average number of rooms per dwelling
# - **AGE:** Proportion of owner-occupied units built before 1940
# - **DIS:** Weighted distances to five Boston employment centers
# - **RAD:** Index of accessibility to radial highways
# - **TAX:** Full-value property-tax rate per 10,000 dollars
# - **PTRATIO:** Pupil-teacher ratio by town
# - **LSTAT:** % lower status of the population
# - **MEDV:** Median value of owner-occupied homes in 1000 dollars

# ## **Importing the necessary libraries and overview of the dataset**

# In[55]:


# Import libraries for data manipulation
import pandas as pd

import numpy as np

# Import libraries for data visualization
import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.graphics.gofplots import ProbPlot

# Import libraries for building linear regression model
from statsmodels.formula.api import ols

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

# Import library for preparing data
from sklearn.model_selection import train_test_split

# Import library for data preprocessing
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")


# ### **Loading the data**

# In[56]:


df = pd.read_csv("Boston.csv")

df.head()


# **Observation:**
# 
# * The price of the house indicated by the variable MEDV is the target variable and the rest of the variables are independent variables based on which we will predict the house price (MEDV).

# ### **Checking the info of the data**

# In[57]:


df.info()


# **Observations:**
# 
# - There are a total of **506 non-null observations in each of the columns**. This indicates that there are **no missing values** in the data.
# - There are **13 columns** in the dataset and **every column is of numeric data type**.

# ## **Exploratory Data Analysis and Data Preprocessing**

# ### **Summary Statistics of this Dataset**

# ### **Question 1:** Write the code to find the summary statistics and write your observations based on that. (4 Marks)

# In[58]:


# Write your code here
df.describe(include='all').T


# **Observations:
# 1) Per capita crime rate by town is 3.6, as 75% values lies within 0 and 3.67. Maximum value is 88.97, so can say that some town has much more crime rate than most of other towns.
# 2) 75 precentile of proportion of residential land zoned for lots over 25000 sq ft is within 12.5 
# 3) Most of the land is not in contact with the Charles river and so concentration of Nitric oxid is higher with the mean of 0.55
# 4) Most of the houses build around 1940 since mean age of the house is 68 with 2.9 years as minimum and 100 years as maximum age of house. 
# 5) Tax is higher for some areas as they are near to radial highways and has weighted distance to employment centers is within 1.12 and 5.18
# 6) As teacher pupil ration by town is 18.45 in the areas where % lower status os population lies and median house value is near 22 thousand dollers which are owner occupied.  

# ### **Univariate Analysis**

# **Let's check the distribution of the variables**

# ### **Question 2:** Write your observations based on the below univariate plots. (4 Marks)

# In[59]:


# Plotting all the columns to look at their distributions
for i in df.columns:
    
    plt.figure(figsize = (7, 4))
    
    sns.histplot(data = df, x = i, kde = True)
    
    plt.show()


# **Observations: 
# 1) Count vs CRIM, ZN, DIS and LSTAT are right skewed, meaning that most of the observations start at one perticular number, so the mean and median has different values.
# 2) Count vs Age and Ptration are left skewed meaning that most of people owned house are older and the pupil teacher ratio is higher where houses owned by older people.
# 3) Average number of rooms per dwelling and medican value of owner occupied house seems normallt distributed meaning that mean and median almost are same. There are some outliers for median value.
# 4) We cannot take charles river contact in our model as it has only two values 1 and 0, so that would be biased.
# 5) Columns INDUS, NOX, RAD and TAX are influenced by outliers, so we can say that they are senstive.

# As the dependent variable is sightly skewed, we will apply a **log transformation on the 'MEDV' column** and check the distribution of the transformed column.

# In[60]:


df['MEDV_log'] = np.log(df['MEDV'])


# In[61]:


sns.histplot(data = df, x = 'MEDV_log', kde = True)


# **Observation:**
# 
# - The log-transformed variable (**MEDV_log**) appears to have a **nearly normal distribution without skew**, and hence we can proceed.

# Before creating the linear regression model, it is important to check the bivariate relationship between the variables. Let's check the same using the heatmap and scatterplot.

# ### **Bivariate Analysis**

# **Let's check the correlation using the heatmap**

# ### **Question 3:** Write the code to plot the correlation heatmap and write your observations based on that. (6 Marks)

# In[62]:


plt.figure(figsize = (12, 10))

cmap = sns.diverging_palette(230, 20, as_cmap = True)

sns.heatmap(df.corr(), annot = True, fmt = '.2f', cmap = cmap)

plt.show()


# **Observations:
# 1) RM (average number of rooms per dwelling) is highly correlated with our target variable MEDV (median value of house occupied by owner). PTRATIO (pupil-teacher rator by town) is moderately correlated with the MEDV. 
# 2) LSTAT (% lower staus of the population) is highly negatively correlated with our target variable MEDV (median value of house occupied by owner).
# 3) CRIM (Crime rate per capita) is  correlated with RAD and TAX followed by NOX, LSTAT and INDUS.
# 4) INDUS is highly correlated with NOX, TAX followed by AGE and RAD, highly negatively correlated with DIS.
# 5) NOX has high correlation with  AGE, RAD, TAX and LSTAT and highly negative correlation with DIS.
# 6) From Heatmap I would say we would face multicolinearity problem as we have positive and negative relationships between independent variabls. 
# 7) To predict MEDV_log the best feature variable would be RM followed by ZN, LSTAT and DIS. 

# Now, we will visualize the relationship between the pairs of features having significant correlations.

# ### **Visualizing the relationship between the features having significant correlations (>= 0.7 or <= -0.7)**

# In[63]:


# Scatterplot to visualize the relationship between AGE and DIS
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'AGE', y = 'DIS', data = df)

plt.show()


# **Observations:**
# - The distance of the houses to the Boston employment centers appears to decrease moderately as the the proportion of the old houses increase in the town. It is possible that the Boston employment centers are located in the established towns where proportion of owner-occupied units built prior to 1940 is comparatively high.

# In[64]:


# Scatterplot to visulaize the relationship between RAD and TAX
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'RAD', y = 'TAX', data = df)

plt.show()


# **Observations:**
# 
# - The correlation between RAD and TAX is very high. But, no trend is visible between the two variables. 
# - The strong correlation might be due to outliers. 

# Let's check the correlation after removing the outliers.

# In[65]:


# Remove the data corresponding to high tax rate
df1 = df[df['TAX'] < 600]

# Import the required function
from scipy.stats import pearsonr

# Calculate the correlation
print('The correlation between TAX and RAD is', pearsonr(df1['TAX'], df1['RAD'])[0])


# **Observation:**
# 
# - So, the high correlation between TAX and RAD is due to the outliers. The tax rate for some properties might be higher due to some other reason.

# In[66]:


# Scatterplot to visualize the relationship between INDUS and TAX
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'INDUS', y = 'TAX', data = df)

plt.show()


# **Observations:**
# 
# - The tax rate appears to increase with an increase in the proportion of non-retail business acres per town. This might be due to the reason that the variables TAX and INDUS are related with a third variable.

# In[67]:


# Scatterplot to visulaize the relationship between RM and MEDV
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'RM', y = 'MEDV', data = df)

plt.show()


# **Observations:**
# 
# - The price of the house seems to increase as the value of RM increases. This is expected as the price is generally higher for more rooms.
# 
# - There are a few outliers in a horizontal line as the MEDV value seems to be capped at 50.

# In[68]:


# Scatterplot to visulaize the relationship between LSTAT and MEDV
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'LSTAT', y = 'MEDV', data = df)

plt.show()


# **Observations:**
# 
# - The price of the house tends to decrease with an increase in LSTAT. This is also possible as the house price is lower in areas where lower status people live.
# - There are few outliers and the data seems to be capped at 50.

# ### **Question 4** (8 Marks):
# - **Create a scatter plot to visualize the relationship between the remaining features having significant correlations (>= 0.7 or <= -0.7) (4 Marks)**
#     - INDUS and NOX
#     - AGE and NOX
#     - DIS and NOX
# - **Write your observations from the plots (4 Marks)**

# In[69]:


# Scatterplot to visualize the relationship between INDUS and NOX
plt.figure(figsize = (6, 6))

sns.scatterplot(x='INDUS', y='NOX', data=df)

plt.show()


# **Observations:1) As the concentration of Nitric oxide increases proportion of not-retail business acres land increases, which seems logical. 2) As conentatrin of Nitric acid reaches 0.57-0.58, we can see constant non-retail business acres land which might be influenced by other variable.

# In[70]:


# Scatterplot to visualize the relationship between AGE and NOX
plt.figure(figsize = (6, 6))

sns.scatterplot(x='AGE', y='NOX', data=df)

plt.show()


# **Observations:
# 1) Houses built before 1940's seems to biult in high nitric acid concentration zone, so we can say that later as the technology become advanced people may realized that houses needs to built in low concentration zone. 
# 2) Nitric acid concentration is capped around 0.9.
# 3) Older houses build in high nitric concentration zone and newer houses are in low concentration zone.

# In[71]:


# Scatterplot to visualize the relationship between DIS and NOX
plt.figure(figsize = (6, 6))

sns.scatterplot(x='DIS', y='NOX', data=df)

plt.show()


# **Observations:1) Data is capped around 0.9 for concentration for Nitric acid. 2) As the distance from the employemnt centers increases, value for the concentartion of nitric acid decreases. SO we can say that the five boston employment centers are in low nitric acid concentration zones. 3) There might be third variable that affect the relationship between these two variables.

# We have seen that the variables LSTAT and RM have a linear relationship with the dependent variable MEDV. Also, there are significant relationships among few independent variables, which is not desirable for a linear regression model. Let's first split the dataset.

# ### **Split the dataset**
# 
# Let's split the data into the dependent and independent variables and further split it into train and test set in a ratio of 70:30 for train and test sets.

# In[72]:


# Separate the dependent variable and indepedent variables
Y = df['MEDV_log']

X = df.drop(columns = {'MEDV', 'MEDV_log'})

# Add the intercept term
X = sm.add_constant(X)


# In[73]:


# splitting the data in 70:30 ratio of train to test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 1)


# Next, we will check the multicollinearity in the training dataset.

# ### **Check for Multicollinearity**
# 
# We will use the Variance Inflation Factor (VIF), to check if there is multicollinearity in the data.
# 
# Features having a VIF score > 5 will be dropped / treated till all the features have a VIF score < 5

# In[74]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to check VIF
def checking_vif(train):
    vif = pd.DataFrame()# convert to pandas dataframe
    vif["feature"] = train.columns # select what features we want

    # Calculating VIF for each feature
    vif["VIF"] = [
        variance_inflation_factor(train.values, i) for i in range(len(train.columns))
    ]
    return vif


print(checking_vif(X_train))


# **Observations:**
# 
# - There are two variables with a high VIF - RAD and TAX (greater than 5). 
# - Let's remove TAX as it has the highest VIF values and check the multicollinearity again.

# ### **Question 5:** Drop the column 'TAX' from the training data and check if multicollinearity is removed? (2 Marks)

# In[75]:


# Create the model after dropping TAX
X_train = X_train.drop(['TAX'], axis=1)

# Check for VIF
print(checking_vif(X_train))


# Now, we will create the linear regression model as the VIF is less than 5 for all the independent variables, and we can assume that multicollinearity has been removed between the variables.

# ### **Question 6:** Write the code to create the linear regression model and print the model summary. Write your observations from the model. (6 Marks)

# **Hint:** Use the sm.OLS() model on the training data

# In[76]:


# Create the model
model1 = sm.OLS(y_train, X_train).fit()

# Get the model summary
model1.summary()


# **Observations:1) Value of R-squared for our model is 0.769 which looks good. 2) But the p-value for ZN, INDUS and AGE is greater than 0.05, so we can say that these variables are not significant while predecting the MEDV value, so we need to remove them and check for r-squared again. 

# ### **Question 7:** Drop insignificant variables (variables with p-value > 0.05) from the above model and create the regression model again. (4 Marks)

# ### **Examining the significance of the model**
# 
# It is not enough to fit a multiple regression model to the data, it is necessary to check whether all the regression coefficients are significant or not. Significance here means whether the population regression parameters are significantly different from zero. 
# 
# From the above it may be noted that the regression coefficients corresponding to ZN, AGE, and INDUS are not statistically significant at level α = 0.05. In other words, the regression coefficients corresponding to these three are not significantly different from 0 in the population. Hence, we will eliminate the three features and create a new model.

# In[77]:


# Create the model after dropping columns 'MEDV', 'MEDV_log', 'TAX', 'ZN', 'AGE', 'INDUS' from df DataFrame
Y = df['MEDV_log']

X = df.drop(columns=['ZN','INDUS','AGE', 'TAX', 'MEDV', 'MEDV_log'], axis=1) # Write your code here

X = sm.add_constant(X)

# Splitting the data in 70:30 ratio of train to test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30 , random_state = 1)

# Create the model
model2 = sm.OLS(y_train, X_train).fit()

# Get the model summary
model2.summary()


# Now, we will check the linear regression assumptions.

# ### **Checking the below linear regression assumptions**
# 
# 1. **Mean of residuals should be 0**
# 2. **No Heteroscedasticity**
# 3. **Linearity of variables**
# 4. **Normality of error terms**

# ### **Question 8:** Write the code to check the above linear regression assumptions and provide insights. (8 Marks)

# ### **1. Check for mean residuals**

# In[78]:


residuals = model2.resid

np.mean(residuals)


# **Observations:The mean is very close to 0, so the assumption of mean of residuals should be 0 satisfied.

# ### **2. Check for homoscedasticity**

# - Homoscedasticity - If the residuals are symmetrically distributed across the regression line, then the data is said to be homoscedastic.
# 
# - Heteroscedasticity- - If the residuals are not symmetrically distributed across the regression line, then the data is said to be heteroscedastic. In this case, the residuals can form a funnel shape or any other non-symmetrical shape.
# 
# - We'll use `Goldfeldquandt Test` to test the following hypothesis with alpha = 0.05:
# 
#     - Null hypothesis: Residuals are homoscedastic
#     - Alternate hypothesis: Residuals have heteroscedastic

# In[79]:


from statsmodels.stats.diagnostic import het_white

from statsmodels.compat import lzip

import statsmodels.stats.api as sms


# In[80]:


name = ["F statistic", "p-value"]

test = sms.het_goldfeldquandt(y_train, X_train)

lzip(name, test)


# **Observations: As we can see the p-value is greater than 0.05, we fail to reject the null hypothesis and so residuals are homoscedastic.

# ### **3. Linearity of variables**
# 
# It states that the predictor variables must have a linear relation with the dependent variable.
# 
# To test the assumption, we'll plot residuals and the fitted values on a plot and ensure that residuals do not form a strong pattern. They should be randomly and uniformly scattered on the x-axis.

# In[81]:


# Predicted values
fitted = model2.fittedvalues

# sns.set_style("whitegrid")
sns.residplot(x = fitted, y = residuals, color = "lightblue", lowess = True)

plt.xlabel("Fitted Values")

plt.ylabel("Residual")

plt.title("Residual PLOT")

plt.show()


# **Observations: No patteren in residuals and fitted value and so the assumption of linearity of variables is satisfied.

# ### **4. Normality of error terms**
# 
# The residuals should be normally distributed.

# In[82]:


# Plot histogram of residuals

plt.figure(figsize=(6,6))
sns.histplot(residuals, kde=True)
plt.show()


# In[83]:


# Plot q-q plot of residuals
import pylab

import scipy.stats as stats

stats.probplot(residuals, dist = "norm", plot = pylab)

plt.show()


# **Observations:In q-q plot we can see a stratight line so the residuals are normally distirbuted and assumption is satisfied.

# ### **Check the performance of the model on the train and test data set**

# ### **Question 9:** Write your observations by comparing model performance of train and test dataset (4 Marks)

# In[84]:


# RMSE
def rmse(predictions, targets):
    return np.sqrt(((targets - predictions) ** 2).mean())


# MAPE
def mape(predictions, targets):
    return np.mean(np.abs((targets - predictions)) / targets) * 100


# MAE
def mae(predictions, targets):
    return np.mean(np.abs((targets - predictions)))


# Model Performance on test and train data
def model_pref(olsmodel, x_train, x_test):

    # In-sample Prediction
    y_pred_train = olsmodel.predict(x_train)
    y_observed_train = y_train

    # Prediction on test data
    y_pred_test = olsmodel.predict(x_test)
    y_observed_test = y_test

    print(
        pd.DataFrame(
            {
                "Data": ["Train", "Test"],
                "RMSE": [
                    rmse(y_pred_train, y_observed_train),
                    rmse(y_pred_test, y_observed_test),
                ],
                "MAE": [
                    mae(y_pred_train, y_observed_train),
                    mae(y_pred_test, y_observed_test),
                ],
                "MAPE": [
                    mape(y_pred_train, y_observed_train),
                    mape(y_pred_test, y_observed_test),
                ],
            }
        )
    )


# Checking model performance
model_pref(model2, X_train, X_test)  


# **Observations:
# 1) Train and test scoer are very close to each other so we can say that the model is not overfitting. 
# 2) Test score are slightly better than train score so we can increase the model complexity to get little better model.
# 3) So, the model2 is performing the best when compared with model1 because in model2 we are dropping insignificant variables.

# ### **Apply cross validation to improve the model and evaluate it using different evaluation metrics**

# In[85]:


# Import the required function

from sklearn.model_selection import cross_val_score

# Build the regression model and cross-validate
linearregression = LinearRegression()                                    

cv_Score11 = cross_val_score(linearregression, X_train, y_train, cv = 10)
cv_Score12 = cross_val_score(linearregression, X_train, y_train, cv = 10, 
                             scoring = 'neg_mean_squared_error')                                  


print("RSquared: %0.3f (+/- %0.3f)" % (cv_Score11.mean(), cv_Score11.std() * 2))
print("Mean Squared Error: %0.3f (+/- %0.3f)" % (-1*cv_Score12.mean(), cv_Score12.std() * 2))


# ### **Question 10:** Get model Coefficients in a pandas dataframe with column 'Feature' having all the features and column 'Coefs' with all the corresponding Coefs. (4 Marks)
# 
# **Hint:** To get values please use coef.values

# In[86]:


coef = model2.params

pd.DataFrame({'Feature' : coef.index, 'Coefs' : coef.values})


# In[87]:


# Let us write the equation of the fit

Equation = "log (Price) = "

print(Equation, end = '\t')

for i in range(len(coef)):
    print('(', coef[i], ') * ', coef.index[i], '+', end = ' ')


# **Note:** There might be slight variation in the coefficients depending on the library version you are using. There will be no deducting in marks for that as long as your observations are aligned with the output. In case, the coefficients vary too much, please make sure your code is correct.

# ### **Question 11:** Write the conclusions and business recommendations derived from the model. (10 Marks)

# Conclusion: 
# 1) We performed EDA, univariate and bivaiate analysis, on all the variables in tha dataset. 
# 2) As there were no missing values in the dataset we started the model building process with all the features. 
# 3) We removed multicolinearity from the dataset and analyzed the model summary report by dropping insignificant variables.
# 4) We checked for four different assupmtion and all of them are satisfied.
# 5) Our model equation is: log (Price) = 	( 4.6493858232666225 ) *  const + ( -0.012500455079104186 ) *  CRIM + ( 0.11977319077019655 ) *  CHAS + ( -1.0562253516683224 ) *  NOX + ( 0.05890657510928066 ) *  RM + ( -0.044068890799405444 ) *  DIS + ( 0.007848474606244105 ) *  RAD + ( -0.04850362079499876 ) *  PTRATIO + ( -0.029277040479796776 ) *  LSTAT 

# 1) We can say that price of the house is depend on 8 different features. 
# 2) As crime rate, concentration of nitric oxide, distance to employement centeres, pupil-teacher ratio and lower status of population increase the price of house drops. 
# 3) The price of house increases if house is near to charles river or easily accessible to radial highways or number of rooms per dwelling. 
