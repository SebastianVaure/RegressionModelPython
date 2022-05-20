#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import pandas as pd # data science essentials
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # enhanced data visualization
import statsmodels.formula.api as smf # regression modeling
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LinearRegression

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# specifying the path and file name
file = './birthweight_low.xlsx'


# reading the file into Python
bwght = pd.read_excel(file)


# In[2]:


#Calculating missing values
bwght.isnull().sum()


# In[3]:


#Getting the average of the variables with missing values
print(bwght [["meduc"]].mean().round(0))
print(bwght[["npvis"]].mean().round(0))
print(bwght[["feduc"]].mean().round(0))


# In[4]:


#Filling the missing values with the averages

# meduc
fill = 14
bwght['meduc'] = bwght['meduc'].fillna(fill)


# npvis
fill = 11
bwght['npvis'] = bwght['npvis'].fillna(fill)


# feduc
fill = 13
bwght['feduc'] = bwght['feduc'].fillna(fill)


# In[5]:


# Declaring set of x-variables
x_variables = ['fage','cigs', 'drink', 'fmaps', 'foth']


# Looping to make x-variables suitable for statsmodels
for val in x_variables:
    print(f"{val} +")


# In[6]:


# Preparing explanatory variable data
bwght_data = bwght.loc[ : , x_variables]


# preparing response variable data
bwght_target = bwght.loc[ : , 'bwght']

# preparing training and testing sets (all letters are lowercase)
x_train, x_test, y_train, y_test = train_test_split(
            bwght_data,
            bwght_target,
            test_size = 0.25,
            random_state = 219)


# checking the shapes of the datasets
print(f"""
Training Data
-------------
X-side: {x_train.shape}
y-side: {y_train.shape}


Testing Data
------------
X-side: {x_test.shape}
y-side: {y_test.shape}
""")


# In[7]:


# merging X_train and y_train so that they can be used in statsmodels
bwght_train = pd.concat([x_train, y_train], axis = 1)


# build a model - All significant variables 
lm_best = smf.ols(formula =  """bwght ~ fage +
cigs +
drink +
fmaps""", 
                  data = bwght_train)


# fit the model based on the data
results = lm_best.fit()



# analyze the summary output
print(results.summary())


# In[8]:


# applying model in scikit-learn

# Preparing a DataFrame based the the analysis above
ols_data   = bwght.loc[ : , x_variables]


# Preparing the target variable
bwght_target = bwght.loc[ : , 'bwght']



#setting up more than one train-test split

# FULL X-dataset (normal Y)
x_train_FULL, x_test_FULL, y_train_FULL, y_test_FULL = train_test_split(
            bwght_data,     # x-variables
            bwght_target,   # y-variable
            test_size = 0.25,
            random_state = 219)


# OLS p-value x-dataset (normal Y)
x_train_OLS, x_test_OLS, y_train_OLS, y_test_OLS = train_test_split(
            ols_data,         # x-variables
            bwght_target,   # y-variable
            test_size = 0.25,
            random_state = 219)


# In[9]:


# INSTANTIATING a model object
lr = LinearRegression()


# FITTING to the training data
lr_fit = lr.fit(x_train_OLS, y_train_OLS)


# PREDICTING on new data
lr_pred = lr_fit.predict(x_test_OLS)


# SCORING the results
print('OLS Training Score :', lr.score(x_train_OLS, y_train_OLS).round(4))  # using R-square
print('OLS Testing Score  :',  lr.score(x_test_OLS, y_test_OLS).round(4)) # using R-square


# saving scoring data for future use
lr_train_score = lr.score(x_train_OLS, y_train_OLS).round(4) # using R-square
lr_test_score  = lr.score(x_test_OLS, y_test_OLS).round(4)   # using R-square


# displaying and saving the gap between training and testing
print('OLS Train-Test Gap :', abs(lr_train_score - lr_test_score).round(4))
lr_test_gap = abs(lr_train_score - lr_test_score).round(4)


# In[10]:


import sklearn.linear_model # linear models


# In[11]:


# INSTANTIATING a model object
lasso_model = sklearn.linear_model.Lasso(alpha = 0.9,
                                         normalize = True) # default magitude


# FITTING to the training data
lasso_fit = lasso_model.fit(x_train_FULL, y_train_FULL)


# PREDICTING on new data
lasso_pred = lasso_fit.predict(x_test_FULL)


# SCORING the results
print('Lasso Training Score :', lasso_model.score(x_train_FULL, y_train_FULL).round(4))
print('Lasso Testing Score  :', lasso_model.score(x_test_FULL, y_test_FULL).round(4))


## the following code has been provided for you ##

# saving scoring data for future use
lasso_train_score = lasso_model.score(x_train_FULL, y_train_FULL).round(4) # using R-square
lasso_test_score  = lasso_model.score(x_test_FULL, y_test_FULL).round(4)   # using R-square


# displaying and saving the gap between training and testing
print('Lasso Train-Test Gap :', abs(lasso_train_score - lasso_test_score).round(4))
lasso_test_gap = abs(lasso_train_score - lasso_test_score).round(4)


# In[12]:


# INSTANTIATING a model object
ard_model = sklearn.linear_model.ARDRegression()


# FITTING the training data
ard_fit = ard_model.fit(x_train_FULL, y_train_FULL)


# PREDICTING on new data
ard_pred = ard_fit.predict(x_test_FULL)


print('Training Score:', ard_model.score(x_train_FULL, y_train_FULL).round(4))
print('Testing Score :', ard_model.score(x_test_FULL, y_test_FULL).round(4))


# saving scoring data for future use
ard_train_score = ard_model.score(x_train_FULL, y_train_FULL).round(4)
ard_test_score  = ard_model.score(x_test_FULL, y_test_FULL).round(4)


# displaying and saving the gap between training and testing
print('ARD Train-Test Gap :', abs(ard_train_score - ard_test_score).round(4))
ard_test_gap = abs(ard_train_score - ard_test_score).round(4)


# In[14]:


# Comparing results

print(f"""
Model      Train Score      Test Score       Train-Test Gap
-----      -----------      -----------     ---------------     
OLS        {lr_train_score}           {lr_test_score}            {abs(lr_train_score - lr_test_score).round(4)}
Lasso      {lasso_train_score}           {lasso_test_score}            {abs(lasso_train_score - lasso_test_score).round(4)}   
*ARD       {ard_train_score}             {ard_test_score}            {abs(ard_train_score - ard_test_score).round(4)}

*This is the final model""")



# 
# 
# 
# 
# 
# 
# 
# References:
# Chase Kusterer, Script 04 - Linear Regression.
