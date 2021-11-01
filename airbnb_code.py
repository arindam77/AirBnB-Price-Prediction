# -*- coding: utf-8 -*-
"""
@author: arindam


"""

"""Import all libraries"""

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error as mse
from uszipcode import SearchEngine
search = SearchEngine(simple_zipcode=True)


"""Importing Data"""

training_data_df = pd.read_csv("C:/Users/Arindam/Music/DELL/AirBnB10K/train10k.csv")
test_data_df = pd.read_csv("C:/Users/Arindam/Music/DELL/AirBnB10K/test2k.csv")

"""Checking missing values"""

missing_data_df = training_data_df.isnull().sum(axis=0).reset_index()
missing_data_df.columns = ['column_name', 'missing_count']
missing_data_df = missing_data_df.loc[missing_data_df['missing_count']>0]
missing_data_df = missing_data_df.sort_values(by='missing_count')
print(missing_data_df)

"""Handling missing values in train and test data"""


# For missing 'bathrooms' values
training_data_df['bathrooms'] = training_data_df['bathrooms'].fillna(0)
test_data_df['bathrooms'] = test_data_df['bathrooms'].fillna(0)

# For missing 'bedrooms' values
training_data_df['bedrooms'] = training_data_df['bedrooms'].fillna(0)
test_data_df['bedrooms'] = test_data_df['bedrooms'].fillna(0)

# For missing 'beds' values
training_data_df['beds'] = training_data_df['beds'].fillna(0)
test_data_df['beds'] = test_data_df['beds'].fillna(0)

# For missing 'host_has_profile_pic' values
training_data_df.loc[training_data_df.host_has_profile_pic == 't', 'host_has_profile_pic'] = 1
training_data_df.loc[training_data_df.host_has_profile_pic == 'f', 'host_has_profile_pic'] = 0
training_data_df['host_has_profile_pic'] = training_data_df['host_has_profile_pic'].fillna(0)
test_data_df.loc[test_data_df.host_has_profile_pic == 't', 'host_has_profile_pic'] = 1
test_data_df.loc[test_data_df.host_has_profile_pic == 'f', 'host_has_profile_pic'] = 0
test_data_df['host_has_profile_pic'] = test_data_df['host_has_profile_pic'].fillna(0)

# For missing 'neighbourhood' values
training_data_df['neighbourhood'] = training_data_df['neighbourhood'].fillna('Unknown')
test_data_df['neighbourhood'] = test_data_df['neighbourhood'].fillna('Unknown')

# For missing 'review_scores_rating' values
training_data_df['review_scores_rating'] = training_data_df['review_scores_rating'].fillna(0)
test_data_df['review_scores_rating'] = test_data_df['review_scores_rating'].fillna(0)

# For missing 'host_response_rate' values
training_data_df['host_response_rate'] = training_data_df['host_response_rate'].apply(str).str.replace('%','')
training_data_df['host_response_rate'] = training_data_df['host_response_rate'].fillna('0')
training_data_df['host_response_rate'] = pd.to_numeric(training_data_df['host_response_rate'])
test_data_df['host_response_rate'] = test_data_df['host_response_rate'].apply(str).str.replace('%','')
test_data_df['host_response_rate'] = test_data_df['host_response_rate'].fillna('0')
test_data_df['host_response_rate'] = pd.to_numeric(test_data_df['host_response_rate'])

# For missing 'thumbnai_url' values
training_data_df['thumbnail_url'] = training_data_df['thumbnail_url'].fillna('Unknown')
test_data_df['thumbnail_url'] = test_data_df['thumbnail_url'].fillna('Unknown')

# For missing 'last_review' values
training_data_df['last_review'] = training_data_df['last_review'].fillna('00-00-00')
test_data_df['last_review'] = test_data_df['last_review'].fillna('00-00-00')

# For missing 'first_review' values
training_data_df['first_review'] = training_data_df['first_review'].fillna('00-00-00')
test_data_df['first_review'] = test_data_df['first_review'].fillna('00-00-00')

# For missing 'host_since' values
training_data_df['host_since'] = training_data_df['host_since'].fillna('00-00-00')
test_data_df['host_since'] = test_data_df['host_since'].fillna('00-00-00')

# For missing 'host_identity_verified' values
training_data_df.loc[training_data_df.host_identity_verified == 't', 'host_identity_verified'] = 1
training_data_df.loc[training_data_df.host_identity_verified == 'f', 'host_identity_verified'] = 0
training_data_df['host_identity_verified'] = training_data_df['host_identity_verified'].fillna(0)
test_data_df.loc[test_data_df.host_identity_verified == 't', 'host_identity_verified'] = 1
test_data_df.loc[test_data_df.host_identity_verified == 'f', 'host_identity_verified'] = 0
test_data_df['host_identity_verified'] = test_data_df['host_identity_verified'].fillna(0)

# For missing zipcode values
training_data_df['zipcode'] = training_data_df['zipcode'].fillna(0)
training_data_df.loc[training_data_df.zipcode == ' ', 'zipcode'] = 0
idx = training_data_df.index[training_data_df['zipcode']==0].tolist()
for i in idx:
    lat = training_data_df['latitude'][i]
    lon = training_data_df['longitude'][i]
    result = np.max(search.by_coordinates(lat, lon, radius=30, returns=5))
    training_data_df['zipcode'][i]=result.values()[0]    
test_data_df['zipcode'] = test_data_df['zipcode'].fillna(0)
test_data_df.loc[test_data_df.zipcode == ' ', 'zipcode'] = 0
idx = test_data_df.index[test_data_df['zipcode']==0].tolist()
for i in idx:
    lat = test_data_df['latitude'][i]
    lon = test_data_df['longitude'][i]
    result = np.max(search.by_coordinates(lat, lon, radius=30, returns=5))
    test_data_df['zipcode'][i]=result.values()[0]
    
#Rechecking for missing values in the test dataframe
missing_data_df = training_data_df.isnull().sum(axis=0).reset_index()
missing_data_df.columns = ['column_name', 'missing_count']
missing_data_df = missing_data_df.loc[missing_data_df['missing_count']>0]
missing_data_df = missing_data_df.sort_values(by='missing_count')
missing_data_df    
training_data_df['int_price'] = np.exp(training_data_df['log_price'])


"""Data Preprocessing"""

training_data_df.to_csv('C:/Users/Arindam/Music/DELL/AirBnB10K/train10k.csv', encoding='utf-8', index=False)
test_data_df.to_csv('C:/Users/Arindam/Music/DELL/AirBnB10K/test2k.csv', encoding='utf-8', index=False)

Rtraining_data_df = pd.read_csv('C:/Users/Arindam/Music/DELL/AirBnB10K/train10k.csv')
Rtest_data_df = pd.read_csv('C:/Users/Arindam/Music/DELL/AirBnB10K/test2k.csv')

"""Eliminating features that has no effect on pricing"""

Rtraining_data_df.drop(['int_price'], axis =1, inplace=True)

Rtraining_data_df.drop(['id'], axis =1, inplace=True)
Rtest_data_df.drop(['id'], axis =1, inplace=True)

Rtraining_data_df.drop(['log_price'], axis =1, inplace=True)

Rtraining_data_df.drop(['neighbourhood'], axis =1, inplace=True)
Rtest_data_df.drop(['neighbourhood'], axis =1, inplace=True)

Rtraining_data_df.drop(['description'], axis =1, inplace=True)
Rtest_data_df.drop(['description'], axis =1, inplace=True)

Rtraining_data_df.drop(['first_review'], axis =1, inplace=True)
Rtest_data_df.drop(['first_review'], axis =1, inplace=True)

Rtraining_data_df.drop(['last_review'], axis =1, inplace=True)
Rtest_data_df.drop(['last_review'], axis =1, inplace=True)

Rtraining_data_df.drop(['host_since'], axis =1, inplace=True)
Rtest_data_df.drop(['host_since'], axis =1, inplace=True)

Rtraining_data_df.drop(['thumbnail_url'], axis =1, inplace=True)
Rtest_data_df.drop(['thumbnail_url'], axis =1, inplace=True)

Rtraining_data_df.drop(['zipcode'], axis =1, inplace=True)
Rtest_data_df.drop(['zipcode'], axis =1, inplace=True)

Rtraining_data_df.drop(['amenities'], axis =1, inplace=True)
Rtest_data_df.drop(['amenities'], axis =1, inplace=True)

Rtraining_data_df.drop(['name'], axis =1, inplace=True)
Rtest_data_df.drop(['name'], axis =1, inplace=True)

Rtraining_data_df.drop(['latitude'], axis =1, inplace=True)
Rtest_data_df.drop(['latitude'], axis =1, inplace=True)

Rtraining_data_df.drop(['longitude'], axis =1, inplace=True)
Rtest_data_df.drop(['longitude'], axis =1, inplace=True)
Rtraining_data_df.drop(['instant_bookable'], axis =1, inplace=True)
Rtest_data_df.drop(['instant_bookable'], axis =1, inplace=True)
print(Rtraining_data_df.dtypes)
rtraining_data_df = Rtraining_data_df.copy()
rtest_data_df = Rtest_data_df.copy()
print(rtraining_data_df.shape)
print(rtest_data_df.shape)


"""Features requiring One Hot Encoding from Train and Test Data"""
def one_hot_encoding(training_data_df,test_data_df,columns):
    
    for i,column in enumerate(columns):
        Xtrain = training_data_df[str(column)].T
        Xtest = test_data_df[str(column)].T
        
        # training_data_df
        lb=LabelBinarizer()
        lb.fit(Xtrain)
        X_classes = len(lb.classes_)
        Xenc = lb.transform(Xtrain)
        Xtrain_enc = pd.DataFrame(data = Xenc, columns = lb.classes_)
        training_data_df.drop([str(column)], axis =1, inplace=True)
        
        # test_data_df
        Xenc = lb.transform(Xtest)
        Xtest_enc = pd.DataFrame(data = Xenc, columns = lb.classes_)
        test_data_df.drop([str(column)], axis =1, inplace=True)
        
        print('Number of classes in '+str(column)+ ' = '+ str(X_classes))
        training_data_df = pd.concat((training_data_df,Xtrain_enc),axis=1)
        test_data_df = pd.concat((test_data_df,Xtest_enc),axis=1) 
    return training_data_df,test_data_df
r_training_data_df , r_test_data_df = one_hot_encoding(rtraining_data_df,rtest_data_df,['city','property_type', 'room_type', 'bed_type', 'cancellation_policy', 'host_response_rate'])

print(r_test_data_df.shape)
print(r_training_data_df.shape)

"""Applying Principal Component Analysis (PCA) """
pca = PCA()
pca_fit = pca.fit_transform(r_training_data_df)
pca_fit.shape

"""Implementing Linear Regression"""

X = pca_fit
y = training_data_df['log_price']

# print("FOR  Linear Regression")
kfold = KFold(n_splits=10,random_state=52,shuffle=True)
average = 0
average1 = 0
for train_idx, test_idx in kfold.split(X,y):    
    X_train, X_CV = X[train_idx], X[test_idx]
    y_train, y_CV = y[train_idx], y[test_idx]
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
        
    pred_CV = lr.predict(X_CV)
    MSE = mse(y_CV, pred_CV)
    RMSE = sqrt(MSE)
    average = average + RMSE

    score = lr.score(X_CV, y_CV)
    average1 = average1 + score
    
    print('R square score = ',score)
    print('RMSE = ',RMSE)

RMSE_AVG = average/10
Rscore_AVG = average1/10
print('*---------------------------*')
print('Average Rscore = ', Rscore_AVG)
print('Average RMSE = ',RMSE_AVG)


# print("FOR Ridge Regression")
"""Implementing Ridge Regression"""


kfold = KFold(n_splits=10,random_state=52,shuffle=True)
average = 0
average1 = 0

for train_idx, test_idx in kfold.split(X,y):    
    
    X_train, X_CV = X[train_idx], X[test_idx]
    y_train, y_CV = y[train_idx], y[test_idx]
    
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    
    pred_CV = ridge.predict(X_CV)
    MSE = mse(y_CV, pred_CV)
    RMSE = sqrt(MSE)
    average = average + RMSE
    
    score = ridge.score(X_CV, y_CV)
    average1 = average1 + score
    
    print('R square score = ',score)
    print('RMSE = ',RMSE)

RMSE_AVG = average/10
Rscore_AVG = average1/10
print('*---------------------------*')
print('Average Rscore = ', Rscore_AVG)
print('Average RMSE = ', RMSE_AVG)

# print("FOR Lasso Regression")

"""Implementing Lasso Regression"""


kfold = KFold(n_splits=10,random_state=52,shuffle=True)
average = 0
average1 = 0

for train_idx, test_idx in kfold.split(X,y):    
    
    X_train, X_CV = X[train_idx], X[test_idx]
    y_train, y_CV = y[train_idx], y[test_idx]
    
    lasso = Lasso(0.0001)
    lasso.fit(X_train, y_train)
        
    pred_CV = lasso.predict(X_CV)
    MSE = mse(y_CV, pred_CV)
    RMSE = sqrt(MSE)
    average = average + RMSE
    
    score = lasso.score(X_CV, y_CV)
    average1 = average1 + score
    
    print('R square score = ',score)
    print('RMSE = ',RMSE)

RMSE_AVG = average/10
Rscore_AVG = average1/10
print('*---------------------------*')
print('Average Rscore = ', Rscore_AVG)
print('Average RMSE = ', RMSE_AVG)

# print("FOR Elastic Net Regression")

"""Implementing Elastic Net Regression"""

kfold = KFold(n_splits=10,random_state=52,shuffle=True)
average = 0
average1 = 0

for train_idx, test_idx in kfold.split(X,y):    
    
    X_train, X_CV = X[train_idx], X[test_idx]
    y_train, y_CV = y[train_idx], y[test_idx]
    
    ENreg = ElasticNet()
    ENreg.fit(X_train, y_train)
    
    pred_CV = ENreg.predict(X_CV)
    MSE = mse(y_CV, pred_CV)
    RMSE = sqrt(MSE)
    average = average + RMSE
    
    score = ridge.score(X_CV, y_CV)
    average1 = average1 + score
    
    print('R square score = ',score)
    print('RMSE = ',RMSE)

RMSE_AVG = average/10
Rscore_AVG = average1/10
print('*---------------------------*')
print('Average Rscore = ', Rscore_AVG)
print('Average RMSE = ', RMSE_AVG)


print("FOR GRID SEARCH")
"""Applying Grid Search"""

parameters = {"alpha":np.logspace(-2,2,50)}
lasso_grid = GridSearchCV(lasso, parameters, cv=10) 
lasso_grid.fit(X,y)

print('Hyper Parameters for Lasso:\n',lasso_grid.best_params_)
print('Score for Lasso:',lasso_grid.best_score_)
lasso_grid.cv_results_

# """TESTING FOR RIDGE """
# parameters = {"alpha":np.logspace(-2,2,50)}
# ridge_grid = GridSearchCV(ridge, parameters, cv=10) 
# ridge_grid.fit(X,y)

# print('Hyper Parameters for Ridge:\n',ridge_grid.best_params_)
# print('Score for Lasso:',ridge_grid.best_score_)
# ridge_grid.cv_results_



"""Predicting values for test dataset"""

pca_pred = PCA()
pca_pred.fit(r_training_data_df)
X_train = pca_pred.transform(r_training_data_df)
y_train = training_data_df['log_price']

X_test = pca_pred.transform(r_test_data_df)

lasso = Lasso(0.01)
lasso.fit(X_train, y_train)

# Price prediction for test dataset
price_predicted = lasso.predict(X_test)
print(price_predicted)



