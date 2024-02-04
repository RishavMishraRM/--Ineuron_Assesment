#Explain how you can implement ML in a real world application.
#Train an SVM regressor on : Bengaluru housing dataset
#Must include in details:
#EDA
#Feature engineering

#Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

#Loading the dataset
data = pd.read_csv('Bengaluru_House_Data.csv')
data.head()

#Exploratory Data Analysis
data.shape
data.info()
data.describe()
data.isnull().sum()

#Dropping the columns which are not required
data.drop(['area_type', 'availability', 'society', 'balcony'], axis = 1, inplace = True)

#Filling the missing values
data['location'].fillna('Sarjapur  Road', inplace = True)
data['size'].fillna('2 BHK', inplace = True)
data['bath'].fillna(2, inplace = True)
data['total_sqft'].fillna(1000, inplace = True)
data['bath'].fillna(2, inplace = True)
data['balcony'].fillna(1, inplace = True)

#Feature Engineering
#Converting the size column to integer
data['size'] = data['size'].apply(lambda x: int(x.split(' ')[0]))

#Converting the total_sqft column to float
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
data[~data['total_sqft'].apply(is_float)].head(10)
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)
data = data[data.total_sqft.notnull()]
data['price'] = data['price']*100000
data['price_per_sqft'] = data['price']/data['total_sqft']
data['location'] = data['location'].apply(lambda x: x.strip())
location_stats = data['location'].value_counts(ascending=False)
location_stats_less_than_10 = location_stats[location_stats<=10]
data['location'] = data['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(data['location'].unique())

#Outlier Removal
data = data[~(data.total_sqft/data.size<300)]
data = data[~(data.total_sqft/data.size>10000)]
data = data[~(data.price_per_sqft<3000)]
data = data[~(data.price_per_sqft>10000)]
data.shape

#Encoding the categorical variables
dummies = pd.get_dummies(data['location'])
data = pd.concat([data, dummies.drop('other', axis = 1)], axis = 1)
data.drop('location', axis = 1, inplace = True)

#Splitting the data into dependent and independent variables    
X = data.drop('price', axis = 1)
y = data['price']

#Splitting the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training the model
svr = SVR()
svr.fit(X_train, y_train)

#Predicting the results
y_pred = svr.predict(X_test)

#Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R2 Score:', r2)

#Hyperparameter Tuning
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3)
grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
mse = mean_squared_error(y_test, grid_predictions)
r2 = r2_score(y_test, grid_predictions)
print('Mean Squared Error:', mse)
print('R2 Score:', r2)