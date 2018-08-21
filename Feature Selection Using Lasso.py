
# StandardScaler method & MinMaxScaler method to be used as data preprocessing before lasso regression model selecting features

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from scipy.stats import boxcox

train = pd.read_csv("raw_data.csv")
train = train.dropna()

# StandardScaler as data preprocessing method, Lasso regression as feature selection method
scaler = StandardScaler()
train_data = scaler.fit_transform(train.iloc[:,1:])

train_x = train_data[:,1:]
train_y = train_data[:,0]

lassocv = LassoCV()
lassocv.fit(train_x, train_y)

a = [False, False]
a.extend(list(lassocv.coef_ != 0))
a = np.array(a)
StandardScaler_result = list(train.iloc[:,a].columns)

print("StandardScaler, number of selected features：", len(StandardScaler_result))
print(StandardScaler_result)


# MinMaxScaler as data preprocessing method, Lasso regression as feature selection method
minmax_scaler = MinMaxScaler()
train_data = minmax_scaler.fit_transform(train.iloc[:,1:])

for i in range(train_data.shape[1]):
    train_data[:,i] = boxcox(train_data[:,i]+1)[0]


train_x = train_data[:,1:]
train_y = train_data[:,0]

lassocv = LassoCV()
lassocv.fit(train_x, train_y)

a = [False, False]
a.extend(list(lassocv.coef_ != 0))
a = np.array(a)
minmax_result = list(train.iloc[:,a].columns)

print("MinMaxScaler, number of selected features：", len(boxcox_result))
print(minmax_result)


# Union two sets of different method as selected features
set(StandardScaler_result).intersection(set(minmax_result))
len(set(StandardScaler_result).intersection(set(minmax_result)))

set(StandardScaler_result).union(set(minmax_result))
