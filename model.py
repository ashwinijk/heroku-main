import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import csv
from sklearn.linear_model import LinearRegression

dataset = [["experience", "test", "interview", "salary"], [0,8,9,50000], [0,8,6,45000], [5,6,7,60000],
           [2,10,10,65000], [7,9,6,70000], [3,7,10,62000], [10,7,5,72000],[11,7,8,80000]];

X = dataset
y = dataset[:][1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3

print(y)
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
