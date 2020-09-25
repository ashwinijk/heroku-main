import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import csv
from sklearn.linear_model import LinearRegression

#Experience, Test_score, Interview_score
X = [[0,8,9], [0,8,6], [5,6,7], [2,10,10], [7,9,6], [3,7,10], [10,7,5],[11,7,8]];

#salary data
y = [50000,45000,60000,65000,70000,62000,72000,80000]
print(X)
print(y)

regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
