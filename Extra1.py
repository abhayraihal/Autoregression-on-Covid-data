# Abhay Singh Raihal
# B20144
# 9319337883

# importing required libraries
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.ar_model import AutoReg as AR

# reading the csv file
series = pd.read_csv('daily_covid_cases.csv', parse_dates=['Date'], index_col=['Date'], sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
# splitting the data into train and test data
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

window = 5 # The lag=5
model = AR(train, lags = window, old_names = False) 
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
print("Coefficients obtained from AR model are : ", np.round(coef, 3).astype(str))

#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

# plt.scatter(test, predictions, marker = "2", color = "blue")  # plotting the scatterplot between actual and predicted test data
# plt.xlabel("actual test values")
# plt.ylabel("predicted test values")
# plt.show()

l1 = [i+1 for i in range(len(test))]
# now we will plot the actual test data and predicted test data to see any change between 2
plt.plot(l1, test, color = "lime", label = "actual test data")
plt.plot(l1, predictions, color = "red", label = "predicted test data")
plt.legend()
plt.show()

# # calculating root mean square percentage error
# rmspe_test = math.sqrt(mean_squared_error(test, predictions))/np.mean(test)*100
# print("RMSE between actual and predicted test data : {} %".format(round(rmspe_test, 3)))
# # calculating mean absolute percentage error
# mape_test = mean_absolute_percentage_error(test, predictions)*100
# print("MAPE between actual and predicted test data : {} %".format(round(mape_test, 3)))




