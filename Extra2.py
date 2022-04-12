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
# splitting the data into train and test
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

l = [1, 5, 10, 15, 25]

rmse, mape = [], []
# running for loop for different values of lag
for i in l:
    window = i # The lag=i
    model = AR(train, lags = window, old_names = False) 
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params # Get the coefficients of AR model
    
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
    # finding the root mean square error and mean absolute percentage error and storing it
    rmspe_test = math.sqrt(mean_squared_error(test, predictions))/np.mean(test)*100
    mape_test = mean_absolute_percentage_error(test, predictions)*100
    rmse.append(rmspe_test)
    mape.append(mape_test)

# initialising a dataframe to store rmse and mape values and get good representation of output
q = pd.DataFrame(index = [1, 2, 3, 4, 5])
q.index.name = "lag"
q["RMSE %"] = np.round(rmse,3)
q["MAPE %"] = np.round(mape,3)
print("RMSE % and MAPE % for each value of lag")
print(q)
# plotting the barplot for rootmeansquare percentage error
plt.bar(l, rmse, color = "cyan", edgecolor = "black")
plt.xlabel("Lagged Values")
plt.ylabel("RMSE %")
plt.show()
# plotting the barplot for mean absolute percentage error
plt.bar(l, mape, color = "orange", edgecolor = "black")
plt.xlabel("Lagged Values")
plt.ylabel("MAPE %")
plt.show()


