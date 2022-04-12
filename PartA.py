# Abhay Singh Raihal
# B20144
# 9319337883

# importing required libraries
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# reading the csv file
df = pd.read_csv("daily_covid_cases.csv")

# a
print("a")
df["Date"] = pd.to_datetime(df["Date"])
# plotting the number of covid cases with each day
plt.plot(df["Date"], df["new_cases"], color = "red")
plt.xlabel("Index of the day")
plt.ylabel("No. of covid-19 cases")
plt.xticks(rotation='vertical')
plt.show()

# b
print("b")
lag = df["new_cases"].iloc[:-1]          # lagged values
orig = df["new_cases"].iloc[1:]          # original values
# finding the pearson's correlation coefficeint
cor = pearsonr(lag,orig)[0]
print(round(cor,3))

# c
print("c")
# plotting the scatterplot of lagged values with original values to see the correlation between 2
plt.scatter(lag, orig, color = "lime", marker = "1")
plt.xlabel("independent variable")
plt.ylabel("dependent variable")
plt.title("Scatterplot b/w given time sequence and one-day lagged generated sequence")
plt.show()

# d
print("d")
corrval = []
# running for loop to iterate over all lagged values as asked in question
for i in range(1,7):
    lag = df["new_cases"].iloc[:-i]           # lagged values
    orig = df["new_cases"].iloc[i:]           # original values
    # finding pearson's correlation coefficient and storing it
    corrval.append(pearsonr(lag,orig)[0])
l = [1, 2, 3, 4, 5, 6]
# plotting the pearson's correlation coefficient with the lag values used
plt.plot(l, corrval, marker = "2", color = "blue")
plt.xlabel("lagged values")
plt.ylabel("correlation-coefficients")
plt.title("Correlation coefficients vs lagged values")
plt.show()

# e
print("e")
# plotting the autocorrelation with lags = 6
plot_acf(df["new_cases"], lags = 6, marker="1", color = "orange")
plt.xlabel("lagged values")
plt.ylabel("correlation")
plt.show()





