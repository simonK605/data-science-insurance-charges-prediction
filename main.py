import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/insurance.csv")

# Drop duplicates, inplace=True will make changes in the original dataframe
data.drop_duplicates(inplace=True)

# Get the first 5 rows
# print(data.head())

# Get the number of rows and columns
# print(data.shape)

# Get the data types
# print(data.info())

# Get data if there are any duplicates
# print(data.duplicated().sum())

# Get the number of males and females
# print(data.value_counts("sex"))

# Plot the number of males and females
# sns.countplot(x="sex", data=data)
# plt.show()

# Convert categorical data to numerical
labelEncoder = preprocessing.LabelEncoder()
data['smoker'] = labelEncoder.fit_transform(data['smoker'])
data['sex'] = labelEncoder.fit_transform(data['sex'])
data['region'] = labelEncoder.fit_transform(data['region'])

x = data[["age", "sex", "bmi", "children", "smoker", "region"]]
y = data[["charges"]]

# Split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Scale the data
scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

def modelResults(predictions):
    print("Mean Absolute Error: ", mean_absolute_error(y_test, predictions))
    print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, predictions)))

lr = LinearRegression()
lr.fit(scaled_x_train, y_train)

predictions = lr.predict(scaled_x_test)
modelResults(predictions)