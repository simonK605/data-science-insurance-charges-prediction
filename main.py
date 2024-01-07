import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
sns.countplot(x="sex", data=data)
plt.show()
