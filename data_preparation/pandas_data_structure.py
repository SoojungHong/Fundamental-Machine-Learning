"""
Pandas three data structure
- Series
- DataFrames
- Panel
"""
import pandas as pd

"""
Series : one-dimensional data structure with homogenous data type
The size of the series is immutable(cannot be changed) but its values are mutable.
"""
# creating Series with Numpy
import numpy as np
data = np.array(['a', 'b', 'c'])
ex_1 = pd.Series(data, index=[0,1,2])
print(data) # numpy array
print(ex_1) # pandas Series


# create Series with dictionary
data = {'a':0, 'b':1, 'c':2}
s = pd.Series(data)  # Note:Dictionary keys are used to construct index here.
print(s)

# Creating Series with Scalar
s = pd.Series(5, index=[0, 1, 2, 3])
print(s) # for each index, print 5


"""
DataFrame is two-dimensional data structure with heterogeneous data type
The size and values of DataFrame are mutable.
"""

# creating DataFrame with list
mylist = [1, 2, 3]
df = pd.DataFrame(mylist)
print(df)

data = [['aa', 0], ['bb', 1], ['cc', 2]]
df = pd.DataFrame(data, columns=['key', 'value'], dtype=float)
print(df)

# read csv file and make DataFrame
import pandas as pd
data = pd.read_csv('Iris.csv')
print(type(data))

# high level statistics with Pandas
print(data.describe())
print(data['SepalLengthCm'].describe())

# If you to change the data type of a specific column in the data frame, use the .astype() function.
test = data['SepalLengthCm'].astype(str)
print(data['SepalLengthCm'])
print(type(test[0]))

# rows in DataFrame are selected, use iloc
print(data.iloc[0:3, ].head())
print(data.iloc[4,:])

# Filtering
print(data[data['Species'] == 'Iris-virginica'].head())

#delete the 'Id' column from the dataframe
print(data)
print(data.drop('Id',axis= 1).head())  # drop column 'Id'  # #other method data.drop(columns="Id").head()

# Delete the rows with labels 0,1
data.drop([0,1],axis=0).head(3)

# Delete the first two rows using iloc selector
data.iloc[2:,].head(3)

# Finding missing values
data.isna().head(3)

# count( ) : Number of non-null observations
print(data['SepalLengthCm'].count())

# sum( ): Sum of values
print(data['SepalLengthCm'].sum())

# groupby()-this function is used for grouping data in pandas

print(data.groupby("Species").count())

"""
The panel is a three-dimensional data structure with heterogeneous data. 
The size and values of a Panel are mutable.
"""
