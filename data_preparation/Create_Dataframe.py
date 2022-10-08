"""
Creating a basic single column Pandas Dataframe
"""
import pandas as pd
data = ['apple', 'sky', 'sea', 'wine']
df = pd.DataFrame(data)
print(df) # That creates a default column name (0) and index names (0,1,2,3..).


"""
Making a DataFrame from a Dictionary of lists
"""
data_dict = {'Country': ['India', 'China', 'United States', 'Pakistan', 'Indonesia'],
             'Population': [1393409038, 1444216107, 332129157, 225199937, 276361783],
             'Currency': ['Indian Rupee', 'Renminbi', 'US Dollar', 'Pakistani Rupee', 'Indonesian Rupiah']}
df = pd.DataFrame(data=data_dict)
print(df)


"""
Making a DataFrame from a list of Lists
"""
# Create a list of lists where each inner list is a row of the DataFrame
data_list = [['India', 1393409038, 'Indian Rupee'],
             ['China', 1444216107, 'Renminbi'],
             ['United States', 332129157, 'US Dollar'],
             ['Pakistan', 225199937, 'Pakistani Rupee'],
             ['Indonesia', 276361783, 'Indonesian Rupiah']]

df = pd.DataFrame(data=data_list, columns=['Country', 'Population', 'Currency'])
print(df)

"""
Create a dataframe with a list of Dictionaries
"""
# Create a list of dictionaries where the keys are the column names and the values are a particular feature value.
list_of_dicts = [{'Country': 'India', 'Population': 139409038, 'Currency': 'Indian Rupee'},
                 {'Country': 'China', 'Population': 1444216107, 'Currency': 'Renminbi'},
                 {'Country': 'United States', 'Population': 332129157, 'Currency': 'US Dollar'},
                 {'Country': 'Pakistan', 'Population': 225199937, 'Currency': 'Pakistani Rupee'},
                 {'Country': 'Indonesia', 'Population': 276361763, 'Currency': 'Indonesian Rupiah'}, ]

df = pd.DataFrame(list_of_dicts)
print(df)


"""
Making a DataFrame from a Numpy array
"""
import numpy as np
data_nparray = np.array([['India', 1393409038, 'Indian Rupee'],
                         ['China', 1444216107, 'Renminbi'],
                         ['United States', 332129157, 'US Dollar'],
                         ['Pakistan', 225199937, 'Pakistani Rupee'],
                         ['Indonesia', 276361783, 'Indonesian Rupiah']])

df = pd.DataFrame(data=data_nparray)
print(df)

# Create dataframe with user specified column names
data_nparray = np.array([['India', 1393409038, 'Indian Rupee'],
                         ['China', 1444216107, 'Renminbi'],
                         ['United States', 332129157, 'US Dollar'],
                         ['Pakistan', 225199937, 'Pakistani Rupee'],
                         ['Indonesia', 276361783, 'Indonesian Rupiah']])

df = pd.DataFrame(data=data_nparray, columns=['Country', 'Population', 'Currency'])

"""
Create numpy array that each of array element (list) is a feafure values
"""
# Create a numpy array where each inner array is a list of values of a particular feature
data_array = np.array(
    [['India', 'China', 'United States', 'Pakistan', 'Indonesia'],
     [1393409038, 1444216107, 332129157, 225199937, 276361783],
     ['Indian Rupee', 'Renminbi', 'US Dollar', 'Pakistani Rupee', 'Indonesian Rupiah']])

# Create a dictionary where the keys are the column names and each element of data_array is the feature value.
dict_array = {
    'Country': data_array[0],
    'Population': data_array[1],
    'Currency': data_array[2]}

# Create the DataFrame
df = pd.DataFrame(dict_array)
print(df)

"""
Making a DataFrame using a zip function
"""
# Create the countries list(1st object)
countries = ['India', 'China', 'United States', 'Pakistan', 'Indonesia']

# Create the population list(2nd object)
population = [1393409038, 1444216107, 332129157, 225199937, 276361783]

# Create the currency list (3rd object)
currency = ['Indian Rupee', 'Renminbi', 'US Dollar', 'Pakistani Rupee', 'Indonesian Rupiah']

# Zip the three objects
data_zipped = zip(countries, population, currency)

# Pass the zipped object as the data parameter and mention the column names explicitly
df = pd.DataFrame(data_zipped, columns=['Country', 'Population', 'Currency'])

df

"""
Zip 
"""
languages = ['Java', 'Python', 'JavaScript']
versions = [14, 3, 6]

result = zip(languages, versions)
print('result : ', result) # this object -> result :  <zip object at 0x111ad4040>
print(list(result)) # Output: [('Java', 14), ('Python', 3), ('JavaScript', 6)]

"""
Making indexed Pandas DataFrame
"""
# Create the DataFrame
data_dict = {'Country': ['India', 'China', 'United States', 'Pakistan', 'Indonesia'],
             'Population': [1393409038, 1444216107, 332129157, 225199937, 276361783],
             'Currency': ['Indian Rupee', 'Renminbi', 'US Dollar', 'Pakistani Rupee', 'Indonesian Rupiah']}

# Make the list of indices
indices = ['Ind', 'Chi', 'US', 'Pak', 'Indo']

# Pass the indices to the index parameter
df = pd.DataFrame(data=data_dict, index=indices)
print(df)


"""
Making a new dataframe from existing dataframe
Dataframe can be concat by vertically or horizontally
"""
# Create 1st DataFrame
countries = ['India', 'China', 'United States', 'Pakistan', 'Indonesia']
df1 = pd.DataFrame(data=countries, columns=['Country'])
print(df1)

two_columns = {
    'col1': ['a', 'b', 'c', 'd', 'e'],
    'col2': ['aa', 'ba', 'ac', 'ad', 'ae']
}
df2 = pd.DataFrame(two_columns)
print(df2)

# concat two dataframes
joined_df = pd.concat([df1, df2], axis=1) # horizontally concat, i.e. horizontally
print(joined_df)

"""
vertically, i.e. row-wise Dataframe concatenate 
"""
d1 = {
    'col1':[1, 2, 3],
    'col2':[4, 5, 6]
}
df1 = pd.DataFrame(d1) # dictionary should be transformed to DataFrame
d2 = {
    'col1':[7,8],
    'col2':[9,10]
}
df2 = pd.DataFrame(d2)

joined_df = pd.concat([df1, df2], axis=0) # axis = 0 means, vertically (row-wise) concatenation
print(joined_df)


"""
while reading JSON file, increase the row of DataFrame 
"""
col_name = ['id', 'ridge2_x_diff', 'ridge2_y_diff', 'class'] # class 1 : yes, class 0 : no

df = pd.DataFrame(columns=col_name) # dataframe that contains all yes, no data

for f in glob.glob(yes_data_path + "*.json"):
    fname = os.path.basename(f)

    with open(yes_data_path+fname, "r") as yes_f:
        data = json.load(yes_f)
        ridge2_x_move_diff = np.max(data['Ridge2']['x']) - np.min(data['Ridge2']['x'])
        ridge2_y_move_diff = np.max(data['Ridge2']['y']) - np.min(data['Ridge2']['y'])

        df.loc[len(df.index)] = [fname, ridge2_x_move_diff, ridge2_y_move_diff, 1]
