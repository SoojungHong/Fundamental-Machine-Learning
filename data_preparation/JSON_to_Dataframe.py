import pandas as pd

"""
Read JSON from file
"""
df = pd.read_json('text.json')
print(df)
print(df.info())


"""
Flattening nested list from JSON object
"""
df = pd.read_json('test2.json')
print(df)  # dataframe

# to flattern values with json_normalization()
import json
# load data using Python JSON module
with open('test2.json', 'r') as f:
    data = json.loads(f.read()) # loaded one is json, not yet dataframe

print(data)

# flatten data
df_nested_list = pd.json_normalize(data, record_path=['students'])
print('df_nested_list : \n', df_nested_list)

# we can use the argument meta to specify a list of metadata we want in the result.

df_nested_list_with_meta = pd.json_normalize(
    data,
    record_path=['students'],
    meta=['school_name', 'class']
)
print('df_nested_list_with_meta : \n', df_nested_list_with_meta)


"""
Flattening nested list and dict from JSON object
"""
import json
with open('test3.json', 'r') as f:
    data = json.loads(f.read())

# Normalize data
df = pd.json_normalize(data, record_path=['students'])
print(df)

"""
include class, president (a property of info), and tel (a property of contacts.info), 
we can use the argument meta to specify the path to the property.
"""
df = pd.json_normalize(
    data,
    record_path = ['students'],
    meta = [
        'class',
        ['info','president'],
        ['info', 'contacts', 'tel']
    ]
)

print(df)

"""
Extracting a single value from deeply nested JSON
"""
from glom import glom
df = pd.read_json('test4.json')
print(df['students'].apply(lambda row: glom(row, 'grade.math')))
