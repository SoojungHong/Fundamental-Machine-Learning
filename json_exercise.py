
"""
JSON : JavaScript Object Notation

"""

import json
import requests

data = {
    "president": {
        "name": "Zaphod Beeblebrox",
        "species": "Betelgeusian"
    }
}

#print(data)


"""
using json object, create json file 
"""
with open("data_file.json", "w") as write_file:
    json.dump(data, write_file)


"""
read json file
"""
with open("data_file.json", "r") as read_file:
    data = json.load(read_file)


print("read json : ", data)


"""
real world example

you’ll use JSONPlaceholder, a great source of fake JSON data for practice purposes.
"""
response = requests.get("https://jsonplaceholder.typicode.com/todos")
todos = json.loads(response.text)

print(todos)

# Map of userId to number of complete TODOs for that user
todos_by_user = {}


for todo in todos:
    print(todo)
    if todo['completed'] == True :
        try :
            # increment the existing user's count
            todos_by_user[todo['userId']] += 1
        except KeyError:
            # this user never existed in dict
            todos_by_user[todo['userId']] = 1


print(todos_by_user)


# Create a sorted list of (userId, num_complete) pairs. i.e. sort with largest completed task's user first

# todos_by_user.sort() # this is in-place sort, i.e. sort itself directly
top_users = sorted(todos_by_user.items(), key=lambda x: x[1], reverse=True)
print(top_users)


# Get the maximum number of complete TODOs.
print('debug : ', top_users[0][1])
max_complete = top_users[0][1]

users = []

for user, count in top_users:
    if count >= max_complete:
        users.append(str(user))
    else:
        break

max_users = " and ".join(users)

print(max_users)




