import numpy as np
import matplotlib.pyplot as plt

"""
example 1 
"""
x = np.arange(1, 10, 3)
y = np.arange(2, 11, 3)

m, b = np.polyfit(x, y, 1)  # ployfit() : Least squares polynomial fit.

#plt.plot(x, y, 'yo', x, m*x+b, '--k') # arg : x, y, format (ro or yo, etc) <-- this example plots two plots 
#plt.plot(x, y, 'ro')
plt.plot(x, m*x+b, '--k')
plt.show()



"""
example 2 
"""
plt.style.use("seaborn-darkgrid")
fig, ax = plt.subplots(1, 1)
ax.set_box_aspect(1)

plt.scatter(X, y)

line = np.linspace(-1, 1, num=100).reshape(-1, 1)
plt.plot(line, regressor.predict(line), c="peru")
plt.show()



"""
scatter plot example
"""

import matplotlib.pyplot as plt
x = df['SepalLengthCm']
y = df['PetalLengthCm']

setosa_x = x[:50] # in data, setosa is until row 50
setosa_y = y[:50]

versicolor_x = x[50:]
versicolor_y = y[50:]

plt.figure(figsize=(8,6))
plt.scatter(setosa_x, setosa_y, marker='+', color='green')
plt.scatter(versicolor_x, versicolor_y, marker='_', color='red')
plt.show()

