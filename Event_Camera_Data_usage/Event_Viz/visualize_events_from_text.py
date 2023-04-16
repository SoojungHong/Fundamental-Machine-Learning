
import os
import numpy as np
from matplotlib import pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

plt.close('all')

filename_sub = 'slider_depth/events_chunk.txt'

def extract_data(filename):
    infile = open(filename, 'r')
    timestamp = []
    x = []
    y = []
    pol = []
    for line in infile:
        words = line.split()
        timestamp.append(float(words[0]))
        x.append(int(words[1]))
        y.append(int(words[2]))
        pol.append(int(words[3]))
    infile.close()
    return timestamp,x,y,pol

# Call the function to read data    
timestamp, x, y, pol = extract_data(filename_sub)



# For this exercise, we just provide the sensor size (height, width)
img_size = (180,240)


# %% Brightness incremet image (Balance of event polarities)
num_events = 5000  # Number of events used
print("Brightness incremet image: numevents = ", num_events)

# Compute image by accumulating polarities.
img = np.zeros(img_size, np.int)
for i in range(num_events):
    # Need to convert the polarity bit from {0,1} to {-1,+1} and accumulate
    img[int(y[i]),int(x[i])] += (2*pol[i]-1)

#-------------------------------------    
# Display the image in grayscale
#-------------------------------------
fig = plt.figure()
fig.suptitle('Balance of event polarities')
maxabsval = np.amax(np.abs(img))
plt.imshow(img, cmap='gray', clim=(-maxabsval,maxabsval))
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()


#-----------------------------------------------------------------------------------------------------
# %% 2D Histograms of events, split by polarity (positive and negative events in separate images)
#-----------------------------------------------------------------------------------------------------
img_pos = np.zeros(img_size, np.int)
img_neg = np.zeros(img_size, np.int)
for i in range(num_events):
    if (pol[i] > 0):
        img_pos[int(y[i]),int(x[i])] += 1 # count events
    else:
        img_neg[int(y[i]),int(x[i])] += 1

fig = plt.figure()
fig.suptitle('Histogram of positive events')
plt.imshow(img_pos)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()


fig = plt.figure()
fig.suptitle('Histogram of negative events')
plt.imshow(img_neg)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()


#---------------------------------------------------------------
# %% Thresholded brightness increment image (Ternary image)
#---------------------------------------------------------------

# What if we only use 3 values in the event accumulation image?
# Saturated signal: -1, 0, 1
# For example, store the polarity of the last event at each pixel
img = np.zeros(img_size, np.int)
for i in range(num_events):
    img[int(y[i]),int(x[i])] = (2*pol[i]-1)  # no accumulation; overwrite the stored value

# Display the ternary image
fig = plt.figure()
fig.suptitle('Last event polarity per pixel')
plt.imshow(img, cmap='gray')
#plt.imshow(img, cmap='bwr')
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()



# _____________________________________________________________________________
# %% Time surface (or time map, or SAE="Surface of Active Events")
num_events = len(timestamp)
print("Time surface: numevents = ", num_events)

img = np.zeros(img_size, np.float32)
t_ref = timestamp[-1] # time of the last event in the packet
tau = 0.03 # decay parameter (in seconds)
for i in range(num_events):
    img[int(y[i]),int(x[i])] = np.exp(-(t_ref-timestamp[i]) / tau)

fig = plt.figure()
fig.suptitle('Time surface (exp decay). Both polarities')
plt.imshow(img)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()


#------------------------------------------------------------------------------------
# %% Time surface (or time map, or SAE), using polarity as sign of the time map
#------------------------------------------------------------------------------------
sae = np.zeros(img_size, np.float32)
for i in range(num_events):
    if (pol[i] > 0):
        sae[int(y[i]),int(x[i])] = np.exp(-(t_ref-timestamp[i]) / tau)
    else:
        sae[int(y[i]),int(x[i])] = -np.exp(-(t_ref-timestamp[i]) / tau)

fig = plt.figure()
fig.suptitle('Time surface (exp decay), using polarity as sign')
plt.imshow(sae, cmap='seismic') # using color (Red/blue)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()


#---------------------------------------------------------------------------------
# %% "Balance of time surfaces"
# Accumulate exponential decays using polarity as sign of the time map
#---------------------------------------------------------------------------------
sae = np.zeros(img_size, np.float32)
for i in range(num_events):
    if (pol[i] > 0):
        sae[int(y[i]),int(x[i])] += np.exp(-(t_ref-timestamp[i]) / tau)
    else:
        sae[int(y[i]),int(x[i])] -= np.exp(-(t_ref-timestamp[i]) / tau)

fig = plt.figure()
fig.suptitle('Time surface (exp decay), balance of both polarities')
#plt.imshow(sae)
maxabsval = np.amax(np.abs(sae))
plt.imshow(sae, cmap='seismic', clim=(-maxabsval,maxabsval))
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()


#---------------------------------------------------
# %% Average timestamp per pixel
#---------------------------------------------------
sae = np.zeros(img_size, np.float32)
count = np.zeros(img_size, np.int)
for i in range(num_events):
    sae[int(y[i]),int(x[i])] += timestamp[i]
    count[int(y[i]),int(x[i])] += 1
    
# Compute per-pixel average if count at the pixel is >1
count [count < 1] = 1  # to avoid division by zero
sae = sae / count

fig = plt.figure()
fig.suptitle('Average timestamps regardless of polarity')
plt.imshow(sae)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()
