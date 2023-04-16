
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


#---------------------------------------
# %% 3D plot 
# Time axis in horizontal position
#---------------------------------------

m = 2000 # Number of points to plot
print("Space-time plot and movie: numevents = ", m)

# Plot without polarity
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect('equal') # only works for time in Z axis
ax.scatter(x[:m], timestamp[:m], y[:m], marker='.', c='b')
ax.set_xlabel('x [pix]')
ax.set_ylabel('time [s]')
ax.set_zlabel('y [pix] ')
ax.view_init(azim=-90, elev=-180) # Change viewpoint with the mouse, for example
plt.show()




# _____________________________________________________________________________
# %% Voxel grid

num_bins = 5
print("Number of time bins = ", num_bins)

t_max = np.amax(np.asarray(timestamp[:m]))
t_min = np.amin(np.asarray(timestamp[:m]))
t_range = t_max - t_min
dt_bin = t_range / num_bins # size of the time bins (bins)
t_edges = np.linspace(t_min,t_max,num_bins+1) # Boundaries of the bins

# Compute 3D histogram of events manually with a loop
# ("Zero-th order or nearest neighbor voting")
hist3d = np.zeros(img.shape+(num_bins,), np.int)
for ii in range(m):
    idx_t = int( (timestamp[ii]-t_min) / dt_bin )
    if idx_t >= num_bins:
        idx_t = num_bins-1 # only one element (the last one)
    hist3d[y[ii],x[ii],idx_t] += 1

# Checks:
print("hist3d")
print(hist3d.shape)
print(np.sum(hist3d)) # This should equal the number of votes


