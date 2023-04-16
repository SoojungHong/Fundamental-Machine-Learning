# read npy file and visualize in 2D and 3D visualization 

import numpy as np

#---------------------------
# Data load from npy file
#---------------------------

fname = r"C:\Users\SHong\Downloads\N-Caltech101\N-Caltech101\training\gerenuk\gerenuk_3.npy"
data = np.load(fname)

print(data.shape)
timestamp = list(data[:,2])
x = list(data[:,0])
y = list(data[:,1])
pol = list(data[:,3])


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

    
#----------------    
# visualize
#----------------

# Display the image in grayscale
fig = plt.figure()
fig.suptitle('Balance of event polarities')
maxabsval = np.amax(np.abs(img))
plt.imshow(img, cmap='gray', clim=(-maxabsval,maxabsval))
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



# _____________________________________________________________________________
# %% 3D plot 
# Time axis in horizontal position

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
