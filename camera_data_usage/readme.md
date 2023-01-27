## Camera data 
### Event Camera data
Retrieving accuate semantic information is difficult, because high dynamic range (HDR) and high speed condition
Therefore, it is difficult to retrieve semantic information for image-based algorithm.

The advantage of using Event camera, they feature a much high dynamic range and resilient to motion blur. 
It is still early stage for event camera, for using Semantic Segmentation. 

Event-camera (such as  Dynamic Vision Sensor) quickly output sparse moving edge information.
Their sparse and rapid output is ideal for driving low-latency CNN. 
It is potentially allowing real-time interaction for human pose estimators. 

In code for model, Event-camera data make frame from event stream representation. 

How to use Event-camera data?
common first step: encode the event information into an image-like representation, in order to facilitate its processing.
We discuss in detail different previous work event representations (encoding spatial and temporal information) as well as our proposed representation (with a different way of encoding the temporal information)

How Event data (by Even camera) look like? Answer is in Chapter 3.1  



### Time of Flight (ToF) camera 
ToF camera is more robust to illumination and color variation. 
The result of ToF camera is the depth video feature. (Compared to RGB camera, RGB or Grayscale camera don't have depth information)
Depth information localize object in 3D space.
Depth camera such as MS Kinect, RealSense, Asus Xtion provide 3D geometric information fo the scene. 
It is beneficial for various applications such as action recognition, gesture classification. Also person detection, detection of anomaly behavior. 
Other RGB image can provide depth information like adding additional depth channel and make RGB-D. 
However Time-of-Flight camera data provide the depth information of realistic scenarios and following camera angles as they would be common in a surveilance context.

What Is HDR (High Dynamic Range)? 
Dynamic range describes the extremes in that difference, and how much detail can be shown in between. Essentially, dynamic range is display contrast, and HDR ...

What is Kalman Filter? 
Kalman filters are used to optimally estimate the variables of interests when they can't be measured directly, but an indirect measurement is available. 
They are also used to find the best estimate of states by combining measurements from various sensors in the presence of noise.

How is Kalman filter used for tracking?
In it, the Kalman filter is used to predict and update the location and velocity of an object given a video stream, and detections on each of the frames. At some point, we wanted to also account for the camera tilt and pan by adjusting the location of the object given the angles of movement.


## Reference of Camera sensor data usage 

https://www.tel.com/museum/exhibition/principle/cmos.html



https://github.com/uzh-rpg/ess/blob/main/train.py



https://rpg.ifi.uzh.ch/docs/ECCV22_Sun.pdf



https://github.com/Shathe/Ev-SegNet



https://drive.google.com/file/d/1eTX6GXy5qP9I4PWdD4MkRRbEtfg65XCr/view


## Reference of ToF camera data usage

https://www.researchgate.net/publication/353544192_High-speed_object_detection_with_a_single-photon_time-of-flight_image_sensor

This paper shows how pre-process the ToF image before using as input to UNet for Semantic Segmentation. 
'''
Each frame must be pre-processed before being fed to the neural network. All data types
are normalized to values between 0 and 1. Depth data is corrected with a calibration frame
to compensate for temporal skew in photon timing across the SPAD array. SPC-256 data is
processed with a median ﬁlter of size 2
×
2 to remove any outliers in the frame. SPC-64 is a
resized version of SPC-256. Histogram data is altered by subtracting the median of each pixel’s
histogram to remove the background level. When summing all the bins of this histogram, we get
a type of intensity image that preserves the active illumination from the laser but has minimal
contribution from ambient light. This active intensity data, together with the depth frame, are
concatenated to form the data type Act_I-D. The dataset as a whole is augmented by applying a
horizontal ﬂip to each frame, thus doubling the quantity of frames. Finally, the dataset’s order is
shuﬄed randomly to prevent tuning the weights of the neural network speciﬁcally for a given class
before seeing examples of others. Not doing so can lead to a local minimum in the optimisation
problem far from the absolute minimum'''
