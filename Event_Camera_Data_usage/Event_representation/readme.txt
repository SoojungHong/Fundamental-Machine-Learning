## reference 
https://github.com/LarryDong/event_representation

## Event frame
Event frame is the simplest representation. Considering polarity, each pixel in image would only be +1/0/-1, which means a positive/no/negative event occurs here.
This is just updating the last event polarity value to (x,y) image pixel 

## Event Accumulate frame
Event Accumulate frame is sometimes called **event intensity frame**, or **histogram of events[1]**. Each pixel would be a number that indicate the intensity. For a uint8 image, the range would be (0, 255), where 128 means no events (or the same number of positive/negative events), >128 means more positive events occurred here, and vice versa.


## Time-surface 
Time-surface is also caled surface of active events, which include both spatio and temporal information. The value of each pixel should be

## 3D graph representation
First proposed by Yin Bi in his ICCV19 and TIP20 papers[8,9]. Also appeared in Yongjian Deng's [6] Paper, and Simon Schaefer's paper[10] in CVPR2022.
The key idea is to use a 3D graph to orgnize event stream for further processing (like classification).

Steps: 
1. Voxelize the event stream; 
2. Select N important voxels (based on the number of events in each voxel) for denoise; 
3. Calcuate the 2D histgram as the feature vector in each voxel; 
4. The 3D coordinate, and the feature, construct a Vertex in a graph; 
5. Data association and further processing can be dealed by graph (see paper for more details).


## Binary Event History Image (BEHI)
BEHI representation is first proposed in Wang's IROS2022 paper [11]. This representation draws all events (without polarity) in one frame during the whole histroy time 
This representation can be used to predict the future position of a moving object with less data (only 1 binary frame). The above image shows a BEHI when a ball is flying towards a robot camera. Please check the paper for more details (the collision position & time can be estimated by a network).



