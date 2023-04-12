## reference 
https://github.com/LarryDong/event_representation

## Event frame
Event frame is the simplest representation. Considering polarity, each pixel in image would only be +1/0/-1, which means a positive/no/negative event occurs here.
This is just updating the last event polarity value to (x,y) image pixel 

## Event Accumulate frame
Event Accumulate frame is sometimes called **event intensity frame**, or **histogram of events[1]**. Each pixel would be a number that indicate the intensity. For a uint8 image, the range would be (0, 255), where 128 means no events (or the same number of positive/negative events), >128 means more positive events occurred here, and vice versa.


## Time-surface 
Time-surface is also caled surface of active events, which include both spatio and temporal information. The value of each pixel should be
