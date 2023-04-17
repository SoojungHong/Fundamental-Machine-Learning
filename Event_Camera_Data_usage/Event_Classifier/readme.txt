# Reference : rpg_event_representation_learning 

# example of training 
    - input is event.npy files 
    - input of the model is batch sized voxel (i.g., 4 boxes) 
    - output batch sized number of Tensor with probabilities of all labels
    - cross-entropy loss is calculated based on output (prediction) and label 
