# FacialLandmarkRegression

Dataset can be found in the following link
http://vis-www.cs.umass.edu/lfw/

The pretrained model for alex net is automatically downloaded and loaded in training.
 'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
 
 dataset.py -- creates the image tensor by cropping the bounding box from the image and randomly apply data augmentation like crop with offset, flip ,brightening 
 
 LFWNet.py -- modified alex net for the LFW dataset
 
 Accuracy:
 
 
