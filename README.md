# FacialLandmarkRegression

Dataset can be found in the following link
http://vis-www.cs.umass.edu/lfw/

The pretrained model for alex net is automatically downloaded and loaded in training.
 'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
 
 dataset.py -- creates the image tensor by cropping the bounding box from the image and randomly apply data augmentation like crop with offset, flip ,brightening 
 
 LFWNet.py -- modified alex net for the LFW dataset
 
 Training /Validation Error:
 
 ![Error plot](https://github.com/vishnusanjayrs/FacialLandmarkRegression/blob/master/plots/Training%20validation%20error.png)
 
 Accuracy:
 
 ![Accuracy plot](https://github.com/vishnusanjayrs/FacialLandmarkRegression/blob/master/plots/Test%20Accuracy.png)
 


An average 1% MSE is achieved on the
testing data. Based on the loss, we are expecting 1%0.5 x 225 = 20 pixel absolute error.
However, based on the percentage of detected points, the average error seems to be at
15 pixels, this is different than what we are expecting. One reason can be due to the
variance of the testing data. We do expect better results to be achieved when more
training data is given.
For a future project, an increasing number of training data would be helpful. Also,
applying random blurring for data augmentation can be implemented. Group the loss
data on left eyes, right eyes, mouth, and nose can also be investigated. From our
result, the prediction on the nose seems less accurate than other points.
 
 
