# Project 5 Summary:

## Cellular Nucleus Image Segementation

A common goal in biology is to count to see whether the number of nuclei for a type of cell has increased or decreased, or to calculate their change in size in response a test drug. Traditional methods rely on identifying locations of these nuclei with the naked eye and either manually counting or marking their positions with certain software programs. The challange further grows when dealing with different types of cells. You may still have to rely on the naked eye, but now you have to adjust the software settings accordingly for each different type of cell so that the computer can render a viewable image. This is quite a tedious and time consuming task. 

My goal for this project was to implement a convolutional neural network to see if I could segment out regions of nuclei from given microscope images in hopes of reducing the time required to identify and count the number of nuclei within a sample. This could potentially reduce the timeline, and thus cost, for the process of testing drugs.

## Data

In acquiring data, Kaggle has a [dataset](https://www.kaggle.com/c/data-science-bowl-2018/data), labeled *stage1_train*, of 670 raw microscope images of nuclei of various types. Each image has a multitude of nuclei. In addition, the ground truth labels designating the locations of these nuclei are represented as binary masks stored as a separate image. 

Note that Kaggle provides a separate mask image for each individual nucleus within an image. As a result, during my image pre-processing work, I decided to overlay all the separated masks into one image. This made calculating loss easier when time came to implementing our model.

Ideally, we would like to compare the output image from the neural network to the ground truth mask to evaluate the effectiveness of the model. The *test set* on Kaggle does not have ground truth masks, only raw images. Therefore, to create an evaluable test set, I performed a 600/70 train/test split on the 670 images from *stage1_train*. Lastly, all images were resized to 128x128 pixels for consistency.

## Tools

In resizing and converting the training and test images between RGB and grayscale, I utilized the python libraries Scikit-Image and OpenCV. I then used Keras with a Tensorflow backend to build my neural network. 

Particularly when training a convolutional neural network, you would want to run your code on your machine's GPU, rather than it's CPU. A task that takes 20 minutes on your CPU would finish less than 60 seconds when running on your GPU, relatively speaking. As a result, I uploaded my data to an [AWS](https://aws.amazon.com/) GPU instance and trained my model there. 

By default, your machine will use the CPU. Based on my experience, I found the process of specifying to your machine to use the GPU to be more cumbersome than necessary, even for the top results on Stack Overflow regarding this topic. In my research, I found an elegant solution. Assuming you are using Python, simply wrap your code with the following below, and witness your training time decrease significantly:

```python
import tensorflow as tf

with tf.device('/gpu:0'):
    '''
    your code here 
    '''
```

After the model has trained and predicted, I used NumPy to manage the image arrays to evaluate my results.

## Model

I created my convolutional neural network based on the [U-Net architecture](https://arxiv.org/abs/1505.04597), one that has been shown to work adequately when handling segmentation of biomedical images. The first 4 layers consist of convolutions each utilizing a different number of 3x3 pixels kernels, coupled with max pooling and 10% dropouts. The second half of the neural network then upsamples the convolutions back to the original dimensions of the input image. All the layers applied the "[elu](https://sefiks.com/2018/01/02/elu-as-a-neural-networks-activation-function/)" activation function, a derivative of the common "[relu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))". Binary cross entropy was used for the loss in conjunction with the adam optimizer.

As previously stated, each image has been previously scaled to 128x128 pixels. In addition, the images are represented as an array of values corresponding to the RGB format. Specifically the neural network takes in an image of the dimensions (128,128,3), and outputs a grayscale image of dimensions (128,128,1).

To binarize the output image, I took the average grayscale value of the image, and applied a threshold based on this value. Anything below the average was black, and anything at the average or above was white. I binarized the output so that we can compare more easily to the ground truth to evaluate the results.

The metric used was Intersection over Union, or also technically known was Jaccard Distance. This metric simply evaluates how dissimilar two separate datasets are. In this specific case of image segmentation, the metric value is calculated by taking the area of regions of nuclei predicted correctly by the neural network, and dividing that by the total area of false negative, false positive, and true positive regions. We can observe these regions by overlaying the ground truth ontop of the predicted result. The Intersection over Union metric value thus ranges between 0 to 1. A perfect score of 1 translates to the model being able to pick up on all regions of nuclei within an image without any false negatives and false positives.

## Conclusions

Although I originally configured the model to run for 50 epochs, there was no significant improvement in the validation loss and Intersection over Uinon metric after about 20 or so epochs. During training, the model achieved an average binary cross entorpy loss of 0.0971 with an average metric value of 0.81.

When performing on the test set of 70 images that I had set aside, the model achieved an average Intersection over Union value of 0.75. This value is not too far off 0.81, so we can conclude that we are not drastically overfitting during training.

From this, we can conclude that the U-Net architecture does work adequately in segementing our regions of nuclei. Future work on the project would be to implement other segmentation architectures such as 100 Layers Tiramisu, or AlexNet to see if any imporovements can be made. 

The takeaway here is that regarding a biologist's line of work and needs, this current result, althought not 100% optimal, is already extremely satisfactory. The model consistently captures large chunks of nuclei, and any regions of error are insignificant for the interests of the biologist. As a biologist, with these marked regions of nuclei, you can apply other programs to now count their number or calculate their size. This will enable a biologist to perform the same task of analyzing a type of cell's reaction to a test drug, but in a much shorter amount of time. A common challenge that persists however, is when the cells' nuclei overlap. How does the computer know to treat those regions as 2 distinct nuclei, versus one? I do have my proposed solutions, but that is for another discussion. 

It is important to reiterate that now with this model, you can segment our regions of nuclei in a format that the computer can undertand, and going forward, you can apply other computational techniques that were either not viable, or too time consuming for the task at hand.

