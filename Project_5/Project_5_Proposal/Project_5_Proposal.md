# Project 5 Proposal

------

### Background

Biologists currently rely on either the naked eye to observe effects of a drug on a sample of cells, or utilizing basic and highly sample-specific code to do so. In hopes of implementing a data science model that can be applied generally to a wide array of nuclei samples, the time of observing and understanding of a drug's effectivness on a sample of cells can be significantly reduced. By saving time in the detection process, cost of labor will decrease and efforts can be focused more on diagnosis.

------

### Topic

Identification / segmentation of cells' nuclei from microscope images

------

### Domain

Given the topic of this project, the realm of work will most likely be in *image segmentation*. As a result, utilization of *Convolutional Neural Networks* is optimal. To build the networks, TensorFlow, Keras, and Numpy will be used. The most favorable networks to try are **U-Net** and **One-Hundred Layers Tiramisu**. To handle numerical data formatting, Pandas is appropriate. Currently, the metric of choice to judge the model is accuracy. We want to ensure that the model will segment out the right sections of an image within an range of error.

------

### Data

The dataset is downloaded from Kaggle as part of a recently finished competition started in January 2018 and hosted by MIT. There is a collection of image files, each showing apparent nuclei in that image. Based on reading of the Kaggle data description, extraneous cellular structures do not appear in the images most likely due to their having been filtered out by a separate process. 

Included in the training data for each image, there are *masks*. Masks are just highlighted regions in the image that segment out each individual nucleus. Each individual nucleus has 1 respective mask for each image file.

Lastly, there is a .csv file with columns: {**Image ID**, **Encoded Pixels**}. Image ID refers to each unique image file. Each observation in *Encoded Pixels* appears to be a list of integers; Presumeably, the integers seem to represent the boundary, interpreted as pixel locations, of each nucleus in an image.

![ff3407842ada5bc18be79ae453e5bdaa1b68afc842fc22fa618ac6e6599d0bb3 copy](/Users/christopher/Desktop/projects/Metis_Projects/Project_5/Project_5_Proposal/img/ff3407842ada5bc18be79ae453e5bdaa1b68afc842fc22fa618ac6e6599d0bb3 copy.png)

![e7188c37d87ef9e3f6bf3a226fede0f1cf753c472ab4182330d18a20b34cae07](/Users/christopher/Desktop/projects/Metis_Projects/Project_5/Project_5_Proposal/img/e7188c37d87ef9e3f6bf3a226fede0f1cf753c472ab4182330d18a20b34cae07.png)

![Screen Shot 2018-08-30 at 11.55.57 AM](/Users/christopher/Desktop/projects/Metis_Projects/Project_5/Project_5_Proposal/img/Screen Shot 2018-08-30 at 11.55.57 AM.png)

------

### Known Unknowns

1. Interpretation of the "Encoded Pixels" column in the .csv files is unclear.
2. Each image appears to have different types of cells due to the various shapes of nuclei. We do not know what conditions the nuclei for each image have been exposed to. As a result, we can treat all nuclei in general as the same type with similar geometry until assumptions have to be changed.