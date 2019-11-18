# EDAN95
Applied Machine Learning

## Assignment 2, Lab 3
### Objectives
The objectives of this assignment are to:

Write a program to recognize flowers on images
Learn how to manage an image data set
Apply convolutional networks to images
Know what Python generators are
Understand class activation
Write a short report on your experiments. This report is mandatory to pass the assignment.

### Programming

#### Collecting a Dataset
1. You will collect a dataset from Kaggle. Register, it is free, and you will have access to lots of datasets.
2. Download the Flower corpus. You can find a local copy in the /usr/local/cs/EDAN95/datasets folder.
3. Alternatively, you can use another dataset, provided that it has more than two categories. Tell your teacher in advance in this case to check if it is acceptable. You can use the Google dataset search: https://toolbox.google.com/datasetsearch.
4. Split randomly your dataset into training, validation, and test sets: Use a 60/20/20 ratio. You will read all the file names and create a list of pairs, (file_name, category). You will then shuffle your list and save your partition of the data.

To speed up the lab, you can also
1. Start with the partition available in the /usr/local/cs/EDAN95/datasets folder; or
2. You can also use the code available from https://github.com/pnugues/edan95.

#### Building a Simple Convolutional Neural Network

Create a simple convolutional network and train a model with the train set. You can start from the architecture proposed by Chollet, Listing 5.5, and a small number of epochs. Use the ImageDataGenerator class to scale your images as in the book:
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
                
You will need to modify some parameters so that your network handles multiple classes.
You will also adjust the number of steps so that your generator in the fitting procedure sees all the samples.
You will report the training and validation losses and accuracies and comment on the possible overfit.
Apply your network to the test set and report the accuracy as well as the confusion matrix you obtained. As with fitting, you may need to adjust the number of steps so that your network tests all the samples.
Try to improve your model by modifying some parameters and evaluate your network again.
Using Image Augmentation
The flower dataset is relatively small. A way to expand such datasets is to generate artificial images by applying small transformations to existing images. Keras provides a built-in class for this: ImageDataGenerator. You will reuse it and apply it to the flower data set.

Using the network from the previous exercise, apply some transformations to your images. You can start from Chollet, Listing 5.11.
Report the training and validation losses and accuracies and comment on the possible overfit.
Apply your network to the test set and report the accuracy as well as the confusion matrix you obtained.
Using a Pretrained Convolutional Base
Some research teams have trained convolutional neural networks on much larger datasets. We have seen during the lecture that the networks can model conceptual patterns as they go through the layers. This was identified by Le Cun in his first experiments (http://yann.lecun.com/exdb/lenet/). In this last part, you will train classifiers on top of a pretrained convolutional base.

Build a network that consists of the Inception V3 convolutional base and two dense layers. As in Chollet, Listing 5.17, you will program an extract_features() function.
Train your network and report the training and validation losses and accuracies.
Apply your network to the test set and report the accuracy as well as the confusion matrix you obtained.
Modify your program to include an image transformer. Train a new model.
Apply your network to the test set and report the accuracy as well as the confusion matrix you obtained.
Passing the Assignment
To pass the assignment, you need to reach an accuracy of 75 (even 80 ideally) with your best network.

### Report
You will write a short report of about two pages on your experiments:

You will describe the architectures you designed and the results you obtained;
You will run Chollet's notebook 5.4 and read the article Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization by Selvaraju et al. From this, you will reformulate and comment the paragraph on Visualizing heatmaps of class activation in the notebook
To write the report, you will use Overleaf.com. The submission procedure will be specified later.

You must submit this report no later than one week after you have complete the lab.
