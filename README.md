## Project Motivation
This project is the Capstone Project for Data Scientist Nanodegree at Udacity. The goal of the project is to create a pipeline that takes an image and detects whether a human or dog is present; then, predicting the breed for the dog or deciding what dog breed the human looks similar to.

## Project Overview
In this project, I have learned how to build a pipeline to process real-world, user-supplied images. Given an image of a dog, my algorithm will identify an estimate of the dog’s breed. If supplied an image of a human, the code will identify the resembling dog breed. By completing this project, I understood the challenges involved in joining together a series of models designed to perform various tasks in a data processing pipeline.

## Libraries Used
- Numpy
- OpenCV
- Matplotlib
- Keras (with Tensorflow backend)

## Required Dataset
1. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `/Dog_Breed_Classifier/dog_images`. 

2. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `/Dog_Breed_Classifier/lfw`.

3. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `/Dog_Breed_Classifier/bottleneck_features`.

## File Description
- **`lfw (dir) :`** contains the human images.
- **`dog_images (dir) :`** contains the dog images.
- **`dog_app.ipynb :`** The Jupyter Notebook where the project is implemented.
- **`sample_images (dir) :`** contains sample images to test the final algorithm.
- **`saved_models (dir) :`** contains the best models saved after training the models.
- **`extract_bottleneck_features.py :`** Python file that contains functions to extract bottleneck features.
- **`haarcascades (dir) :`** contains the pre-trained face detectors provided by OpenCV, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).
- **`bottleneck_features (dir) :`** contains the bottleneck features of different architectures like VGG19, Resnet50, InceptionV3, or Xception.

## Result
In this project, I used both existing neural network and created a model from scratch. The neural network that I created by myself was 7% accurate. Considering that less than 1% would have been by just pure random classification, it’s better than that.

However, using an existing model like Xception (using Transfer Learning), I was able to obtain an accuracy of 83% to classify a dog's breed.
