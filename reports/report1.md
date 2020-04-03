---
title: 'A simple Convolutional Neural Network' 


subtitle: "Project 1"
author:
- Jesús Adolfo Mejía de Dios

documentclass: scrartcl
numbersections: true
colorlinks: true
urlcolor: blue

---

 


# Introduction

This project is focused on the implementation of a given Convolutional Neural Network (CNN) to classify $32 \times 32$ RGB images. That CNN uses three convolutional layers with RELU activation functions, three Pool layers and one fully connected layer activated by a softmax function. Such CNN is part of an [online demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html). 

![Convolutional Neural Network used in this project.](imgs/conv.png){ width=90%}

The optimization method to train this CNN is called [ADAMW](https://arxiv.org/abs/1711.05101) which is a variant of [ADAM](https://arxiv.org/abs/1412.6980v8) defined by fixing weight decay regularization.

The following section is focused on describing the dataset to train the CNN described above.

## Dataset

The dataset used here is the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset that consists of 60000 color images ($32 \times 32$) in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 


![CIFAR-10 dataset](imgs/cifar10.png){ width=50% }

Different CNNs have been proposed to improve the learning rate on that dataset (see figure below). Actually, the best algorithm is called Big Transfer (BiT): General Visual Representation Learning with 99.3% accuracy. 

![Image Classification on CIFAR-10.](imgs/chart.png){ width=90% }

# Implementation

The provided CNN was implemented in the [Julia Programming Language](https://julialang.org/) since it is open-source software and provides an extendible  library for machine learning called [Flux](https://fluxml.ai). The implementation of the CNN described before  is as follows:

```julia
testCNN() = Chain(
  Conv((5, 5), 3 => 16, relu, pad=(2, 2), stride=(1, 1)),
  MaxPool((2,2), stride=(2, 2)),
  Conv((5, 5), 16 => 20, relu, pad=(2, 2), stride=(1, 1)),
  MaxPool((2,2), stride=(2, 2)),
  Conv((5, 5), 20 => 20, relu, pad=(2, 2), stride=(1, 1)),
  MaxPool((2,2), stride=(2, 2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(320, 10),
  softmax 
)
```

In this work 60000 was divided into 40000 for training and 10000 for testing and the remaining 10000 images were used to validate the model.


# Experiments on Dummy Images

Different images where downloaded from the internet and the results are showed in following figures.
![Titanic ship.](imgs/titanic.jpg.png){ width=80% }

![Delorean car.](imgs/delorean.jpg.png){ width=80% }


![Cybertruck.](imgs/cyber.jpg.png){ width=80% }

![A bird.](imgs/bird.jpg.png){ width=80% }



# Conclusions



